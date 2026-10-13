"""Numerical checks for Keye's T4 attention adaptation, without model weights."""

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.keye_sdpa import adapt_keye_model_source, keye_vision_sdpa_forward, stage_keye_sdpa

try:
    import torch
except ImportError:
    torch = None


# A small source fixture exercises the same upstream entry points and import guard.
SOURCE = '''from transformers.modeling_utils import PreTrainedModel, sdpa_attention_forward
from transformers.activations import GELUActivation, ACT2FN, PytorchGELUTanh
assert is_flash_attn_2_available()
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None
class KeyeVL1_5VisionAttention:
    def forward(self, hidden_states, **kwargs):
        raise AssertionError("unadapted forward")
def apply_rotary_pos_emb_flashatt(q, k, cos, sin):
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
'''


class KeyeSourceTests(unittest.TestCase):
    def test_patch_removes_flash_dependency_and_preserves_other_code(self):
        adapted = adapt_keye_model_source(SOURCE)
        tree = ast.parse(adapted)
        self.assertFalse(any(isinstance(node, ast.Assert) for node in tree.body))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertNotIn("flash_attn", imports)
        self.assertNotIn("flash_attn.layers.rotary", imports)
        self.assertIn("transformers.integrations.sdpa_attention", imports)
        self.assertNotIn("PytorchGELUTanh", adapted)
        namespace = {}
        attention = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        exec(compile(ast.Module(body=[attention], type_ignores=[]), "fixture", "exec"), namespace)
        self.assertTrue(callable(namespace["KeyeVL1_5VisionAttention"].forward))

    def test_unknown_code_is_rejected_and_source_model_is_never_changed(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized Keye"):
            adapt_keye_model_source("class OtherModel: pass\n")
        with self.assertRaisesRegex(ValueError, "removed PytorchGELUTanh"):
            adapt_keye_model_source(SOURCE + "activation = PytorchGELUTanh()\n")
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            name = "modeling_keye_vl_1_5.py"
            (directory / name).write_text(SOURCE)
            (directory / "model.safetensors").write_bytes(b"original weights")
            (directory / "config.json").write_text("{}")
            temporary, staged, identity = stage_keye_sdpa(directory)
            self.addCleanup(temporary.cleanup)
            self.assertEqual((directory / name).read_text(), SOURCE)
            self.assertTrue((staged / "model.safetensors").is_symlink())
            self.assertEqual((staged / "model.safetensors").read_bytes(), b"original weights")
            self.assertFalse((staged / name).is_symlink())
            self.assertNotEqual(identity["upstream_model_code_sha256"], identity["adapted_model_code_sha256"])
            other, _, same_identity = stage_keye_sdpa(directory)
            self.addCleanup(other.cleanup)
            self.assertEqual(identity, same_identity)


@unittest.skipIf(torch is None, "Numerical SDPA checks require PyTorch")
class KeyeAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.layer = SimpleNamespace(
            num_heads=2, head_dim=4, scale=0.5, dropout=0.0, training=False,
            q_proj=torch.nn.Linear(8, 8), k_proj=torch.nn.Linear(8, 8),
            v_proj=torch.nn.Linear(8, 8), out_proj=torch.nn.Linear(8, 8),
            q_norm=torch.nn.LayerNorm(4), k_norm=torch.nn.LayerNorm(4),
        )
        self.hidden = torch.randn(1, 5, 8)
        angles = torch.randn(5, 2).repeat(1, 2)
        self.rope = angles.cos(), angles.sin()

    def reference(self, boundaries, mask=None):
        layer = self.layer
        query = layer.q_norm(layer.q_proj(self.hidden).view(1, 5, 2, 4))
        key = layer.k_norm(layer.k_proj(self.hidden).view(1, 5, 2, 4))
        value = layer.v_proj(self.hidden).view(1, 5, 2, 4).transpose(1, 2)
        # Express the non-interleaved rotary transform as pairs, independently of
        # the adapter's rotate-half implementation.
        cos, sin = (item[:, :2].unsqueeze(0).unsqueeze(2) for item in self.rope)
        def rotate(tensor):
            a, b = tensor[..., :2], tensor[..., 2:]
            return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).transpose(1, 2)
        query, key = rotate(query), rotate(key)
        pieces = []
        for start, end in zip(boundaries, boundaries[1:]):
            scores = query[:, :, start:end] @ key[:, :, start:end].transpose(-1, -2) * 0.5
            if mask is not None:
                scores = scores + mask[..., start:end, start:end]
            pieces.append(scores.softmax(-1) @ value[:, :, start:end])
        output = torch.cat(pieces, dim=2).transpose(1, 2).reshape(1, 5, 8)
        return layer.out_proj(output)

    def test_packed_sdpa_matches_reference_with_rotary_and_qk_normalization(self):
        output, weights = keye_vision_sdpa_forward(
            self.layer, self.hidden, cu_seqlens=torch.tensor([0, 2, 5]), rope_emb=self.rope)
        torch.testing.assert_close(output, self.reference([0, 2, 5]))
        self.assertIsNone(weights)
        # Changing the second image cannot influence attention in the first image.
        changed = self.hidden.clone()
        changed[:, 2:] += 100
        isolated, _ = keye_vision_sdpa_forward(
            self.layer, changed, cu_seqlens=torch.tensor([0, 2, 5]), rope_emb=self.rope)
        torch.testing.assert_close(output[:, :2], isolated[:, :2])

    def test_masked_and_unpacked_attention_match_reference(self):
        mask = torch.zeros(1, 1, 5, 5)
        mask[..., 0, 1] = float("-inf")
        output, _ = keye_vision_sdpa_forward(
            self.layer, self.hidden, attention_mask=mask,
            cu_seqlens=torch.tensor([0, 2, 5]), rope_emb=self.rope)
        torch.testing.assert_close(output, self.reference([0, 2, 5], mask))
        output, _ = keye_vision_sdpa_forward(self.layer, self.hidden, rope_emb=self.rope)
        torch.testing.assert_close(output, self.reference([0, 5]))

    def test_invalid_boundaries_and_attention_weight_requests_fail(self):
        for boundaries in ([1, 5], [0, 4], [0, 3, 2, 5], [0, 0, 5]):
            with self.subTest(boundaries=boundaries), self.assertRaisesRegex(ValueError, "boundaries"):
                keye_vision_sdpa_forward(self.layer, self.hidden, cu_seqlens=torch.tensor(boundaries))
        with self.assertRaisesRegex(ValueError, "attention weights"):
            keye_vision_sdpa_forward(self.layer, self.hidden, output_attentions=True)

    def test_fp16_keeps_output_dtype_and_matches_float32_reference(self):
        reference = self.reference([0, 2, 5])
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "q_norm", "k_norm"):
            getattr(self.layer, name).half()
        output, _ = keye_vision_sdpa_forward(
            self.layer, self.hidden.half(), cu_seqlens=torch.tensor([0, 2, 5]), rope_emb=self.rope)
        self.assertEqual(output.dtype, torch.float16)
        torch.testing.assert_close(output.float(), reference, rtol=0.01, atol=0.001)


if __name__ == "__main__":
    unittest.main()
