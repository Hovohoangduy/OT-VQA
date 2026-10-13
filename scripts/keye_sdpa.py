"""Stage Keye-VL-1.5 custom code with a FlashAttention-free vision path.

Weights and preprocessing are unchanged. The staged module uses upstream's
PyTorch rotary embeddings and SDPA separately for each packed image sequence.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import tempfile
from pathlib import Path


def keye_vision_sdpa_forward(self, hidden_states, attention_mask=None,
                            output_attentions=False, cu_seqlens=None, rope_emb=None):
    # This function is embedded in the upstream module, whose globals supply torch.
    import torch

    if output_attentions:
        raise ValueError("Keye SDPA evaluation does not return attention weights.")
    batch, length, width = hidden_states.shape
    query = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim)
    key = self.k_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim)
    value = self.v_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim)
    if self.q_norm is not None and self.k_norm is not None:
        query, key = self.q_norm(query), self.k_norm(key)
    if rope_emb is not None:
        # Match FlashAttention's non-interleaved rotation, computed in float32.
        cos, sin = (item.unsqueeze(-2).float() for item in rope_emb)

        def rotate(tensor):
            first, second = tensor.float().chunk(2, dim=-1)
            half = torch.cat((-second, first), dim=-1)
            return (tensor.float() * cos + half * sin).to(tensor.dtype)

        query, key = rotate(query), rotate(key)
    query, key, value = (item.transpose(1, 2) for item in (query, key, value))
    boundaries = [0, length]
    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        if (batch != 1 or len(boundaries) < 2 or boundaries[0] != 0
                or boundaries[-1] != length
                or any(end <= start for start, end in zip(boundaries, boundaries[1:]))):
            raise ValueError("Invalid packed image boundaries for Keye SDPA.")
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        mask = attention_mask
        if mask is not None:
            mask = mask[..., start:end, start:end]
        outputs.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, start:end].contiguous(), key[:, :, start:end].contiguous(),
            value[:, :, start:end].contiguous(), attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False, scale=self.scale,
        ))
    output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, length, width)
    return self.out_proj(output), None


def adapt_keye_model_source(source: str) -> str:
    """Patch only recognized entry points; reject incompatible upstream revisions."""
    tree = ast.parse(source)
    replacements = []
    found = set()
    for node in tree.body:
        if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Name)
                and node.test.func.id == "is_flash_attn_2_available"):
            replacements.append((node.lineno, node.end_lineno, "# FlashAttention is optional for this SDPA adaptation.\n"))
            found.add("assert")
        elif (isinstance(node, ast.If) and isinstance(node.test, ast.Call)
              and isinstance(node.test.func, ast.Name)
              and node.test.func.id == "is_flash_attn_2_available"):
            replacements.append((node.lineno, node.end_lineno,
                                 "flash_attn_varlen_func = None\napply_rotary_emb = None\n"))
            found.add("imports")
        elif isinstance(node, ast.FunctionDef) and node.name == "apply_rotary_pos_emb_flashatt":
            replacements.append((node.lineno, node.end_lineno,
                                 "def apply_rotary_pos_emb_flashatt(q, k, cos, sin):\n"
                                 "    return apply_rotary_pos_emb_vision(q, k, cos, sin)\n"))
            found.add("rotary")
        elif isinstance(node, ast.ClassDef) and node.name == "KeyeVL1_5VisionAttention":
            forward = next((item for item in node.body if isinstance(item, ast.FunctionDef)
                            and item.name == "forward"), None)
            if forward is not None:
                implementation = inspect.getsource(keye_vision_sdpa_forward).replace(
                    "def keye_vision_sdpa_forward(", "def forward(", 1)
                replacements.append((forward.lineno, forward.end_lineno,
                                     "".join("    " + line for line in implementation.splitlines(True))))
                found.add("vision")
    if found != {"assert", "imports", "rotary", "vision"}:
        raise ValueError("Unrecognized Keye-VL-1.5 model code; cannot apply the SDPA adaptation safely. "
                         "Use the supported upstream revision b4d413e4f42a3999af1f5ba0e281c7d51d6b93de.")
    # Normalize older imports for the notebook's pinned Transformers version.
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        removed = {"transformers.modeling_utils": "sdpa_attention_forward",
                   "transformers.activations": "PytorchGELUTanh"}.get(node.module)
        names = [name for name in node.names if name.name != removed]
        if len(names) != len(node.names):
            imports = "from " + node.module + " import " + ", ".join(
                name.name + (" as " + name.asname if name.asname else "") for name in names)
            if not names:
                imports = ""
            if removed == "sdpa_attention_forward":
                imports += "\nfrom transformers.integrations.sdpa_attention import sdpa_attention_forward"
            elif any(isinstance(item, ast.Name) and item.id == "PytorchGELUTanh" for item in ast.walk(tree)):
                raise ValueError("Unrecognized Keye code uses the removed PytorchGELUTanh activation.")
            replacements.append((node.lineno, node.end_lineno, imports + "\n"))
    lines = source.splitlines(True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start - 1:end] = [replacement]
    adapted = "".join(lines)
    compile(adapted, "modeling_keye_vl_1_5.py", "exec")
    return adapted


def stage_keye_sdpa(directory: Path):
    """Return a temporary local model view and stable hashes for resume checks.

    Keep the returned TemporaryDirectory alive until model loading is finished.
    Symlinks reuse weight shards without modifying the original HF cache or inputs.
    """
    name = "modeling_keye_vl_1_5.py"
    source = (directory / name).read_text(encoding="utf-8")
    adapted = adapt_keye_model_source(source)
    temporary = tempfile.TemporaryDirectory(prefix="ot_vqa_keye_sdpa_")
    staged = Path(temporary.name)
    try:
        for path in directory.iterdir():
            if path.name != name:
                (staged / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
        (staged / name).write_text(adapted, encoding="utf-8")
    except BaseException:
        temporary.cleanup()
        raise
    identity = {
        "attention_adapter": "keye-vl-1.5-sdpa-v1",
        "upstream_model_code_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapted_model_code_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
    }
    return temporary, staged, identity
