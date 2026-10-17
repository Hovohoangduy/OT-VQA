"""CPU regression using real Transformers models, without downloaded weights."""

import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_llm import resolve_attention_implementation


class GemmaAttentionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Gemma4Config, Gemma4ForConditionalGeneration
            from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
        except ImportError as error:
            raise unittest.SkipTest("Requires PyTorch and Transformers with Gemma 4 support") from error
        cls.torch = torch
        cls.config_class = Gemma4Config
        cls.model_class = Gemma4ForConditionalGeneration
        cls.audio_class = Gemma4AudioModel

    def test_checkpoint_load_and_image_generation_with_eager_audio(self):
        torch = self.torch
        config = self.config_class(
            text_config={
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 8, "global_head_dim": 8, "max_position_embeddings": 128,
                "vocab_size_per_layer_input": 64, "hidden_size_per_layer_input": 8,
                "num_kv_shared_layers": 0, "sliding_window": 16,
                "layer_types": ["sliding_attention", "full_attention"],
                "use_bidirectional_attention": "vision",
            },
            vision_config={
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "patch_size": 2, "pooling_kernel_size": 2, "position_embedding_size": 16,
            },
            audio_config={
                "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
                "subsampling_conv_channels": [16, 8], "conv_kernel_size": 3,
                "attention_chunk_size": 4, "attention_context_left": 5,
                "attention_context_right": 0, "output_proj_dims": 32,
                "use_clipped_linears": False,
            },
            image_token_id=4, boi_token_id=5, eoi_token_id=6,
            audio_token_id=7, boa_token_id=8, eoa_token_id=9, video_token_id=10,
            attn_implementation="eager",
        )
        with tempfile.TemporaryDirectory() as directory:
            self.model_class(config).save_pretrained(directory)
            # Older Transformers versions incorrectly advertised SDPA for audio.
            # Exercise the unsupported-audio path on those versions too.
            with patch.object(self.audio_class, "_supports_sdpa", False):
                with self.assertRaisesRegex(ValueError, "Gemma4AudioModel does not support"):
                    self.model_class.from_pretrained(directory, attn_implementation="sdpa")
                attention = resolve_attention_implementation(config.to_dict(), "sdpa")
                model = self.model_class.from_pretrained(directory, attn_implementation=attention).eval()
            self.assertEqual(model.config.audio_config._attn_implementation, "eager")
            self.assertEqual(model.config.text_config._attn_implementation, "sdpa")
            self.assertEqual(model.config.vision_config._attn_implementation, "sdpa")
            with torch.inference_mode():
                output = model.generate(
                    input_ids=torch.tensor([[2, 4, 11, 12]]),
                    attention_mask=torch.ones(1, 4, dtype=torch.long),
                    mm_token_type_ids=torch.tensor([[0, 1, 0, 0]]),
                    pixel_values=torch.zeros(1, 4, 12),
                    image_position_ids=torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]),
                    max_new_tokens=1, do_sample=False,
                )
            self.assertEqual(tuple(output.shape), (1, 5))
