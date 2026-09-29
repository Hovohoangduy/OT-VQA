"""Offline integration checks for spatial vision features and pooled text."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import (BertConfig, BertModel, BertTokenizer,
                          MobileViTImageProcessor, MobileViTV2Config,
                          MobileViTV2Model)

from model.vqa_model import VQAModel


class EmbeddingBackboneTests(unittest.TestCase):
    def test_mobilevitv2_spatial_tokens_and_text_pooling(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = root / 'text'
            text.mkdir()
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'red', 'what', 'color', '?']
            (text / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
            BertTokenizer(vocab=str(text / 'vocab.txt')).save_pretrained(text)
            BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16,
                                 num_hidden_layers=1, num_attention_heads=4,
                                 intermediate_size=32)).save_pretrained(text)

            image = root / 'image'
            MobileViTV2Model(MobileViTV2Config(
                width_multiplier=0.25, image_size=64,
            )).save_pretrained(image)
            MobileViTImageProcessor(
                size={'height': 64, 'width': 64},
                crop_size={'height': 64, 'width': 64},
                do_resize=False, do_center_crop=False,
            ).save_pretrained(image)

            model = VQAModel(text_model=str(text), image_model=str(image),
                             output_size=16, d_model=16, ffn_hidden=32,
                             num_layers=1).eval()
            pixels = torch.rand(2, 3, 64, 64)
            features, _ = model.image_model(pixels)
            processed = model.image_model.process(
                images=pixels, do_rescale=False, return_tensors='pt',
            )['pixel_values']
            expected_map = model.image_model.model(pixel_values=processed).last_hidden_state
            self.assertEqual(features.shape, (2, 4, model.image_model.output_dim))
            torch.testing.assert_close(features, expected_map.flatten(2).transpose(1, 2))

            model.question_encoder.use_mean_pooling = True
            states, blocked, _, pooled = model.question_encoder.encode_tokens(
                ['what color ?', 'red'],
            )
            # The Sentence Transformers pool includes CLS and SEP but excludes PAD.
            attention = model.question_encoder.tokenizer(
                ['what color ?', 'red'], return_tensors='pt', padding=True,
            )['attention_mask'].unsqueeze(-1)
            expected_pool = (states * attention).sum(1) / attention.sum(1)
            torch.testing.assert_close(pooled,
                                       torch.nn.functional.normalize(expected_pool, dim=-1))
            with patch.object(model.ot_fusion, 'forward', wraps=model.ot_fusion.forward) as fusion:
                memory, memory_blocked = model.encode(pixels, ['what color ?', 'red'])
            self.assertEqual(fusion.call_args.args[0].shape[1], 4)
            self.assertEqual(memory.shape[0], 2)
            self.assertEqual(memory_blocked.shape, memory.shape[:2])
            logits, targets = model(pixels, ['what color ?', 'red'], ['red', 'red'],
                                    max_len=5)
            self.assertEqual(logits.shape, (2, 4, len(vocab)))
            self.assertEqual(targets.shape, (2, 4))


if __name__ == '__main__':
    unittest.main()
