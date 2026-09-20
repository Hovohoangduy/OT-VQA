"""Offline logic regressions using small real Transformers models, not hub downloads."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    BertConfig, BertModel, BertTokenizer,
    DeiTConfig, DeiTImageProcessor, DeiTModel,
    ViTConfig, ViTImageProcessor, ViTModel,
)

from configs.config import Config
from model.vqa_model import VQAModel
from model.features_extraction import ImageEmbedding
from model.fusion_methods import parse_fusion_spec
from model.optimal_transport import OTConfig
from model.ot_san import OTSANConfig
from model.decoder_model import MultiHeadAttention, MultiHeadCrossAttention, scaled_dot_product
from model.sans import StackAttention
from utils.data_processing import process_dataframe
from utils.data_processing import preprocess_text
from utils.vqa_dataset import VQADataset, resolve_image_root
from utils.metrics import compute_em_and_f1
from utils.json_to_csv import convert_json_folder
from utils.checkpoint import load_model, save_checkpoint
from train import train
from test import evaluation


class ModelLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.text = cls.root / 'text'
        cls.text.mkdir()
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'red', 'blue', 'what', 'color', '?', 'a', 'b']
        (cls.text / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
        BertTokenizer(vocab=str(cls.text / 'vocab.txt')).save_pretrained(cls.text)
        BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=4, intermediate_size=32,
                             hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(cls.text)
        cls.visual = cls.root / 'visual'
        DeiTModel(DeiTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                             intermediate_size=32, image_size=32, patch_size=16)).save_pretrained(cls.visual)
        DeiTImageProcessor(size={'height': 32, 'width': 32}, crop_size={'height': 32, 'width': 32}).save_pretrained(cls.visual)
        cls.vit_visual = cls.root / 'vit-visual'
        ViTModel(ViTConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
            intermediate_size=32, image_size=32, patch_size=16,
        )).save_pretrained(cls.vit_visual)
        ViTImageProcessor(
            size={'height': 32, 'width': 32},
            crop_size={'height': 32, 'width': 32},
        ).save_pretrained(cls.vit_visual)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_model(self):
        return VQAModel(text_model=str(self.text), image_model=str(self.visual),
                        output_size=16, d_model=16, ffn_hidden=32, num_layers=2, drop_prob=0)

    def make_ot_model(self, fusion='uot'):
        method = parse_fusion_spec(fusion).method
        fusion_configs = {
            'ban': {'glimpses': 2, 'hidden_dim': 8, 'dropout': 0},
            'mutan': {'rank': 2, 'factor_dim': 8, 'dropout': 0},
            'cross_attention': {
                'layers': 1, 'heads': 4, 'ffn_hidden': 32, 'dropout': 0,
            },
            'qformer': {
                'query_tokens': 3, 'layers': 1, 'heads': 4,
                'ffn_hidden': 32, 'dropout': 0,
            },
        }
        return VQAModel(
            text_model=str(self.text), image_model=str(self.visual),
            output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
            drop_prob=0, fusion=fusion,
            ot_config=OTConfig(ot_dim=8, epsilon=0.1, max_iterations=30),
            ot_san_config=OTSANConfig(hidden_dim=8, num_layers=1, dropout=0),
            fusion_config=fusion_configs.get(method),
        )

    def test_shifted_targets_and_backward_for_single_image(self):
        model = self.make_model()
        images = torch.rand(1, 3, 32, 32)
        logits, targets = model(images, ['what color ?'], ['red'], max_len=6)
        self.assertEqual(logits.shape, (1, 5, 12))
        self.assertEqual(targets.tolist(), [[5, 3, 0, 0, 0]])
        loss = nn.functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=model.pad_token_id)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.question_encoder.lstm.weight_ih_l0.grad)
        self.assertIsNotNone(model.answer_embedding.token_embeddings.word_embeddings.weight.grad)
        self.assertIsNone(next(model.image_model.model.parameters()).grad)
        self.assertFalse(model.image_model.model.training)
        self.assertIsNot(model.san_model[0], model.san_model[1])

    def test_training_can_learn_an_answer_then_generate_without_reference(self):
        torch.manual_seed(7)
        model = self.make_model()
        # Disable dropout to make this optimization regression deterministic.
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        images = torch.rand(1, 3, 32, 32)
        history = []
        for _ in range(50):
            logits, target = model(images, ['what color ?'], ['red'], max_len=6)
            loss = nn.functional.cross_entropy(logits.transpose(1, 2), target, ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        self.assertLess(history[-1], history[0] * 0.25)
        result = model.generate(images, ['what color ?'], max_len=6)
        self.assertEqual(model.answers_from_ids(result), ['red'])

    def test_future_answer_changes_cannot_change_earlier_predictions(self):
        model = self.make_model().eval()
        memory = torch.randn(1, 1, 16)
        with torch.no_grad():
            first = model.decode(torch.tensor([[2, 5, 6, 3]]), memory)
            second = model.decode(torch.tensor([[2, 5, 10, 11]]), memory)
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_image_tokens_preserved_and_pixels_scaled_once(self):
        model = self.make_model().eval()
        images = torch.ones(1, 3, 32, 32)
        expected = model.image_model.process(images=images, do_rescale=False, return_tensors='pt')
        with torch.no_grad():
            actual, _ = model.image_model(images)
            direct = model.image_model.model(**expected).last_hidden_state
        self.assertEqual(actual.shape, (1, 6, 16))
        self.assertEqual(model.image_model.spatial_tokens(actual).shape, (1, 4, 16))
        torch.testing.assert_close(actual, direct)
        # White pixels normalized using DeiT mean/std are approximately +1.
        self.assertGreater(expected['pixel_values'].mean().item(), 0.9)

    def test_vit_visual_encoder_removes_only_cls_token(self):
        encoder = ImageEmbedding(str(self.vit_visual)).eval()
        with torch.no_grad():
            hidden, _ = encoder(torch.rand(1, 3, 32, 32))
        self.assertEqual(hidden.shape, (1, 5, 16))
        self.assertEqual(encoder.num_prefix_tokens, 1)
        self.assertEqual(encoder.spatial_tokens(hidden).shape, (1, 4, 16))

    def test_generation_prefix_eos_and_per_sample_padding(self):
        model = self.make_model()
        seen = []
        def decode(ids, memory, **kwargs):
            self.assertFalse(model.training)
            seen.append(ids.clone())
            out = torch.zeros(2, ids.size(1), 12)
            choices = [3, 5] if ids.size(1) == 1 else [6, 3]
            for row, choice in enumerate(choices):
                out[row, -1, choice] = 10
            return out
        with patch.object(model, 'encode', return_value=torch.zeros(2, 1, 16)), patch.object(model, 'decode', side_effect=decode):
            generated = model.generate(torch.zeros(2, 3, 32, 32), ['a', 'b'], max_len=6)
        self.assertEqual(generated.tolist(), [[3, 0], [5, 3]])
        self.assertEqual(seen[0].tolist(), [[2], [2]])
        self.assertEqual(seen[1].tolist(), [[2, 3], [2, 5]])
        self.assertTrue(model.training)
        self.assertEqual(model.answers_from_ids(generated), ['', 'red'])

    def test_training_and_evaluation_include_partial_batch(self):
        model = self.make_model()
        image = self.root / 'sample.jpg'
        Image.new('RGB', (32, 32), color='red').save(image)
        frame = pd.DataFrame({'image': ['sample.jpg'] * 3, 'question': ['what color ?'] * 3, 'answer': ['red'] * 3})
        frame = process_dataframe(frame)
        loader = DataLoader(VQADataset(frame, Config.transforms, self.root), batch_size=2)
        criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        output = io.StringIO()
        with redirect_stdout(output):
            losses, _, _ = train(
                model, loader, 1, optimizer, scheduler, criterion,
                epoch_offset=4, total_epochs=50,
            )
        self.assertEqual(len(losses), 2)
        self.assertIn('Epoch 5/50:', output.getvalue())
        with patch.object(model, 'generate', side_effect=lambda images, questions, ids, **kwargs: torch.tensor([[5, 3]] * len(questions))) as generation:
            loss, em, f1 = evaluation(model, loader, criterion)
        self.assertEqual(generation.call_count, 2)
        self.assertEqual((em, f1), (1.0, 1.0))
        self.assertGreater(loss, 0)

    def test_checkpoint_round_trip_and_legacy_rejection(self):
        model = self.make_model().eval()
        path = self.root / 'checkpoint.pt'
        legacy_state = {}
        for key, value in model.state_dict().items():
            if key.startswith('question_encoder.text_encoder.'):
                key = key.replace('question_encoder.text_encoder.', 'ques_model.text_encoder.', 1)
                key = key.replace('ques_model.text_encoder.', 'ques_model.legacy_encoder.', 1)
            elif key.startswith('question_encoder.lstm.'):
                key = key.replace('question_encoder.lstm.', 'ques_model.lstm.', 1)
            elif key.startswith('answer_embedding.token_embeddings.'):
                key = key.replace('answer_embedding.token_embeddings.', 'ans_model.legacy_embeddings.', 1)
            legacy_state[key] = value
        torch.save({'format_version': 2, 'model_state_dict': legacy_state,
                    'text_model': str(self.text), 'image_model': str(self.visual),
                    'model_config': model.model_config}, path)
        restored = load_model(path, torch.device('cpu'))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        torch.save(model.state_dict(), path)
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            load_model(path, torch.device('cpu'))

    def test_answer_embeddings_can_be_frozen(self):
        model = VQAModel(
            text_model=str(self.text), image_model=str(self.visual),
            output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
            freeze_answer_embeddings=True,
        )
        model.train()
        self.assertFalse(model.answer_embedding.token_embeddings.training)
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in model.answer_embedding.token_embeddings.parameters()
        ))
        self.assertTrue(model.model_config['freeze_answer_embeddings'])

    def test_text_preprocessing_is_english_and_whitespace_only(self):
        self.assertEqual(preprocess_text('  What   color is it?  '), 'What color is it?')
        with self.assertRaisesRegex(ValueError, 'English encoder'):
            self.make_vqa_model_with_text_name('vinai/phobert-base-v2')

    def make_vqa_model_with_text_name(self, text_model):
        return VQAModel(
            text_model=text_model, image_model=str(self.visual),
            output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
        )

    def test_attention_head_merge_and_cross_attention_lengths(self):
        torch.manual_seed(4)
        x = torch.randn(2, 5, 16)
        self_attn = MultiHeadAttention(16, 4)
        qkv = self_attn.qkv_layer(x).reshape(2, 5, 4, 12).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, -1)
        values, _ = scaled_dot_product(q, k, v)
        expected = self_attn.linear_layer(values.transpose(1, 2).reshape(2, 5, 16))
        torch.testing.assert_close(self_attn(x), expected)
        cross = MultiHeadCrossAttention(16, 4)
        self.assertEqual(cross(torch.randn(2, 3, 16), x).shape, (2, 5, 16))
        self.assertEqual(StackAttention(16, 8, dropout=False)(x, torch.randn(2, 1, 16)).shape, (2, 16))

    def test_ot_online_cached_path_checkpoint_and_diagnostics(self):
        model = self.make_ot_model().eval()
        images = torch.rand(2, 3, 32, 32)
        questions = ['what color ?', 'color ?']
        with torch.no_grad():
            online = model.encode(images, questions, return_diagnostics=True)
            image_features, _ = model.image_model(images)
            question_features, question_mask, _ = model.question_encoder.encode_tokens(questions)
            cached = model.encode_from_features(
                image_features.half(), question_features.half(), question_mask, True
            )
        torch.testing.assert_close(online.memory, cached.memory, atol=2e-3, rtol=2e-3)
        self.assertEqual(cached.memory_padding_mask.tolist(), question_mask.tolist())
        self.assertTrue(torch.isfinite(cached.transport.plan).all())
        generated = model.generate_from_features(
            image_features, question_features, question_mask,
            max_len=5, return_diagnostics=True,
        )
        self.assertEqual(generated.generated_ids.size(0), 2)
        path = self.root / 'ot-v3.pt'
        save_checkpoint(
            path, model=model, text_model=str(self.text), image_model=str(self.visual),
            epoch=1, global_step=2,
        )
        restored = load_model(path, torch.device('cpu'))
        self.assertEqual(restored.fusion_type, 'uot')
        with torch.no_grad():
            expected = model.generate(images, questions, max_len=5)
            actual = restored.generate(images, questions, max_len=5)
        torch.testing.assert_close(actual, expected)

    def test_ot_san_online_cached_checkpoint_gradients_and_diagnostics(self):
        torch.manual_seed(13)
        model = self.make_ot_model(fusion='uot_san').eval()
        images = torch.rand(2, 3, 32, 32)
        questions = ['what color ?', 'color ?']
        online = model.encode(images, questions, return_diagnostics=True)
        image_features, _ = model.image_model(images)
        question_features, question_mask, _ = model.question_encoder.encode_tokens(questions)
        cached = model.encode_from_features(
            image_features.half(), question_features.half(), question_mask, True
        )
        self.assertEqual(online.memory.shape[1], question_mask.shape[1] + 1)
        self.assertEqual(online.memory_padding_mask.shape[1], question_mask.shape[1] + 1)
        self.assertFalse(online.memory_padding_mask[:, 0].any())
        self.assertIsNotNone(online.ot_san)
        self.assertEqual(online.ot_san.attention_weights.shape[:2], (2, 1))
        torch.testing.assert_close(online.memory, cached.memory, atol=2e-3, rtol=2e-3)

        model.train()
        logits, targets, transport = model(
            images, questions, ['red', 'blue'], max_len=6,
            return_diagnostics=True,
        )
        loss = nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, ignore_index=model.pad_token_id
        )
        loss.backward()
        self.assertIsNotNone(transport.ot_san)
        self.assertIsNotNone(model.ot_san.gate_logit.grad)
        self.assertIsNotNone(model.ot_fusion.fusion[0].weight.grad)

        path = self.root / 'ot-san-v3.pt'
        save_checkpoint(
            path, model=model, text_model=str(self.text), image_model=str(self.visual),
            epoch=1, global_step=2,
        )
        restored = load_model(path, torch.device('cpu'))
        self.assertEqual(restored.fusion_type, 'uot_san')
        self.assertEqual(restored.model_config['ot_san_config']['hidden_dim'], 8)
        model.eval()
        with torch.no_grad():
            expected = model.generate(images, questions, max_len=5)
            actual = restored.generate(images, questions, max_len=5)
        torch.testing.assert_close(actual, expected)

    def test_balanced_ot_san_selects_balanced_transport(self):
        model = self.make_ot_model(fusion='balanced_ot_san')
        self.assertEqual(model.ot_config.transport_type, 'balanced')
        self.assertIsNotNone(model.ot_san)

    def test_new_fusion_families_support_online_and_cached_paths(self):
        images = torch.rand(2, 3, 32, 32)
        questions = ['what color ?', 'color ?']
        names = [
            'ban', 'uot_ban', 'mutan', 'uot_mutan',
            'cross_attention', 'uot_cross_attention',
            'qformer', 'uot_qformer',
        ]
        for name in names:
            with self.subTest(fusion=name):
                model = self.make_ot_model(name).eval()
                with torch.no_grad():
                    online = model.encode(images, questions, return_diagnostics=True)
                    image_features, _ = model.image_model(images)
                    question_features, question_mask, _ = (
                        model.question_encoder.encode_tokens(questions)
                    )
                    cached = model.encode_from_features(
                        image_features.half(), question_features.half(),
                        question_mask, True,
                    )
                    generated = model.generate_from_features(
                        image_features, question_features, question_mask,
                        max_len=4, return_diagnostics=True,
                    )
                torch.testing.assert_close(
                    online.memory, cached.memory, atol=2e-3, rtol=2e-3
                )
                self.assertEqual(
                    online.memory_padding_mask.shape, online.memory.shape[:2]
                )
                self.assertIsNotNone(online.fusion_output.diagnostics)
                self.assertEqual(generated.generated_ids.size(0), 2)
                self.assertEqual(
                    online.transport is not None,
                    parse_fusion_spec(name).uses_ot,
                )

    def test_new_fusion_checkpoint_restores_method_configuration(self):
        for name in ('ban', 'uot_qformer'):
            with self.subTest(fusion=name):
                model = self.make_ot_model(name).eval()
                path = self.root / f'{name}.pt'
                save_checkpoint(
                    path, model=model, text_model=str(self.text),
                    image_model=str(self.visual), epoch=1, global_step=1,
                )
                restored = load_model(path, torch.device('cpu'))
                self.assertEqual(restored.fusion_type, name)
                self.assertEqual(
                    restored.model_config['fusion_config'],
                    model.model_config['fusion_config'],
                )
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key])

    def test_uot_tiny_batch_learns_and_generates_without_reference(self):
        torch.manual_seed(8)
        model = self.make_ot_model()
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=0.02,
        )
        images = torch.rand(1, 3, 32, 32)
        first_loss = None
        for _ in range(60):
            logits, targets = model(
                images, ['what color ?'], ['red'], max_len=6
            )
            loss = nn.functional.cross_entropy(
                logits.transpose(1, 2), targets, ignore_index=model.pad_token_id
            )
            first_loss = loss.item() if first_loss is None else first_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first_loss * 0.1)
        generated = model.generate(images, ['what color ?'], max_len=6)
        self.assertEqual(model.answers_from_ids(generated), ['red'])

    def test_decoder_ignores_padded_ot_memory_tokens(self):
        model = self.make_ot_model().eval()
        ids = torch.tensor([[2, 5]])
        memory = torch.randn(1, 3, 16)
        changed = memory.clone()
        changed[:, 2] = 1000
        mask = torch.tensor([[False, False, True]])
        with torch.no_grad():
            first = model.decode(ids, memory, memory_padding_mask=mask)
            second = model.decode(ids, changed, memory_padding_mask=mask)
        torch.testing.assert_close(first, second)

    def test_vqa_model_shares_bert_between_question_and_answer_embedding(self):
        model = self.make_model()
        # AnswerEmbedding should directly reference QuestionEmbedding's BertEmbeddings
        self.assertIs(
            model.answer_embedding.token_embeddings,
            model.question_encoder.text_encoder.embeddings,
        )
        self.assertIs(
            model.answer_embedding.tokenizer,
            model.question_encoder.tokenizer,
        )

    def test_vqa_model_san_forward_from_features(self):
        model = self.make_model()
        image_features = torch.randn(2, 5, 16)
        question_features = torch.randn(2, 4, 16)
        question_padding_mask = torch.tensor([[False, False, False, True],
                                              [False, False, True, True]])
        logits, targets = model.forward_from_features(
            image_features, question_features, question_padding_mask,
            answers=['red', 'blue'], max_len=6,
        )
        self.assertEqual(logits.shape[:2], (2, 5))
        loss = nn.functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=model.pad_token_id)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

        generated = model.generate_from_features(
            image_features, question_features, question_padding_mask, max_len=6,
        )
        self.assertEqual(generated.shape, (2, 5))

    def test_vqa_model_skip_encoders(self):
        base_model = self.make_model()
        token_embeddings = base_model.question_encoder.text_encoder.embeddings
        model = VQAModel(
            text_model=str(self.text), image_model=str(self.visual),
            output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
            drop_prob=0, fusion='uot',
            ot_config=OTConfig(ot_dim=8, epsilon=0.1, max_iterations=10),
            skip_encoders=True, token_embeddings=token_embeddings,
        )
        self.assertIsNone(model.image_model.model)
        self.assertIsNone(model.question_encoder.text_encoder)
        self.assertIs(model.answer_embedding.token_embeddings, token_embeddings)

        image_features = torch.randn(2, 5, 16)
        question_features = torch.randn(2, 4, 16)
        question_padding_mask = torch.tensor([[False, False, False, True],
                                              [False, False, True, True]])
        logits, targets = model.forward_from_features(
            image_features, question_features, question_padding_mask,
            answers=['red', 'blue'], max_len=6,
        )
        self.assertEqual(logits.shape[:2], (2, 5))

    def test_ensure_feature_cache_and_reuse(self):
        from scripts.run_fusion_benchmark import ensure_feature_cache
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            img_dir = temp_path / 'images'
            img_dir.mkdir()
            img_file = img_dir / '1.jpg'
            Image.new('RGB', (32, 32), color='red').save(img_file)

            csv_data = pd.DataFrame({
                'anno_id': [1],
                'image': ['1.jpg'],
                'question': ['what color ?'],
                'answer': ['red'],
            })
            train_csv = temp_path / 'train.csv'
            dev_csv = temp_path / 'dev.csv'
            csv_data.to_csv(train_csv, index=False)
            csv_data.to_csv(dev_csv, index=False)

            cache_dir = temp_path / 'cache'

            # 1. First run: extracts features, saves embeddings.pt
            out_cache = ensure_feature_cache(
                cache_dir, train_csv, dev_csv, img_dir,
                str(self.text), str(self.visual),
                torch.device('cpu'), batch_size=1,
            )
            self.assertEqual(out_cache, cache_dir)
            self.assertTrue((cache_dir / 'embeddings.pt').is_file())
            self.assertTrue((cache_dir / 'train' / 'features.pt').is_file())
            self.assertTrue((cache_dir / 'dev' / 'features.pt').is_file())

            # 2. Second run: uses existing cache; patch ImageEmbedding to verify encoders are NOT instantiated
            with patch('scripts.run_fusion_benchmark.ImageEmbedding', side_effect=AssertionError("Should not load ImageEmbedding")):
                with patch('scripts.run_fusion_benchmark.QuestionEmbedding', side_effect=AssertionError("Should not load QuestionEmbedding")):
                    reused_cache = ensure_feature_cache(
                        cache_dir, train_csv, dev_csv, img_dir,
                        str(self.text), str(self.visual),
                        torch.device('cpu'), batch_size=1,
                    )
                    self.assertEqual(reused_cache, cache_dir)

            # 3. Model init with skip_encoders and cached embeddings_path loads without BERT
            model = VQAModel(
                text_model=str(self.text), image_model=str(self.visual),
                output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
                drop_prob=0, fusion='san',
                skip_encoders=True, embeddings_path=cache_dir / 'embeddings.pt',
            )
            self.assertIsNone(model.question_encoder.text_encoder)
            self.assertIsNone(model.image_model.model)
            self.assertIsNotNone(model.answer_embedding.token_embeddings)

class DataLogicTests(unittest.TestCase):
    def test_metrics_count_repeated_words_and_empty_answers(self):
        em, f1 = compute_em_and_f1(['a a b'], ['a b b'])
        self.assertEqual(em, 0)
        self.assertAlmostEqual(f1, 2 / 3)
        self.assertEqual(compute_em_and_f1([''], ['']), (1.0, 1.0))
        self.assertEqual(compute_em_and_f1(['Blue car'], ['blue   car']), (1.0, 1.0))
        with self.assertRaises(ValueError):
            compute_em_and_f1(['a'], [])

    def test_data_question_column_and_optional_annotations(self):
        frame = pd.DataFrame({'image': ['train/a.jpg'], 'question': [' what  color ? '], 'answer': ['red']})
        processed = process_dataframe(frame)
        self.assertEqual(processed.iloc[0]['question'], 'what color ?')
        self.assertEqual(processed.iloc[0]['anno_id'], 0)
        self.assertNotIn('anno_id', frame.columns)
        self.assertNotIn('quesion', processed.columns)
        unlabelled = process_dataframe(frame.drop(columns='answer'), require_answers=False)
        self.assertEqual(unlabelled.iloc[0]['answer'], '')

    def test_json_alternative_answers_are_not_concatenated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'qa.json').write_text(json.dumps({'annotations': [
                {'id': 1, 'image_id': 'a', 'question': 'what?', 'answers': ['red', 'blue']}]}))
            convert_json_folder(root, root / 'csv')
            result = pd.read_csv(root / 'csv' / 'qa.csv')
            self.assertEqual(result.iloc[0]['answer'], 'red')

    def test_gqa_split_paths_match_dataset_loader(self):
        from utils.download_gqa import materialize_split
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('utils.download_gqa.download_image', return_value='a.jpg'):
                rows = materialize_split('train', [{'id': 'a'}], {'a': {'question': 'color?', 'answer': 'red'}}, root, 1)
            self.assertEqual(rows[0]['image'], 'train/a.jpg')
            self.assertEqual(rows[0]['anno_id'], 'a')
            frame = process_dataframe(pd.read_csv(root / 'train.csv'))
            Image.new('RGB', (32, 32)).save(root / 'images' / 'train' / 'a.jpg')
            sample = VQADataset(frame, Config.transforms, root / 'images')[0]
            self.assertEqual(sample[2:], ('color?', 'red'))

    def test_image_root_resolves_bare_and_split_relative_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'images'
            (root / 'train').mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(root / 'train' / 'a.jpg')

            bare = pd.DataFrame({'image': ['a.jpg']})
            self.assertEqual(resolve_image_root(bare, root, 'train'), root / 'train')

            relative = pd.DataFrame({'image': ['train/a.jpg']})
            self.assertEqual(resolve_image_root(relative, root, 'train'), root)

            explicit = root / 'custom'
            self.assertEqual(
                resolve_image_root(bare, root, 'train', override=explicit), explicit
            )


if __name__ == '__main__':
    unittest.main()
