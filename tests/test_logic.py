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
from transformers import BertConfig, BertModel, BertTokenizer, ViTConfig, ViTImageProcessor, ViTModel

from configs.config import Config
from model.vqa_model import VQAModel
from model.decoder_model import MultiHeadAttention, MultiHeadCrossAttention, scaled_dot_product
from model.sans import StackAttention
from utils.data_processing import process_dataframe
from utils.data_processing import preprocess_text
from utils.vqa_dataset import VQADataset, resolve_image_root
from utils.metrics import PAPER_METRICS, compute_em_and_f1, lexical_scores, score_pairs, mean_scores
from utils.json_to_csv import convert_json_folder
from utils.checkpoint import load_model, save_checkpoint
from train import train
from test import evaluation
from diagnose_training import _history_report


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
        ViTModel(ViTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                           intermediate_size=32, image_size=32, patch_size=16)).save_pretrained(cls.visual)
        ViTImageProcessor(size={'height': 32, 'width': 32}, crop_size={'height': 32, 'width': 32}).save_pretrained(cls.visual)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_model(self):
        return VQAModel(text_model=str(self.text), image_model=str(self.visual),
                        output_size=16, d_model=16, ffn_hidden=32, num_layers=2, drop_prob=0)

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
        self.assertEqual(actual.shape, (1, 5, 16))
        torch.testing.assert_close(actual, direct)
        # White pixels normalized using ViT mean/std are approximately +1.
        self.assertGreater(expected['pixel_values'].mean().item(), 0.9)

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
        with patch.object(model, 'encode', return_value=(torch.zeros(2, 1, 16), torch.zeros(2, 1, dtype=torch.bool))), patch.object(model, 'decode', side_effect=decode):
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
        predictions = []
        class ConstantScorer:
            def score(self, candidates, references):
                return None, None, torch.ones(len(candidates))
        with patch.object(model, '_generate_from_memory',
                          side_effect=lambda memory, blocked, max_len: torch.tensor([[5, 3]] * memory.size(0))) as generation, \
             patch.object(model, 'encode', wraps=model.encode) as encoding:
            result = evaluation(model, loader, criterion, predictions=predictions,
                                bert_scorer=ConstantScorer())
        self.assertEqual(generation.call_count, 2)
        self.assertEqual(encoding.call_count, 2)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions[0]['prediction'], 'red')
        self.assertEqual(result['metrics']['em'], 1.0)
        self.assertEqual(result['metrics']['token_f1'], 1.0)
        self.assertEqual(result['metrics']['bertscore_f1'], 1.0)
        self.assertEqual(set(PAPER_METRICS), set(result['metrics']))
        self.assertGreater(result['loss'], 0)

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

    def test_answer_length_is_saved_with_checkpoint(self):
        model = VQAModel(
            text_model=str(self.text), image_model=str(self.visual),
            output_size=16, d_model=16, ffn_hidden=32, num_layers=1,
            max_answer_tokens=10,
        )
        logits, targets = model(torch.rand(1, 3, 32, 32), ['what color ?'], ['red'])
        self.assertEqual(logits.shape, (1, 9, 12))
        self.assertEqual(targets.shape, (1, 9))
        path = self.root / 'long-answer-checkpoint.pt'
        save_checkpoint(path, model=model, text_model=str(self.text), image_model=str(self.visual))
        restored = load_model(path, torch.device('cpu'))
        self.assertEqual(restored.max_answer_tokens, 10)
        self.assertEqual(restored.model_config['max_answer_tokens'], 10)

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

class DataLogicTests(unittest.TestCase):
    def test_metrics_count_repeated_words_and_empty_answers(self):
        em, f1 = compute_em_and_f1(['a a b'], ['a b b'])
        self.assertEqual(em, 0)
        self.assertAlmostEqual(f1, 2 / 3)
        self.assertEqual(compute_em_and_f1([''], ['']), (1.0, 1.0))
        self.assertEqual(compute_em_and_f1(['Blue car'], ['blue   car']), (0.0, 1.0))
        with self.assertRaises(ValueError):
            compute_em_and_f1(['a'], [])

    def test_paper_lexical_metrics_and_aggregation(self):
        exact = lexical_scores('red leaf', 'red leaf')
        self.assertEqual(exact, {'em': 1.0, 'token_f1': 1.0, 'bleu_1': 1.0,
                                 'bleu_2': 1.0, 'rouge_l': 1.0})
        changed = lexical_scores('red leaf', 'leaf red')
        self.assertEqual(changed['em'], 0.0)
        self.assertEqual(changed['token_f1'], 1.0)
        self.assertEqual(changed['bleu_1'], 1.0)
        self.assertEqual(changed['bleu_2'], 0.0)
        self.assertEqual(changed['rouge_l'], 0.5)
        class ConstantScorer:
            def score(self, candidates, references):
                self.candidates, self.references = candidates, references
                return None, None, torch.tensor([0.8, 0.4])
        scorer = ConstantScorer()
        rows = score_pairs(['red leaf', 'red leaf'], ['red leaf', 'leaf red'], scorer)
        self.assertEqual(scorer.candidates, ['red leaf', 'leaf red'])
        self.assertAlmostEqual(mean_scores(rows)['bertscore_f1'], 0.6)
        empty_rows = score_pairs(['', 'leaf', ''], ['', '', 'leaf'], scorer)
        self.assertEqual([row['bertscore_f1'] for row in empty_rows], [1.0, 0.0, 0.0])

    def test_diagnostics_select_lowest_validation_loss(self):
        rows = [
            {'epoch': 1, 'val_loss': 0.4, 'val_token_f1': 0.2, 'train_f1': 0.3},
            {'epoch': 2, 'val_loss': 0.5, 'val_token_f1': 0.9, 'train_f1': 0.95},
        ]
        report = _history_report(rows)
        self.assertEqual(report['best_epoch'], 1)
        self.assertAlmostEqual(report['best_val_loss'], 0.4)

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

    def test_gqa_question_selection_keeps_first_per_image(self):
        from urllib.parse import parse_qs, urlsplit
        from utils.download_gqa import fetch_first_questions
        rows = [
            {'row': {'imageId': 'a', 'question': 'color?', 'answer': 'red'}},
            {'row': {'imageId': 'b', 'question': 'shape?', 'answer': 'round'}},
            {'row': {'imageId': 'a', 'question': 'size?', 'answer': 'big'}},
            {'row': {'imageId': 'b', 'question': 'material?', 'answer': 'wood'}},
            {'row': {'imageId': 'a', 'question': 'ignored?', 'answer': 'yes'}},
        ]
        def fetch(url, cache_dir=None):
            offset = int(parse_qs(urlsplit(url).query)['offset'][0])
            return {'rows': rows if offset == 0 else [], 'num_rows_total': len(rows)}
        with patch('utils.download_gqa.fetch_json', side_effect=fetch):
            selected = fetch_first_questions('train', {'a', 'b'}, workers=1)
        self.assertEqual(selected['a']['question'], 'color?')
        self.assertEqual(selected['b']['question'], 'shape?')


if __name__ == '__main__':
    unittest.main()
