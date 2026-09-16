"""Offline logic regressions using small real Transformers models, not hub downloads."""
import importlib.util
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
from transformers import BertConfig, BertModel, BertTokenizer, DeiTConfig, DeiTModel, DeiTImageProcessor

from configs.config import Config
from model.vqa_model import VQAModel
from model.decoder_model import MultiHeadAttention, MultiHeadCrossAttention, scaled_dot_product
from model.sans import StackAttention
from utils.data_processing import process_dataframe
from utils.ViTextVQA_dataset import ViTextVQA_Dataset
from utils.metrics import compute_em_and_f1
from utils.json_to_csv import convert_json_folder
from utils.checkpoint import load_model
from train import train
from test import evaluation


def load_source(name, filename):
    import sys
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
        cls.alternatives = load_source('openvi_logic', 'model/re-implement_model/OpenviVQA_re-implement.py')
        cls.text_alternatives = load_source('vitext_logic', 'model/re-implement_model/ViTextVQA_re-implement.py')

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
        self.assertIsNotNone(model.ques_model.lstm.weight_ih_l0.grad)
        self.assertIsNotNone(model.ans_model.phobert_embed.word_embeddings.weight.grad)
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
        torch.testing.assert_close(actual, direct)
        # White pixels normalized using DeiT mean/std are approximately +1.
        self.assertGreater(expected['pixel_values'].mean().item(), 0.9)

    def test_generation_prefix_eos_and_per_sample_padding(self):
        model = self.make_model()
        seen = []
        def decode(ids, memory):
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
        frame = process_dataframe(frame, 'en')
        loader = DataLoader(ViTextVQA_Dataset(frame, Config.transforms, self.root), batch_size=2)
        criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        losses, _, _ = train(model, loader, 1, optimizer, scheduler, criterion)
        self.assertEqual(len(losses), 2)
        with patch.object(model, 'generate', side_effect=lambda images, questions, ids: torch.tensor([[5, 3]] * len(questions))) as generation:
            loss, em, f1 = evaluation(model, loader, criterion)
        self.assertEqual(generation.call_count, 2)
        self.assertEqual((em, f1), (1.0, 1.0))
        self.assertGreater(loss, 0)

    def test_checkpoint_round_trip_and_legacy_rejection(self):
        model = self.make_model().eval()
        path = self.root / 'checkpoint.pt'
        torch.save({'format_version': 2, 'model_state_dict': model.state_dict(),
                    'text_model': str(self.text), 'image_model': str(self.visual), 'language': 'en', 'model_config': model.model_config}, path)
        restored, language = load_model(path, torch.device('cpu'))
        self.assertEqual(language, 'en')
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        torch.save(model.state_dict(), path)
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            load_model(path, torch.device('cpu'))

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

    def test_qumlag_infers_missing_modality_masks(self):
        mod = self.alternatives
        cfg = mod.QuMLAGConfig(vocab_size=12, image_feature_dim=8, hidden_dim=16, num_heads=4,
                              num_sa_layers=1, num_ga_layers=1, num_decoder_layers=1, ff_dim=32,
                              dropout=0, max_answer_len=4)
        model = mod.QuMLAG(cfg).eval()
        question = torch.tensor([[0, 5, 1]])
        images = torch.randn(1, 2, 8)
        _, mask = model.encode(question, images, question.eq(1), None)
        self.assertEqual(mask.shape, (1, 5))
        self.assertTrue(mask[0, 2])
        output = model(question, images, max_decode_len=4)
        self.assertEqual(output['generated_ids'].size(0), 1)
        with self.assertRaises(ValueError):
            model(question, images, max_decode_len=5)

    def make_m4c(self):
        mod = self.alternatives
        cfg = mod.M4CConfig(vocab_size=12, d_model=16, num_heads=4, object_feat_dim=8,
                            ocr_det_feat_dim=8, ocr_rec_feat_dim=4, ocr_fasttext_dim=4,
                            num_mmt_layers=2, num_question_layers=1, max_answer_len=4,
                            dropout=0, pretrained_bert=False)
        return mod.M4C(cfg).eval()

    def test_m4c_causal_mask_blocks_indirect_leakage(self):
        model = self.make_m4c()
        mask = model.build_autoregressive_joint_mask(2, 2, 3, 4, torch.device('cpu'))
        self.assertTrue(mask[:7, 7:].all())
        self.assertTrue(mask[7, 8:].all())
        self.assertFalse(mask[8, :9].any())
        obj, ocr, question = torch.randn(1, 2, 16), torch.randn(1, 2, 16), torch.randn(1, 3, 16)
        pad2, pad3 = torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool)
        first, ap = model.encode_answer_tokens(torch.tensor([[0, 5, 6, 2]]), ocr)
        second, _ = model.encode_answer_tokens(torch.tensor([[0, 5, 10, 11]]), ocr)
        with torch.no_grad():
            out1, _, _ = model.mmt_forward(obj, pad2, ocr, pad2, question, pad3, first, ap)
            out2, _, _ = model.mmt_forward(obj, pad2, ocr, pad2, question, pad3, second, ap)
        torch.testing.assert_close(out1[:, :2], out2[:, :2])

    def test_m4c_can_generate_and_reembed_ocr_pointers(self):
        model = self.make_m4c()
        args = dict(question_token_ids=torch.tensor([[0, 5, 2]]), obj_features=torch.randn(1, 2, 8),
                    obj_boxes=torch.rand(1, 2, 4), ocr_det_features=torch.randn(1, 2, 8),
                    ocr_rec_features=torch.randn(1, 2, 4), ocr_fasttext_features=torch.randn(1, 2, 4),
                    ocr_boxes=torch.rand(1, 2, 4))
        steps = []
        def scores(answer, ocr, pad):
            steps.append(answer.size(1))
            out = torch.zeros(1, answer.size(1), 14)
            out[:, -1, 12 if answer.size(1) == 1 else 2] = 10
            return out
        with patch.object(model, 'compute_scores', side_effect=scores):
            result = model(**args)
        self.assertEqual(result['generated_ids'].tolist(), [[12, 2]])
        self.assertEqual(result['scores'].shape, (1, 2, 14))
        train_out = model(**args, answer_prev_ids=torch.tensor([[0, 12, 2]]))
        self.assertEqual(train_out['scores'].shape, (1, 3, 14))

    def test_mlpag_accepts_extended_teacher_forcing_ids(self):
        mod = self.alternatives
        cfg = mod.MLPAGConfig(vocab_size=12, image_feature_dim=8, hidden_dim=16, num_heads=4,
                              num_decoder_layers=1, ff_dim=32, dropout=0, max_answer_len=4)
        model = mod.MLPAG(cfg).eval()
        scene = torch.tensor([[5, 6]])
        ids = torch.tensor([[0, 12, 1]])
        self.assertEqual(model.map_extended_to_vocab_ids(ids, scene).tolist(), [[0, 5, 1]])
        out = model(torch.tensor([[0, 5, 2]]), scene, torch.randn(1, 2, 8), decoder_input_ids=ids)
        self.assertEqual(out['scores'].shape, (1, 3, 14))

    def test_vitext_generation_length_validation(self):
        mod = self.text_alternatives
        cfg = mod.TextVQAConfig(vocab_size=12, visual_feature_dim=8, token_embed_dim=8,
                               hidden_dim=16, num_heads=4, num_encoder_layers=1,
                               num_decoder_layers=1, ff_dim=32, dropout=0, max_answer_len=4)
        for model_type in (mod.PreSTUModel, mod.SaLModel):
            model = model_type(cfg).eval()
            out = model(torch.randn(1, 2, 8), torch.tensor([[1, 5, 2]]),
                        torch.tensor([[5, 6]]), torch.rand(1, 2, 4), max_decode_len=4)
            self.assertEqual(out['generated_ids'].size(0), 1)
            with self.assertRaises(ValueError):
                model.greedyGenerate(torch.randn(1, 2, 16), torch.zeros(1, 2, dtype=torch.bool), 5)


class DataLogicTests(unittest.TestCase):
    def test_metrics_count_repeated_words_and_empty_answers(self):
        em, f1 = compute_em_and_f1(['a a b'], ['a b b'])
        self.assertEqual(em, 0)
        self.assertAlmostEqual(f1, 2 / 3)
        self.assertEqual(compute_em_and_f1([''], ['']), (1.0, 1.0))
        self.assertEqual(compute_em_and_f1(['xin_chao'], ['xin chao']), (1.0, 1.0))
        with self.assertRaises(ValueError):
            compute_em_and_f1(['a'], [])

    def test_data_question_column_and_optional_annotations(self):
        frame = pd.DataFrame({'image': ['train/a.jpg'], 'question': [' what  color ? '], 'answer': ['red']})
        processed = process_dataframe(frame, 'en')
        self.assertEqual(processed.iloc[0]['question'], 'what color ?')
        self.assertEqual(processed.iloc[0]['anno_id'], 0)
        self.assertNotIn('anno_id', frame.columns)
        self.assertNotIn('quesion', processed.columns)
        unlabelled = process_dataframe(frame.drop(columns='answer'), 'en', require_answers=False)
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
            frame = process_dataframe(pd.read_csv(root / 'train.csv'), 'en')
            Image.new('RGB', (32, 32)).save(root / 'images' / 'train' / 'a.jpg')
            sample = ViTextVQA_Dataset(frame, Config.transforms, root / 'images')[0]
            self.assertEqual(sample[2:], ('color?', 'red'))


if __name__ == '__main__':
    unittest.main()
