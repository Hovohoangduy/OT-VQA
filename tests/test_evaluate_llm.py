"""Regression tests for checkpoint inspection and architecture-specific loading.

Run without GPU/weights: python -m unittest discover -s tests -v
"""

import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import evaluate_llm as evaluator


def onevision_config():
    # The real checkpoint omits the top-level model_type.
    return {"architectures": ["LLaVAOneVision1_5_ForConditionalGeneration"],
            "auto_map": dict(evaluator.ONEVISION_AUTO_MAP)}


class CheckpointTests(unittest.TestCase):
    def test_missing_model_type_uses_original_auto_map(self):
        original = onevision_config()
        config = evaluator.normalize_model_config(original)
        self.assertEqual(config["model_type"], evaluator.ONEVISION_MODEL_TYPE)
        self.assertNotIn("model_type", original)
        self.assertIn("transformers==4.57.1", evaluator.runtime_requirements(config))

    def test_unknown_custom_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported vision architecture"):
            evaluator.normalize_model_config({"auto_map": {"AutoModel": "unknown.Model"}})
        altered = onevision_config()
        altered["auto_map"]["AutoConfig"] = "another_repo--configuration.Config"
        altered["model_type"] = evaluator.ONEVISION_MODEL_TYPE
        with self.assertRaisesRegex(ValueError, "original AutoConfig"):
            evaluator.normalize_model_config(altered)

    def test_existing_native_architectures_keep_their_loaders(self):
        for model_type in ("qwen3_vl", "internvl", "mistral3", "glm4v", "gemma4"):
            with self.subTest(model_type=model_type):
                config = {"model_type": model_type}
                self.assertEqual(evaluator.normalize_model_config(config), config)
                self.assertIn("transformers>=5.5,<6", evaluator.runtime_requirements(config))

    def test_local_onevision_inspection_and_python_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(onevision_config()))
            (root / "model.safetensors").write_bytes(b"test weights")
            code = root / "configuration_llavaonevision1_5.py"
            code.write_text("# original code\n")
            source, config = evaluator.inspect_model(str(root))
            self.assertEqual(config["model_type"], evaluator.ONEVISION_MODEL_TYPE)
            _, identity = evaluator.resolve_model(source, None)
            self.assertIn(code.name, identity["files_sha256"])
            code.write_text("# changed code\n")
            _, changed_identity = evaluator.resolve_model(source, None)
            self.assertNotEqual(identity, changed_identity)

    def test_hub_inspection_and_download_use_the_same_commit(self):
        hub = types.ModuleType("huggingface_hub")
        api = MagicMock()
        api.model_info.return_value.sha = "a" * 40
        names = ["config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors",
                 "configuration_llavaonevision1_5.py", "modeling_llavaonevision1_5.py",
                 "tokenizer.json", "chat_template.jinja", "weights.gguf", "consolidated.safetensors"]
        api.list_repo_files.return_value = names
        hub.HfApi = MagicMock(return_value=api)
        hub.hf_hub_download = MagicMock()
        hub.snapshot_download = MagicMock(return_value="/tmp/snapshot")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, huggingface_hub=hub):
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps(onevision_config()))
            hub.hf_hub_download.return_value = str(config_path)
            source, config = evaluator.inspect_model("llava-onevision-1.5-8b", cache_dir=Path(temporary))
            self.assertEqual(config["model_type"], evaluator.ONEVISION_MODEL_TYPE)
            _, identity = evaluator.resolve_model(source, Path(temporary))
            self.assertEqual(identity["revision"], "a" * 40)
            for call in (hub.hf_hub_download.call_args, hub.snapshot_download.call_args):
                self.assertEqual(call.kwargs["revision"], identity["revision"])
                self.assertEqual(call.kwargs["cache_dir"], temporary)
            assets = hub.snapshot_download.call_args.kwargs["allow_patterns"]
            self.assertIn("modeling_llavaonevision1_5.py", assets)
            self.assertIn("configuration_llavaonevision1_5.py", assets)
            self.assertIn("model-00001-of-00001.safetensors", assets)
            self.assertNotIn("weights.gguf", assets)
            self.assertNotIn("consolidated.safetensors", assets)

    def test_gguf_only_repository_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError, "native Transformers weights"):
            evaluator.validate_model_files(["config.json", "model.gguf"])


class RuntimeTests(unittest.TestCase):
    def modules(self, version="4.57.1", capability=7):
        torch = MagicMock(__version__="2.10.0")
        torch.cuda.device_count.return_value = 2
        torch.cuda.get_device_capability.return_value = (capability, 5)
        transformers = types.ModuleType("transformers")
        transformers.__version__ = version
        transformers.AutoModelForCausalLM = MagicMock()
        transformers.Qwen3VLForConditionalGeneration = MagicMock()
        return {"torch": torch, "accelerate": MagicMock(), "transformers": transformers}

    def test_t4_uses_float16_and_custom_auto_loader(self):
        modules = self.modules()
        with patch.dict(sys.modules, modules):
            plan = evaluator.validate_runtime(evaluator.normalize_model_config(onevision_config()), "auto")
        self.assertEqual(plan[3:5], (2, "float16"))
        self.assertIs(plan[5], modules["transformers"].AutoModelForCausalLM)

    def test_ampere_uses_bfloat16(self):
        with patch.dict(sys.modules, self.modules(capability=8)):
            plan = evaluator.validate_runtime(evaluator.normalize_model_config(onevision_config()), "auto")
        self.assertEqual(plan[4], "bfloat16")

    def test_t4_rejects_explicit_bfloat16(self):
        with patch.dict(sys.modules, self.modules()):
            with self.assertRaisesRegex(ValueError, "select float16 for T4"):
                evaluator.validate_runtime(evaluator.normalize_model_config(onevision_config()), "bfloat16")

    def test_onevision_rejects_transformers5_before_weight_download(self):
        with patch.dict(sys.modules, self.modules(version="5.5.0")):
            with self.assertRaisesRegex(RuntimeError, "needs transformers==4.57.1"):
                evaluator.validate_runtime(evaluator.normalize_model_config(onevision_config()), "auto")

    def test_native_models_use_transformers5(self):
        modules = self.modules(version="5.5.0")
        with patch.dict(sys.modules, modules):
            plan = evaluator.validate_runtime({"model_type": "qwen3_vl"}, "auto")
        self.assertIs(plan[5], modules["transformers"].Qwen3VLForConditionalGeneration)

    def test_custom_code_enabled_only_for_onevision(self):
        for model_type in (evaluator.ONEVISION_MODEL_TYPE, "qwen3_vl"):
            with self.subTest(model_type=model_type):
                modules = self.modules()
                transformers = modules["transformers"]
                transformers.AutoProcessor = MagicMock()
                loader = MagicMock()
                model = loader.from_pretrained.return_value.eval.return_value
                model.hf_device_map = {"model.visual": 0, "lm_head": 1}
                plan = (modules["torch"], MagicMock(__version__="1.0"), transformers,
                        2, "float16", loader, {})
                with patch.object(evaluator, "validate_runtime", return_value=plan):
                    evaluator.TransformersClient(Path("/tmp/checkpoint"), "test", {"model_type": model_type})
                for call in (loader.from_pretrained.call_args, transformers.AutoProcessor.from_pretrained.call_args):
                    self.assertEqual(call.kwargs["trust_remote_code"], model_type == evaluator.ONEVISION_MODEL_TYPE)
                    self.assertTrue(call.kwargs["local_files_only"])

    def test_gemma_audio_attention_reaches_loader_and_run_metadata(self):
        for attention in ("sdpa", "eager"):
            with self.subTest(attention=attention):
                modules = self.modules(version="5.7.0")
                transformers = modules["transformers"]
                transformers.AutoProcessor = MagicMock()
                loader = MagicMock()
                loader.from_pretrained.return_value.eval.return_value.hf_device_map = {"model": 0}
                plan = (modules["torch"], MagicMock(__version__="1.0"), transformers,
                        2, "float16", loader, {})
                config = {"model_type": "gemma4", "audio_config": {}}
                with patch.object(evaluator, "validate_runtime", return_value=plan):
                    client = evaluator.TransformersClient(Path("/tmp/checkpoint"), "gemma", config,
                                                          attention=attention)
                resolved = loader.from_pretrained.call_args.kwargs["attn_implementation"]
                if attention == "sdpa":
                    self.assertEqual(resolved["audio_config"], "eager")
                    self.assertEqual(resolved["text_config"], "sdpa")
                    self.assertEqual(resolved["vision_config"], "sdpa")
                else:
                    self.assertEqual(resolved, "eager")
                self.assertEqual(client.backend_config["attention_implementation"], resolved)

    def test_other_architectures_and_gemma_without_audio_keep_requested_attention(self):
        configs = [{"model_type": model_type} for model_type in evaluator.MODEL_CLASSES]
        configs += [{"model_type": "gemma4", "audio_config": None},
                    {"model_type": "qwen3_vl", "audio_config": {}}]
        for config in configs:
            for attention in ("sdpa", "eager"):
                with self.subTest(config=config, attention=attention):
                    self.assertEqual(evaluator.resolve_attention_implementation(config, attention), attention)


class NotebookTests(unittest.TestCase):
    def test_notebook_bundles_current_sources_and_valid_python(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/evaluate_llm_kaggle.ipynb").read_text())
        bundles = None
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]), filename=f"cell {index}")
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "BUNDLED_SOURCES"
                        for target in statement.targets):
                    bundles = ast.literal_eval(statement.value)
                    self.assertLess(index, 4, "Restore sources before dependency selection")
        self.assertIsNotNone(bundles)
        for name, bundled in bundles.items():
            self.assertEqual(bundled, (root / name).read_text(), name)


class AnswerScoringTests(unittest.TestCase):
    def test_glm_log_examples_keep_content_and_score_six_of_seven(self):
        predictions = ["yes", "STOP WORRYING", "steps", "metal", "no", "no", "left"]
        references = ["yes", "sticker", "steps", "metal", "no", "no", "left"]
        rows = []
        for index, (prediction, reference) in enumerate(zip(predictions, references)):
            raw = f"<|begin_of_box|>{prediction}<|end_of_box|>"
            row = evaluator.make_record(evaluator.Example(str(index), "image.jpg", "?", reference),
                                        {"prediction": raw, "raw_response": raw, "latency_seconds": 0.1})
            self.assertEqual(row["prediction"], prediction)
            self.assertEqual(row["raw_response"], raw)
            rows.append(row)
        with tempfile.TemporaryDirectory() as temporary:
            report = evaluator.save_outputs(Path(temporary), {r["question_id"]: r for r in rows}, {}, 7)
        self.assertAlmostEqual(report["accuracy"], 6 / 7)
        self.assertFalse(rows[1]["correct"])

    def test_cleanup_preserves_prose_case_punctuation_and_unknown_tokens(self):
        cases = {
            "<|begin_of_box|>**Yes!**<|end_of_box|>": "Yes!",
            "<|begin_of_box|>yes": "yes",  # Truncated closing marker.
            "Answer: <|begin_of_box|>yes<|end_of_box|> because it is visible.":
                "Answer: yes because it is visible.",
            "<|unknown|>yes": "<|unknown|>yes",
            "<|begin_of_box|>2<|end_of_box|>, <|begin_of_box|>3<|end_of_box|>": "2, 3",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(evaluator.normalize_llm_answer(raw), expected)
                self.assertEqual(evaluator.normalize_llm_answer(expected), expected)
        row = evaluator.make_record(evaluator.Example("id", "image.jpg", "?", "yes!"),
                                    {"prediction": "<|begin_of_box|>Yes!<|end_of_box|>"})
        self.assertFalse(row["correct"])
        self.assertTrue(row["normalized_correct"])

    def test_plantexpert_and_errors_use_the_same_cleanup_without_forgiving_errors(self):
        example = evaluator.Example("id", "image.jpg", "?", "yes")
        result = {"prediction": "<|begin_of_box|>yes<|end_of_box|>"}
        self.assertEqual(evaluator.make_record(example, result, dataset="plantexpert")["em"], 1)
        self.assertFalse(evaluator.make_record(example, result, error="failed")["correct"])
        self.assertIsNone(evaluator.make_record(
            evaluator.Example("id", "image.jpg", "?", None), result)["correct"])


class SavedScoringTests(unittest.TestCase):
    def create_old_run(self, root):
        questions = root / "questions.csv"
        questions.write_text("question_id,image,question,answer\n1,image.jpg,Is it visible?,yes\n")
        (root / "image.jpg").touch()
        args = evaluator.parse_args(["--model", "glm-4.6v-flash", "--questions", str(questions),
                                     "--images", str(root), "--output-dir", str(root / "output")])
        client = MagicMock(base_url="transformers://local")
        raw = "<|begin_of_box|>yes<|end_of_box|>"
        client.answer.return_value = {"prediction": raw, "raw_response": raw, "latency_seconds": 0.1}
        evaluator.evaluate(args, client)
        journal = args.output_dir / "predictions.jsonl"
        row = json.loads(journal.read_text())
        row.update(prediction=raw, correct=False, normalized_correct=False)
        journal.write_text(json.dumps(row) + "\n")
        manifest = args.output_dir / "run_config.json"
        config = json.loads(manifest.read_text())
        config["answer_normalization"] = "whitespace_markdown_v1"
        manifest.write_text(json.dumps(config))
        client.answer.reset_mock()
        return args, client, raw

    def test_resume_rescores_and_persists_legacy_rows_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, client, raw = self.create_old_run(Path(temporary))
            args.resume = True
            report = evaluator.evaluate(args, client)
            client.answer.assert_not_called()
            self.assertEqual(report["accuracy"], 1)
            self.assertEqual(report["config"]["answer_normalization"], evaluator.ANSWER_NORMALIZATION)
            rows = [json.loads(line) for line in (args.output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["prediction"], raw)
            self.assertEqual(rows[-1]["prediction"], "yes")
            self.assertEqual(rows[-1]["raw_response"], raw)
            self.assertTrue(rows[-1]["correct"])
            evaluator.evaluate(args, client)
            self.assertEqual(len((args.output_dir / "predictions.jsonl").read_text().splitlines()), 2)

    def test_offline_rescore_needs_neither_model_nor_original_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, raw = self.create_old_run(Path(temporary))
            args.questions.unlink()
            (Path(temporary) / "image.jpg").unlink()
            with patch.object(evaluator, "inspect_model") as inspect, patch.object(
                    evaluator, "TransformersClient") as client:
                self.assertEqual(evaluator.main(["--rescore-only", "--output-dir", str(args.output_dir)]), 0)
                inspect.assert_not_called()
                client.assert_not_called()
            report = json.loads((args.output_dir / "report.json").read_text())
            self.assertEqual(report["accuracy"], 1)
            self.assertEqual(report["examples_selected"], 1)
            self.assertTrue(report["complete"])
            self.assertEqual(report["config"]["model"], "zai-org/GLM-4.6V-Flash")
            latest = json.loads((args.output_dir / "predictions.jsonl").read_text().splitlines()[-1])
            self.assertEqual(latest["raw_response"], raw)

    def test_resume_still_rejects_generation_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, client, _ = self.create_old_run(Path(temporary))
            args.resume = True
            args.seed += 1
            with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
                evaluator.evaluate(args, client)
            client.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
