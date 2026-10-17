"""Regression tests for checkpoint inspection and architecture-specific loading.

Run without GPU/weights: python -m unittest discover -s tests -v
"""

import ast
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout

from scripts import evaluate_llm as evaluator
from utils import scivqa_prompt


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
        self.assertIn("utils/scivqa_prompt.py", bundles)

    def test_scivqa_notebook_passes_context_and_split_to_cli(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/evaluate_llm_kaggle.ipynb").read_text())
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec("".join(notebook["cells"][1]["source"]), namespace)
            namespace["subprocess_json"] = MagicMock(side_effect=[
                {"model": evaluator.MODEL_PRESETS["llava-onevision-1.5-8b"], "folder": "fixture"},
                {"revision": "a" * 40, "model_type": evaluator.ONEVISION_MODEL_TYPE, "dtype": "float16"},
            ])
            exec("".join(notebook["cells"][6]["source"]), namespace)
        compile(namespace["preflight_code"], "notebook preflight", "exec")
        with tempfile.TemporaryDirectory() as temporary:
            command = namespace["evaluation_command"](Path(temporary), limit=7)
        args = evaluator.parse_args(command[4:])
        self.assertEqual((args.dataset, args.split, args.scivqa_context), ("scivqa", "test", "full"))
        self.assertEqual(args.questions.name, "test.csv")
        self.assertEqual(args.bertscore_model, "roberta-large")
        self.assertEqual(args.limit, 7)


class SciVQATests(unittest.TestCase):
    def row(self, **changes):
        row = {
            "anno_id": "q1", "image": "figure.png", "question": "Which line is highest?",
            "original_question": "Which line is highest?", "answer": "B",
            "answer_options": '[{"A": "Red"}, {"B": "Blue"}]',
            "caption": "The experiment compares two systems.", "figure_type": "line chart",
            "compound": "True", "figs_numb": "2",
            "qa_pair_type": "closed-ended finite answer set non-binary visual",
            "paper_id": "PAPER_SENTINEL", "pdf_url": "https://example.org/paper",
        }
        row.update(changes)
        return row

    def write_rows(self, root, rows):
        path = root / "val.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root / "figure.png").write_bytes(b"fixture: fake client does not decode images")
        return path

    def args(self, root, **flags):
        args = evaluator.parse_args([
            "--dataset", "scivqa", "--split", "dev", "--questions", str(root / "val.csv"),
            "--images", str(root), "--output-dir", str(root / "results"), "--no-bertscore",
        ])
        for name, value in flags.items():
            setattr(args, name, value)
        return args

    def test_full_context_contains_choices_caption_and_correct_panel_count(self):
        row = self.row(answer="GOLD_SENTINEL")
        prompt = scivqa_prompt.build_prompt(row["question"], row)
        for text in ("A: Red; B: Blue", row["caption"], "Figure type: line chart",
                     "contains 2 subfigures", "Use the panel referenced in the question",
                     "Return only the letter(s)"):
            self.assertIn(text, prompt)
        for text in ("GOLD_SENTINEL", "PAPER_SENTINEL", "https://example.org", "Navigate to subfigure 2"):
            self.assertNotIn(text, prompt)

    def test_exported_caption_and_choices_are_not_duplicated(self):
        row = self.row()
        row["question"] += "\nAnswer options: A: Red; B: Blue\nFigure caption: " + row["caption"]
        prompt = scivqa_prompt.build_prompt(row["question"], row)
        self.assertEqual(prompt.count("Answer options:"), 1)
        self.assertEqual(prompt.count("Figure caption:"), 1)

    def test_question_only_keeps_choices_and_excludes_figure_context(self):
        row = self.row()
        row["question"] += "\nFigure caption: " + row["caption"]
        prompt = scivqa_prompt.build_prompt(row["question"], row, "question-only")
        self.assertIn("A: Red; B: Blue", prompt)
        self.assertNotIn(row["caption"], prompt)
        self.assertNotIn("Figure type", prompt)
        self.assertNotIn("subfigures", prompt)

    def test_unanswerable_annotation_does_not_change_prompt(self):
        row = self.row(answer_options="[]", qa_pair_type="unanswerable")
        first = scivqa_prompt.build_prompt(row["question"], row)
        row.update(qa_pair_type="closed-ended infinite answer set visual", answer="NEW_GOLD")
        self.assertEqual(first, scivqa_prompt.build_prompt(row["question"], row))
        self.assertNotIn("unanswerable", first)

    def test_binary_and_false_compound(self):
        row = self.row(answer_options="[]", compound="False", qa_pair_type="closed-ended finite answer set binary visual")
        prompt = scivqa_prompt.build_prompt(row["question"], row)
        self.assertIn("Answer with 'Yes' or 'No'.", prompt)
        self.assertNotIn("Return only the letter(s)", prompt)
        self.assertNotIn("Subfigure navigation", prompt)

    def test_options_are_validated(self):
        for options in ('{}', '[{"A":"one"},{"A":"two"}]', '[{"A":"one","B":"two"}]', '[{"Z":"one"}]'):
            with self.subTest(options=options), self.assertRaises(ValueError):
                scivqa_prompt.build_prompt("Question?", self.row(answer_options=options))

    def test_csv_preserves_metadata_and_rejects_missing_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_rows(root, [self.row()])
            example = evaluator.load_examples(path, "scivqa")[0]
            self.assertIn("Figure caption", example.question)
            self.assertEqual(example.metadata["paper_id"], "PAPER_SENTINEL")
            self.assertEqual(example.answer, "B")
            row = self.row()
            del row["caption"]
            self.write_rows(root, [row])
            with self.assertRaisesRegex(ValueError, "context columns missing"):
                evaluator.load_examples(path, "scivqa")

    def test_cli_scivqa_defaults_and_explicit_overrides(self):
        args = evaluator.parse_args(["--dataset", "scivqa"])
        self.assertEqual(args.questions, Path("data/scivqa_dataset/test.csv"))
        self.assertEqual(args.split, "test")
        self.assertIn("scivqa_test_full", args.output_dir.name)
        self.assertEqual((args.max_tokens, args.bertscore_model, args.bertscore_rescale),
                         (256, "roberta-large", False))
        args = evaluator.parse_args(["--dataset", "scivqa", "--split", "test",
                                     "--scivqa-context", "question-only", "--bertscore-rescale"])
        self.assertEqual(args.questions.name, "test.csv")
        self.assertTrue(args.bertscore_rescale)
        self.assertIn("test_question-only", str(args.output_dir))
        dev = evaluator.parse_args(["--dataset", "scivqa", "--split", "dev"])
        self.assertEqual(dev.questions, Path("data/scivqa_dataset/val.csv"))
        self.assertIn("scivqa_dev_full", dev.output_dir.name)
        legacy = evaluator.parse_args(["--dataset", "plantexpert"])
        self.assertEqual(legacy.questions.name, "test.csv")
        self.assertEqual(legacy.bertscore_model, "bert-base-uncased")
        self.assertTrue(legacy.bertscore_rescale)

    def test_preview_does_not_load_models_or_expose_gold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_rows(root, [self.row(answer="GOLD_SENTINEL")])
            with patch.object(evaluator, "inspect_model") as inspect, redirect_stdout(io.StringIO()) as out:
                code = evaluator.main(["--dataset", "scivqa", "--questions", str(path),
                                       "--images", str(root), "--preview-prompts", "1"])
            self.assertEqual(code, 0)
            inspect.assert_not_called()
            self.assertNotIn("GOLD_SENTINEL", out.getvalue())
            self.assertIn("Figure caption", json.loads(out.getvalue())[0]["user_prompt"])

    def test_evaluation_resume_and_changed_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_rows(root, [self.row()])
            args = self.args(root)
            client = MagicMock(base_url="fixture://local")
            client.answer.return_value = {"prediction": "B", "raw_response": "B", "latency_seconds": 0.1}
            with redirect_stdout(io.StringIO()):
                report = evaluator.evaluate(args, client)
                self.assertEqual(report["generated_metrics"]["rouge_l_f1"], 1.0)
                self.assertIsNone(report["generated_metrics"]["bertscore_f1"])
                self.assertEqual(len(report["generated_metrics"]), 9)
                self.assertIn("provided context", client.answer.call_args.kwargs["system_prompt"])
                self.assertIn("Figure caption", client.answer.call_args.args[1].question)
                args.resume = True
                evaluator.evaluate(args, client)
                self.assertEqual(client.answer.call_count, 1)
                args.scivqa_context = "question-only"
                with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
                    evaluator.evaluate(args, client)

    def test_failed_and_unlabeled_rows_and_all_semantic_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = {}
            for identifier, answer, error in (("ok", "B", ""), ("failed", "B", "failure"), ("unlabeled", None, "")):
                example = evaluator.Example(identifier, "figure.png", "Which?", answer, {"qa_pair_type": "example type"})
                records[identifier] = evaluator.make_record(
                    example, {"prediction": "B", "latency_seconds": 0.1}, error, "scivqa")
            scorer = MagicMock(hash="fixture-hash")
            scorer.score.return_value = ([0.7], [0.8], [0.75])
            report = evaluator.save_outputs(output, records, {"dataset": "scivqa", "bertscore": {}}, 3, scorer)
            self.assertEqual(report["generated_metrics"]["rouge_l_f1"], 0.5)
            self.assertEqual(report["generated_metrics"]["bertscore_precision"], 0.35)
            self.assertEqual(report["generated_metrics"]["bertscore_recall"], 0.4)
            self.assertEqual(report["generated_metrics"]["bertscore_f1"], 0.375)
            self.assertEqual(report["by_qa_pair_type"]["example type"]["count"], 2)
            self.assertIsNone(records["unlabeled"]["rouge_l_f1"])
            self.assertFalse(report["complete"])

    def test_failed_inference_is_retried_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_rows(root, [self.row()])
            args = self.args(root, continue_on_error=True)
            client = MagicMock(base_url="fixture://local")
            client.answer.side_effect = [RuntimeError("fixture failure"),
                                        {"prediction": "B", "raw_response": "B", "latency_seconds": 0.1}]
            with redirect_stdout(io.StringIO()):
                failed = evaluator.evaluate(args, client)
                self.assertEqual(failed["generated_metrics"]["rouge_l_f1"], 0.0)
                self.assertFalse(failed["complete"])
                args.resume = True
                recovered = evaluator.evaluate(args, client)
            self.assertEqual(recovered["errors"], 0)
            self.assertTrue(recovered["complete"])
            self.assertEqual(recovered["generated_metrics"]["rouge_l_f1"], 1.0)
            self.assertEqual(len((root / "results/predictions.jsonl").read_text().splitlines()), 2)

    def test_training_scivqa_scorer_still_requires_semantic_scorer_by_default(self):
        with self.assertRaisesRegex(ValueError, "BERTScore scorer is required"):
            evaluator.scivqa_score_pairs(["A"], ["A"], None)


if __name__ == "__main__":
    unittest.main()
