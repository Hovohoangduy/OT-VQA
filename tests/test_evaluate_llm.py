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


if __name__ == "__main__":
    unittest.main()
