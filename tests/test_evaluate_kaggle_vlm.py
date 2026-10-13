"""Check format routing, Transformers generation, scoring and resume without GPUs."""

import json
import io
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import evaluate_kaggle_vlm as runner
from scripts.build_kaggle_notebook import build_notebook
from scripts.evaluate_gqa_lmstudio import Example, PLANTEXPERT_SYSTEM_PROMPT


class Tensor:
    """Minimal tensor fixture that detects accidental casting of token IDs."""
    def __init__(self, values, floating=False):
        self.values = values
        self.floating = floating
        self.shape = (1, len(values))
        self.moves = []

    def is_floating_point(self):
        return self.floating

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Tensor(self.values[index[1]], self.floating)
        return Mock(item=Mock(return_value=self.values[index]))

    def __len__(self):
        return len(self.values)


class KaggleVLMTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.questions = self.root / "test.csv"
        self.questions.write_text("image,question,answer\n1.jpg,What color?,red\n")
        self.image = self.root / "1.jpg"
        self.image.write_bytes(b"image-fixture")
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text('{"model_type":"internvl"}')
        (self.model_dir / "model.safetensors").write_bytes(b"weights")
        self.flags = ["--model-dir", str(self.model_dir), "--questions", str(self.questions),
                      "--images", str(self.root), "--output-dir", str(self.root / "results")]

    def hub(self, names):
        hub = Mock()
        hub.HfApi.return_value.model_info.return_value.sha = "pinned-commit"
        hub.HfApi.return_value.list_repo_files.return_value = names
        hub.snapshot_download.return_value = str(self.model_dir)
        return hub

    def test_detects_original_caprl_and_gguf_using_pinned_metadata(self):
        cases = [
            ("internlm/CapRL-InternVL3.5-8B", ["config.json", "model.safetensors"], "transformers"),
            ("https://huggingface.co/Jobaar/CapRL-InternVL3.5-8B-GGUF:Q4_K_M",
             ["CapRL-Q4_K_M.gguf", "mmproj-CapRL-F16.gguf"], "gguf"),
        ]
        for model_dir, names, backend in cases:
            with self.subTest(backend=backend):
                hub = self.hub(names)
                with patch.dict("sys.modules", {"huggingface_hub": hub}):
                    source = runner.inspect_model(model_dir)
                self.assertEqual(source["backend"], backend)
                self.assertEqual(source["revision"], "pinned-commit")
                hub.HfApi.return_value.list_repo_files.assert_called_once_with(
                    source["repo"], revision="pinned-commit")
                hub.snapshot_download.assert_not_called()

    def test_local_detection_overrides_and_invalid_formats(self):
        self.assertEqual(runner.inspect_model(str(self.model_dir))["backend"], "transformers")
        with self.assertRaisesRegex(ValueError, "no files for backend 'gguf'"):
            runner.inspect_model(str(self.model_dir), backend="gguf")
        (self.model_dir / "Vision-Q4_K_M.gguf").write_bytes(b"fixture")
        with self.assertRaisesRegex(ValueError, "vision GGUF projector"):
            runner.inspect_model(str(self.model_dir))
        (self.model_dir / "mmproj-Vision-F16.gguf").write_bytes(b"fixture")
        self.assertEqual(runner.inspect_model(str(self.model_dir))["backend"], "gguf")
        self.assertEqual(runner.inspect_model(str(self.model_dir), backend="transformers")["backend"],
                         "transformers")
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            runner.inspect_model(str(self.root / "missing"))
        hub = self.hub(["config.json", "model.safetensors"])
        with patch.dict("sys.modules", {"huggingface_hub": hub}), \
                self.assertRaisesRegex(ValueError, "suffix selects GGUF"):
            runner.inspect_model("internlm/CapRL-InternVL3.5-8B:Q4_K_M")

    def test_explicit_gguf_quantization_is_used_during_inspection(self):
        hub = self.hub(["Vision-Q2_K.gguf", "Vision-Q6_K.gguf", "mmproj-Vision-F16.gguf"])
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            self.assertEqual(runner.inspect_model("owner/Vision-GGUF", quantization="Q6_K")["backend"], "gguf")
            with self.assertRaisesRegex(ValueError, "Conflicting quantizations"):
                runner.inspect_model("owner/Vision-GGUF:Q2_K", quantization="Q6_K")

    def test_transformers_download_pins_processor_and_shards_and_excludes_assets(self):
        names = ["config.json", "preprocessor_config.json", "tokenizer.json", "chat_template.jinja",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                 "model.safetensors.index.json", "modeling_custom.py", "assets/image.png", "README.md"]
        hub = self.hub(names)
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            source = runner.inspect_model("internlm/CapRL-InternVL3.5-8B")
            directory, identity = runner.resolve_transformers_model(source, self.root / "cache", False)
        self.assertEqual(directory, self.model_dir)
        self.assertEqual(identity, {"repo": "internlm/CapRL-InternVL3.5-8B", "revision": "pinned-commit"})
        kwargs = hub.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["revision"], "pinned-commit")
        self.assertEqual(kwargs["allow_patterns"], names[:7])
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            runner.resolve_transformers_model(source, self.root / "cache", True)
        self.assertIn("modeling_custom.py", hub.snapshot_download.call_args.kwargs["allow_patterns"])

    def test_local_identity_detects_changed_weights_and_processor(self):
        source = runner.inspect_model(str(self.model_dir))
        _, before = runner.resolve_transformers_model(source, self.root, False)
        for name in ("model.safetensors", "config.json"):
            (self.model_dir / name).write_bytes(b"changed")
        _, after = runner.resolve_transformers_model(source, self.root, False)
        self.assertNotEqual(before["files_sha256"]["model.safetensors"], after["files_sha256"]["model.safetensors"])
        self.assertNotEqual(before["files_sha256"]["config.json"], after["files_sha256"]["config.json"])

    def client(self, placement=None):
        torch = Mock(__version__="torch-test", float16="fp16", bfloat16="bf16")
        torch.cuda.device_count.return_value = 2
        torch.cuda.get_device_capability.return_value = (7, 5)
        torch.inference_mode.side_effect = lambda: nullcontext()
        torch.is_tensor.side_effect = lambda value: isinstance(value, Tensor)
        model = Mock(device="cuda:0", dtype="fp16", hf_device_map=placement or {"vision": 0, "text": 1})
        model.eval.return_value = model
        model.generation_config.eos_token_id = [2]
        model.generate.return_value = Tensor([10, 11, 12, 20, 2])
        processor = Mock()
        processor.apply_chat_template.return_value = {
            "input_ids": Tensor([10, 11, 12]), "attention_mask": Tensor([1, 1, 1]),
            "pixel_values": Tensor([0.1, 0.2], floating=True),
        }
        processor.decode.return_value = " **red**\n"
        transformers = Mock(__version__="4.57.1")
        transformers.AutoProcessor.from_pretrained.return_value = processor
        transformers.AutoModelForImageTextToText.from_pretrained.return_value = model
        modules = {"torch": torch, "transformers": transformers, "accelerate": Mock(__version__="1-test")}
        return modules, torch, transformers, model, processor

    def test_t4_loads_fp16_balanced_without_cpu_offload_or_remote_code(self):
        modules, _, transformers, _, _ = self.client()
        with patch.dict("sys.modules", modules):
            client = runner.TransformersClient(self.model_dir, "caprl")
        kwargs = transformers.AutoModelForImageTextToText.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["torch_dtype"], "fp16")
        self.assertEqual(kwargs["device_map"], "balanced")
        self.assertEqual(kwargs["max_memory"], {0: 12 * 2**30, 1: 12 * 2**30, "cpu": 0})
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(client.backend_config["dtype"], "float16")
        modules, _, _, _, _ = self.client({"text": "cpu"})
        with patch.dict("sys.modules", modules), self.assertRaisesRegex(RuntimeError, "do not fit"):
            runner.TransformersClient(self.model_dir, "caprl")

    def test_generation_keeps_image_and_prompt_and_excludes_reference_and_prompt_tokens(self):
        modules, torch, _, model, processor = self.client()
        with patch.dict("sys.modules", modules):
            client = runner.TransformersClient(self.model_dir, "caprl")
        example = Example("1", "1.jpg", "What color?", "SECRET REFERENCE")
        result = client.answer("caprl", example, self.image, 2, 42, PLANTEXPERT_SYSTEM_PROMPT)
        messages = processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"], PLANTEXPERT_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], [
            {"type": "image", "url": str(self.image.resolve())}, {"type": "text", "text": "What color?"}])
        self.assertNotIn("SECRET REFERENCE", str(messages))
        inputs = processor.apply_chat_template.return_value
        self.assertEqual(inputs["input_ids"].moves, [(("cuda:0",), {})])
        self.assertEqual(inputs["pixel_values"].moves, [((), {"device": "cuda:0", "dtype": "fp16"})])
        self.assertEqual(processor.decode.call_args.args[0].values, [20, 2])
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        torch.manual_seed.assert_called_once_with(42)
        self.assertEqual(result["prediction"], "red")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        model.generate.return_value = Tensor([10, 11, 12, 20, 21])
        self.assertEqual(client.answer("caprl", example, self.image, 2, 42)["finish_reason"], "length")
        client.context_size = 4
        with self.assertRaisesRegex(ValueError, "exceeding context-size"):
            client.answer("caprl", example, self.image, 2, 42)

    def test_auto_dispatch_preserves_legacy_gguf_runner(self):
        source = {"backend": "gguf", "repo": "owner/model", "revision": "pinned"}
        with patch.object(runner, "inspect_model", return_value=source), \
                patch.object(runner.gguf, "main", return_value=0) as legacy, \
                patch.object(runner, "TransformersClient") as client, patch("builtins.print"):
            self.assertEqual(runner.main(self.flags), 0)
        self.assertEqual(legacy.call_args.args[0][-2:], ["--hf-revision", "pinned"])
        client.assert_not_called()

    def test_transformers_integration_scores_resumes_and_rejects_changed_revision(self):
        source = {"backend": "transformers", "repo": "internlm/CapRL-InternVL3.5-8B", "revision": "pinned"}
        client = Mock(base_url="transformers://local", backend_config={"engine": "transformers"})
        client.answer.return_value = {"prediction": "red", "raw_response": "red", "latency_seconds": 0.1,
                                      "finish_reason": "stop", "usage": {}}
        with patch.object(runner, "inspect_model", return_value=source), \
                patch.object(runner.gguf, "gpu_environment", return_value=({"CUDA_VISIBLE_DEVICES": "0,1"}, ["T4", "T4"])), \
                patch.object(runner, "resolve_transformers_model", side_effect=lambda *a: (self.model_dir, {
                    "repo": source["repo"], "revision": source["revision"]})), \
                patch.object(runner, "read_transformers_config", return_value={"model_type": "internvl"}), \
                patch.object(runner, "TransformersClient", return_value=client), \
                patch.dict(os.environ, {}, clear=True), patch("builtins.print"):
            self.assertEqual(runner.main(self.flags), 0)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")
            report = json.loads((self.root / "results/report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["accuracy"], 1.0)
            self.assertEqual(report["config"]["backend"]["model_source"]["revision"], "pinned")
            client.answer.reset_mock()
            self.assertEqual(runner.main([*self.flags, "--resume"]), 0)
            client.answer.assert_not_called()
            source["revision"] = "changed"
            self.assertEqual(runner.main([*self.flags, "--resume"]), 1)
            client.answer.assert_not_called()

    def test_bad_data_stops_before_hub_inspection(self):
        with patch.object(runner, "inspect_model") as inspect, patch("builtins.print"):
            self.assertEqual(runner.main([*self.flags, "--images", str(self.root / "missing")]), 1)
        inspect.assert_not_called()

    def test_notebook_skips_llama_build_for_transformers(self):
        source = next("".join(cell["source"]) for cell in build_notebook()["cells"]
                      if "".join(cell["source"]).startswith("LLAMA_ROOT = WORK"))
        namespace = {"WORK": self.root, "SELECTED_BACKEND": "transformers"}
        with patch("builtins.print"):
            exec(source, namespace)
        self.assertFalse((self.root / "llama.cpp").exists())

    def test_notebook_uses_original_model_and_rejects_stale_backend_selection(self):
        source = next("".join(cell["source"]) for cell in build_notebook()["cells"]
                      if "".join(cell["source"]).startswith("def evaluation_command():"))
        model_dir = "internlm/CapRL-InternVL3.5-8B"
        namespace = {
            "sys": sys, "LLAMA_SERVER": self.root / "missing-server", "WORK": self.root,
            "HF_REVISION": "main", "CONTEXT_SIZE": 8192, "DATASET": "plantexpert",
            "QUESTIONS": self.questions, "IMAGES": self.root, "IMAGE_SPLIT": "test",
            "MAX_TOKENS": 256, "NO_BERTSCORE": True, "MODEL_DIR": model_dir,
            "BACKEND": "auto", "SELECTED_BACKEND": "transformers",
            "SELECTED_MODEL": {"revision": "pinned"}, "SELECTED_MODEL_DIR": model_dir,
            "SELECTED_HF_REVISION": "main", "SELECTED_BACKEND_SETTING": "auto",
            "TRUST_REMOTE_CODE": False, "SELECTED_TRUST_REMOTE_CODE": False,
            "ATTENTION_IMPLEMENTATION": "auto", "SELECTED_ATTENTION_IMPLEMENTATION": "auto",
        }
        exec(source, namespace)
        command = namespace["evaluation_command"]()
        extra, options, _, _ = runner.parse_args(command[4:])
        self.assertEqual(extra.backend, "transformers")
        self.assertEqual(options.model_dir, model_dir)
        self.assertEqual(options.hf_revision, "pinned")
        self.assertEqual(options.cache_dir, self.root / "hf-models")
        self.assertNotIn("--llama-server", command)
        namespace["TRUST_REMOTE_CODE"] = True
        with self.assertRaisesRegex(AssertionError, "Loading settings changed"):
            namespace["evaluation_command"]()
        namespace["SELECTED_TRUST_REMOTE_CODE"] = True
        self.assertIn("--trust-remote-code", namespace["evaluation_command"]())
        namespace["MODEL_DIR"] = "different/model"
        with self.assertRaisesRegex(AssertionError, "Re-run the backend selection"):
            namespace["evaluation_command"]()

    @staticmethod
    def keye_config():
        return {"model_type": "KeyeVL1_5", "auto_map": {
            "AutoConfig": "configuration_keye_vl_1_5.KeyeVL1_5Config",
            "AutoModel": "modeling_keye_vl_1_5.KeyeVL1_5ForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_keye_vl_1_5.KeyeVL1_5ForConditionalGeneration",
        }}

    def test_keye_requires_explicit_custom_code_and_correct_model_class(self):
        with self.assertRaisesRegex(ValueError, "TRUST_REMOTE_CODE = True"):
            runner.transformers_loading_plan(self.keye_config(), False)
        plan = runner.transformers_loading_plan(self.keye_config(), True)
        self.assertEqual(plan, {"loader": "AutoModel", "attention": "sdpa",
                                "image_processing": "keye-vl-1.5"})
        self.assertEqual(runner.transformers_loading_plan(self.keye_config(), True, "sdpa"), plan)
        self.assertEqual(runner.transformers_loading_plan(self.keye_config(), True, "flash_attention_2")["attention"],
                         "flash_attention_2")
        with self.assertRaisesRegex(ValueError, "SDPA adaptation"):
            runner.transformers_loading_plan(self.keye_config(), True, "eager")

    def test_keye_rejects_t4_and_missing_flash_attention_before_weight_download(self):
        (self.model_dir / "config.json").write_text(json.dumps(self.keye_config()))
        modules, torch, _, _, _ = self.client()
        with patch.object(runner.gguf, "gpu_environment", return_value=({"CUDA_VISIBLE_DEVICES": "0,1"}, ["T4", "T4"])), \
                patch.object(runner, "resolve_transformers_model") as download, \
                patch.dict("sys.modules", modules), patch.dict(os.environ, {}, clear=True), \
                patch("builtins.print") as printed:
            self.assertEqual(runner.main(self.flags), 1)
            self.assertIn("TRUST_REMOTE_CODE = True", str(printed.call_args))
            self.assertEqual(runner.main([*self.flags, "--trust-remote-code",
                                         "--attention-implementation", "flash_attention_2"]), 1)
            self.assertIn("T4 x2 is unsupported", str(printed.call_args))
            download.assert_not_called()
        torch.cuda.get_device_capability.return_value = (8, 0)
        with patch.object(runner.importlib.util, "find_spec", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "FlashAttention-2 is missing"):
            runner.validate_attention_hardware(torch, "flash_attention_2")

    def test_custom_config_download_is_small_and_uses_pinned_revision(self):
        config_file = self.model_dir / "config.json"
        config_file.write_text(json.dumps(self.keye_config()))
        hub = self.hub(["config.json", "model.safetensors"])
        hub.hf_hub_download.return_value = str(config_file)
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            config = runner.read_transformers_config({"repo": "Kwai-Keye/Keye-VL-1_5-8B",
                                                     "revision": "b4d413e"}, self.root)
        self.assertEqual(config["model_type"], "KeyeVL1_5")
        hub.hf_hub_download.assert_called_once_with(repo_id="Kwai-Keye/Keye-VL-1_5-8B",
            filename="config.json", revision="b4d413e", cache_dir=str(self.root))
        hub.snapshot_download.assert_not_called()

    def test_keye_loads_registered_automodel_and_preprocesses_image_separately(self):
        (self.model_dir / "config.json").write_text(json.dumps(self.keye_config()))
        modules, torch, transformers, model, processor = self.client()
        torch.cuda.get_device_capability.return_value = (8, 0)
        transformers.AutoModel.from_pretrained.return_value = model
        inputs = processor.apply_chat_template.return_value
        inputs["image_grid_thw"] = Tensor([1, 2, 2])
        inputs["metadata"] = ["keep this list"]
        processor.apply_chat_template.return_value = "rendered prompt with image placeholder"
        processor.return_value = inputs
        image_module = Mock()
        opened = image_module.Image.open.return_value
        opened.__enter__ = Mock(return_value=opened)
        opened.__exit__ = Mock(return_value=False)
        with patch.dict("sys.modules", {**modules, "PIL": image_module}), \
                patch.object(runner.importlib.util, "find_spec", return_value=Mock()):
            client = runner.TransformersClient(self.model_dir, "keye", trust_remote_code=True,
                                               attention="flash_attention_2")
            result = client.answer("keye", Example("1", "1.jpg", "What color?", "SECRET"), self.image, 2, 42)
        transformers.AutoModelForImageTextToText.from_pretrained.assert_not_called()
        kwargs = transformers.AutoModel.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")
        self.assertTrue(kwargs["trust_remote_code"])
        messages = processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[1]["content"][0], {"type": "image", "image": str(self.image.resolve())})
        self.assertEqual(messages[1]["content"][1]["text"], "What color?/no_think")
        self.assertNotIn("SECRET", str(messages))
        self.assertFalse(processor.apply_chat_template.call_args.kwargs["tokenize"])
        self.assertEqual(processor.call_args.kwargs["text"], ["rendered prompt with image placeholder"])
        self.assertEqual(processor.call_args.kwargs["images"], [opened.convert.return_value])
        self.assertEqual(model.generate.call_args.kwargs["metadata"], ["keep this list"])
        self.assertEqual(inputs["image_grid_thw"].moves, [(("cuda:0",), {})])
        self.assertEqual(result["prediction"], "red")
        self.assertEqual(client.backend_config["question_suffix"], "/no_think")

    def test_keye_sdpa_loads_staged_code_on_t4_and_records_adaptation(self):
        (self.model_dir / "config.json").write_text(json.dumps(self.keye_config()))
        modules, _, transformers, model, _ = self.client()
        transformers.AutoModel.from_pretrained.return_value = model
        staged = self.root / "adapted-model"
        identity = {"attention_adapter": "keye-vl-1.5-sdpa-v1", "adapted_model_code_sha256": "hash"}
        with patch.dict("sys.modules", modules), \
                patch("scripts.keye_sdpa.stage_keye_sdpa", return_value=(Mock(), staged, identity)) as stage, \
                patch.object(runner.importlib.util, "find_spec") as package:
            client = runner.TransformersClient(self.model_dir, "keye", trust_remote_code=True)
        stage.assert_called_once_with(self.model_dir)
        package.assert_not_called()
        kwargs = transformers.AutoModel.from_pretrained.call_args.kwargs
        self.assertEqual(transformers.AutoModel.from_pretrained.call_args.args[0], str(staged))
        self.assertEqual(transformers.AutoProcessor.from_pretrained.call_args.args[0], str(staged))
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertEqual(kwargs["torch_dtype"], "fp16")
        self.assertEqual(client.backend_config["attention_adapter"], identity["attention_adapter"])
        self.assertEqual(client.backend_config["adapted_model_code_sha256"], "hash")

    def test_notebook_keye_sdpa_skips_flash_attention_install_on_t4(self):
        source = next("".join(cell["source"]) for cell in build_notebook()["cells"]
                      if "".join(cell["source"]).startswith("SELECTED_MODEL = inspect_model"))
        _, torch, _, _, _ = self.client()
        namespace = {"MODEL_DIR": "Kwai-Keye/Keye-VL-1_5-8B", "HF_REVISION": "main", "BACKEND": "auto",
                     "TRUST_REMOTE_CODE": True, "ATTENTION_IMPLEMENTATION": "auto", "WORK": self.root,
                     "torch": torch, "sys": sys, "subprocess": Mock(),
                     "inspect_model": Mock(return_value={"backend": "transformers", "revision": "pinned"}),
                     "read_transformers_config": Mock(return_value=self.keye_config()),
                     "transformers_loading_plan": runner.transformers_loading_plan,
                     "validate_attention_hardware": runner.validate_attention_hardware}
        with patch("builtins.print"):
            exec(source, namespace)
        self.assertEqual(namespace["LOADING_PLAN"]["attention"], "sdpa")
        calls = namespace["subprocess"].run.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertNotIn("flash-attn", str(calls))
        self.assertIn("einops>=0.8", str(calls))

    def test_logged_runner_surfaces_child_error_and_saves_both_streams(self):
        log = self.root / "logs/vlm-smoke.log"
        output = io.StringIO()
        command = [sys.executable, "-u", "-c",
                   "import sys; print('Selected backend: transformers'); print('Error: custom code required', file=sys.stderr); sys.exit(1)"]
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "custom code required") as raised:
            runner.run_logged_evaluation(command, self.root, log)
        self.assertIn(str(log), str(raised.exception))
        self.assertIn("Selected backend: transformers", output.getvalue())
        self.assertIn("Error: custom code required", output.getvalue())
        self.assertEqual(log.read_text(), output.getvalue())
        with redirect_stdout(io.StringIO()):
            runner.run_logged_evaluation([sys.executable, "-c", "print('success')"], self.root, log)
        self.assertEqual(log.read_text(), "success\n")


if __name__ == "__main__":
    unittest.main()
