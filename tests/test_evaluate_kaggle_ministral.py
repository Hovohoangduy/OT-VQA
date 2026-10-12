"""Check two-GPU launch, cleanup, notebook portability, and scoring without GPUs."""

import ast
import io
import json
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import evaluate_kaggle_ministral as runner
from scripts.build_kaggle_notebook import BUNDLED_FILES, ROOT, build_notebook
from scripts.evaluate_gqa_lmstudio import LMStudioClient, evaluate


class KaggleEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model.gguf"
        self.projector = self.root / "mmproj.gguf"
        self.binary = self.root / "llama-server"
        for path in (self.model, self.projector, self.binary):
            path.write_bytes(b"fixture")
        self.questions = self.root / "test.csv"
        self.questions.write_text("image,question,answer\n1.jpg,What color?,red\n")
        (self.root / "1.jpg").write_bytes(b"fixture-image")

    def flags(self, *extra):
        return ["--questions", str(self.questions), "--images", str(self.root),
                "--llama-server", str(self.binary), "--output-dir", str(self.root / "results"),
                *extra]

    def test_default_gpu_split_and_visibility(self):
        gpu_list = Mock(stdout="GPU-first, Tesla T4, 15360 MiB\nGPU-second, Tesla T4, 15360 MiB\n")
        with patch.dict("os.environ", {}, clear=True), patch.object(runner.subprocess, "run", return_value=gpu_list):
            environment, devices = runner.gpu_environment()
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(len(devices), 2)
        for visible in ("0", "", "-1", "0,0", "0,5"):
            with self.subTest(visible=visible), patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": visible}), \
                    patch.object(runner.subprocess, "run", return_value=gpu_list), self.assertRaises(RuntimeError):
                runner.gpu_environment()
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-second,GPU-first"}), \
                patch.object(runner.subprocess, "run", return_value=gpu_list):
            environment, _ = runner.gpu_environment()
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-second,GPU-first")

    def test_server_uses_two_gpus_and_vision_projector(self):
        command = runner.server_command(self.binary, self.model, self.projector, "vision", 8080, 8192)
        for flag, value in (("--tensor-split", "1,1"), ("--split-mode", "layer"),
                            ("--parallel", "1"), ("--ctx-size", "8192"),
                            ("--mmproj", str(self.projector.resolve()))):
            self.assertEqual(command[command.index(flag) + 1], value)
        with self.assertRaises(FileNotFoundError):
            runner.server_command(self.binary, self.model, self.root / "missing", "vision", 8080, 8192)

    def test_options_keep_existing_dataset_scoring(self):
        options, args = runner.parse_args(["--quantization", "Q8_0", "--dataset", "plantexpert"])
        self.assertEqual(args.model, runner.model_identifier(runner.DEFAULT_MODEL_DIR, "Q8_0"))
        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.base_url, "http://127.0.0.1:0/v1")
        self.assertIn(args.model + "_plantexpert", str(args.output_dir))
        self.assertEqual(options.context_size, 8192)
        _, args = runner.parse_args(self.flags("--dataset", "gqa", "--limit", "5", "--resume"))
        self.assertEqual(args.max_tokens, 64)
        self.assertTrue(args.resume)
        self.assertEqual(args.limit, 5)

    def test_health_waits_through_loading(self):
        process = Mock()
        process.poll.return_value = None
        loading = urllib.error.HTTPError("local", 503, "loading", {}, io.BytesIO())
        response = Mock()
        response.__enter__ = Mock(return_value=io.StringIO('{"status":"ok"}'))
        response.__exit__ = Mock(return_value=False)
        with patch.object(runner.urllib.request, "urlopen", side_effect=[loading, response]), \
                patch.object(runner.time, "sleep") as sleep:
            runner.wait_for_server(process, 8080, 10)
        sleep.assert_called_once()
        process.poll.return_value = 1
        process.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
            runner.wait_for_server(process, 8080, 10)
        with patch.object(runner.time, "monotonic", side_effect=[0, 11]), \
                self.assertRaisesRegex(RuntimeError, "did not become ready"):
            runner.wait_for_server(process, 8080, 10)

    def test_server_cleanup_on_startup_failure_and_interruption(self):
        for error in (RuntimeError("startup failed"), KeyboardInterrupt()):
            process = Mock()
            process.poll.return_value = None
            with self.subTest(error=type(error).__name__), \
                    patch.object(runner.socket, "socket"), \
                    patch.object(runner.subprocess, "Popen", return_value=process), \
                    patch.object(runner, "wait_for_server", side_effect=error), \
                    self.assertRaises(type(error)):
                with runner.running_server(["server"], {}, self.root / "server.log", 8080, 10):
                    self.fail("Startup failure must not yield a server")
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=15)

    def test_download_pins_weights_and_projector_to_same_revision(self):
        fake_hub = Mock()
        fake_hub.HfApi.return_value.model_info.return_value.sha = "fixed-commit"
        fake_hub.HfApi.return_value.list_repo_files.return_value = [
            "Example-VL-Q4_K_M.gguf", "Example-VL-Q8_0.gguf", "mmproj-Example-VL-f16.gguf",
        ]
        fake_hub.hf_hub_download.side_effect = [str(self.model), str(self.projector)]
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            model, projector, source = runner.resolve_model("someone/Example-VL-GGUF", self.root)
        self.assertEqual((model, projector), (self.model, self.projector))
        self.assertEqual(source["revision"], "fixed-commit")
        calls = fake_hub.hf_hub_download.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.kwargs["revision"] == "fixed-commit" for call in calls))
        self.assertEqual(calls[1].kwargs["filename"], "mmproj-Example-VL-f16.gguf")
        self.assertTrue(all(call.kwargs["repo_id"] == "someone/Example-VL-GGUF" for call in calls))

    def test_local_model_needs_no_hub_and_tracks_changed_weights(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            model, projector, source = runner.resolve_model(str(self.root), self.root / "cache")
            self.model.write_bytes(b"changed-model")
            _, _, updated = runner.resolve_model(str(self.root), self.root / "cache")
        self.assertEqual(model, self.model)
        self.assertEqual(projector, self.projector)
        self.assertNotEqual(source["files_sha256"]["model.gguf"], updated["files_sha256"]["model.gguf"])
        self.assertFalse((self.root / "cache").exists())
        with self.assertRaisesRegex(FileNotFoundError, "MODEL_DIR does not exist"):
            runner.resolve_model(str(self.root / "missing"), self.root / "cache")

    def test_hf_url_and_quantization_suffix_select_new_model(self):
        fake_hub = Mock()
        fake_hub.HfApi.return_value.model_info.return_value.sha = "fixed-commit"
        fake_hub.HfApi.return_value.list_repo_files.return_value = [
            "Different-VL-Q4_K_M.gguf", "Different-VL-Q8_0.gguf", "mmproj-Different-VL-F16.gguf",
        ]
        fake_hub.hf_hub_download.side_effect = [str(self.model), str(self.projector)]
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            _, _, source = runner.resolve_model("https://huggingface.co/other/Different-GGUF:Q8_0", self.root)
        self.assertEqual(source["repo"], "other/Different-GGUF")
        self.assertEqual(source["quantization"], "Q8_0")
        self.assertEqual(fake_hub.hf_hub_download.call_args_list[0].kwargs["filename"], "Different-VL-Q8_0.gguf")
        self.assertIn(runner.model_identifier("other/Different-GGUF"),
                      fake_hub.hf_hub_download.call_args_list[0].kwargs["local_dir"])

    def test_automatic_selection_handles_official_names_and_shards(self):
        prefix = "Ministral-3-8B-Instruct-2512"
        weights = [f"{prefix}-{quant}.gguf" for quant in ("Q8_0", "BF16", "Q4_K_M", "Q5_K_M")]
        model, projector, _ = runner.select_gguf_files([*weights, f"{prefix}-BF16-mmproj.gguf"])
        self.assertEqual(model, f"{prefix}-Q4_K_M.gguf")
        self.assertEqual(projector, f"{prefix}-BF16-mmproj.gguf")
        shards = ["Qwen-VL-Q4_K_M-00001-of-00002.gguf", "Qwen-VL-Q4_K_M-00002-of-00002.gguf"]
        model, _, selected = runner.select_gguf_files([*shards, "mmproj-Qwen-VL-F16.gguf"])
        self.assertEqual(model, shards[0])
        self.assertEqual(selected, shards)
        with self.assertRaisesRegex(ValueError, "Missing GGUF weight shards"):
            runner.select_gguf_files([shards[0], "mmproj-Qwen-VL-F16.gguf"])

    def test_missing_or_ambiguous_vision_files_fail_before_download(self):
        cases = [
            (["weights.safetensors"], "GGUF model weights"),
            (["Qwen-Q4_K_M.gguf"], "vision GGUF projector"),
            (["Qwen-Q4_K_M.gguf", "Other-Q4_K_M.gguf", "mmproj.gguf"], "multiple model families"),
            (["Qwen-Q4_K_M.gguf", "copy/Qwen-Q4_K_M.gguf", "mmproj.gguf"], "unique GGUF weights"),
            (["Qwen-Q4_K_M.gguf", "mmproj-Other-F16.gguf"], "No projector matches"),
            (["Qwen-Q4_K_M.gguf", "mmproj.gguf", "copy/mmproj.gguf"], "Multiple matching vision projectors"),
        ]
        for files, error in cases:
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, error):
                runner.select_gguf_files(files)

    def test_changing_only_notebook_model_dir_changes_commands_and_outputs(self):
        cell = next(cell for cell in build_notebook()["cells"]
                    if "".join(cell["source"]).startswith("def evaluation_command():"))
        namespace = {
            "sys": sys, "LLAMA_SERVER": self.binary, "WORK": self.root, "HF_REVISION": "main",
            "CONTEXT_SIZE": 8192, "DATASET": "gqa", "QUESTIONS": self.questions,
            "IMAGES": self.root, "IMAGE_SPLIT": "test", "MAX_TOKENS": 64,
            "NO_BERTSCORE": True, "RESULTS_ROOT": self.root / "results",
            "model_identifier": runner.model_identifier, "MODEL_DIR": runner.DEFAULT_MODEL_DIR,
        }
        exec("".join(cell["source"]), namespace)
        original_output = namespace["evaluation_output"]()
        for value in ("different/Other-GGUF", str(self.root)):
            namespace["MODEL_DIR"] = value
            command = namespace["evaluation_command"]()
            self.assertEqual(command[command.index("--model-dir") + 1], value)
            self.assertNotIn("--quantization", command)
            self.assertNotEqual(namespace["evaluation_output"](), original_output)
            _, args = runner.parse_args(command[4:])
            self.assertEqual(namespace["evaluation_output"]().name, args.output_dir.name)

    def test_integration_reports_predictions_and_guards_model_revision(self):
        response = {"choices": [{"message": {"content": "red"}, "finish_reason": "stop"}]}
        client = LMStudioClient("http://127.0.0.1:8080/v1", 10, 0)
        client.models = Mock(return_value=[runner.model_identifier(runner.DEFAULT_MODEL_DIR)])
        with patch.object(runner, "gpu_environment", return_value=({"CUDA_VISIBLE_DEVICES": "0,1"}, ["T4", "T4"])), \
                patch.object(runner, "resolve_model", return_value=(self.model, self.projector, {"revision": "abc"})), \
                patch.object(runner, "available_server_port", return_value=8080), \
                patch.object(runner.subprocess, "run", return_value=Mock(stdout="version: test", stderr="")), \
                patch.object(runner, "running_server", return_value=nullcontext()), \
                patch.object(runner, "LMStudioClient", return_value=client), \
                patch.object(client, "request", return_value=response), patch("builtins.print"):
            self.assertEqual(runner.main(self.flags()), 0)
        report = json.loads((self.root / "results/report.json").read_text())
        self.assertEqual(report["accuracy"], 1.0)
        self.assertTrue(report["complete"])
        self.assertEqual(report["config"]["backend"]["model_source"]["revision"], "abc")
        _, resumed = runner.parse_args(self.flags("--resume"))
        resumed.backend_config = report["config"]["backend"]
        resumed.allow_server_port_change = True
        client.base_url = "http://127.0.0.1:8081/v1"
        with patch.object(client, "answer") as answer, patch("builtins.print"):
            resumed_report = evaluate(resumed, client)
        self.assertTrue(resumed_report["complete"])
        answer.assert_not_called()
        resumed.allow_server_port_change = False
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            evaluate(resumed, client)
        _, args = runner.parse_args(self.flags("--resume"))
        args.backend_config = report["config"]["backend"]
        args.backend_config["model_source"]["revision"] = "changed"
        args.allow_server_port_change = True
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            evaluate(args, client)

    def test_bad_dataset_stops_before_model_download(self):
        with patch.object(runner, "gpu_environment", return_value=({}, ["T4", "T4"])), \
                patch.object(runner, "resolve_model") as download, patch("builtins.print"):
            result = runner.main(self.flags("--images", str(self.root / "missing")))
        self.assertEqual(result, 1)
        download.assert_not_called()

    def full_cell(self, resume=False):
        source = next("".join(cell["source"]) for cell in build_notebook()["cells"]
                      if "".join(cell["source"]).startswith("# Rebuild the arguments"))
        namespace = {
            "evaluation_output": lambda: self.root / "results",
            "evaluation_command": lambda: [sys.executable, "-m", "scripts.evaluate_kaggle_ministral"],
            "WORK": self.root, "EVAL_ROOT": self.root, "FULL_LIMIT": None,
            "RESUME": resume, "subprocess": subprocess,
        }
        return source, namespace

    def test_full_cell_automatically_resumes_existing_predictions(self):
        output = self.root / "results/full"
        output.mkdir(parents=True)
        (output / "run_config.json").write_text('{"model":"unchanged"}')
        predictions = output / "predictions.jsonl"
        predictions.write_text('{"question_id":"1","prediction":"red"}\n')
        source, namespace = self.full_cell()
        with patch.object(subprocess, "run") as run, patch("builtins.print"):
            exec(source, namespace)
        command = run.call_args.args[0]
        self.assertEqual(command.count("--resume"), 1)
        self.assertEqual(command[command.index("--output-dir") + 1], str(output))
        self.assertEqual(predictions.read_text(), '{"question_id":"1","prediction":"red"}\n')

    def test_full_cell_starts_new_runs_and_preserves_unrecognized_outputs(self):
        source, namespace = self.full_cell()
        with patch.object(subprocess, "run") as run, patch("builtins.print"):
            exec(source, namespace)
        self.assertNotIn("--resume", run.call_args.args[0])
        output = self.root / "results/full"
        output.mkdir(parents=True)
        artifact = output / "predictions.jsonl"
        artifact.write_text("saved output")
        with patch.object(subprocess, "run") as run, \
                self.assertRaisesRegex(ValueError, "no run_config.json"):
            exec(source, namespace)
        run.assert_not_called()
        self.assertEqual(artifact.read_text(), "saved output")

    def test_occupied_port_selects_another_without_touching_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen()
            occupied = server.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "occupied.*--server-port 0"):
                runner.available_server_port(occupied)
            selected = runner.available_server_port()
            self.assertNotEqual(selected, occupied)
            self.assertGreater(selected, 0)
            with socket.create_connection(("127.0.0.1", occupied), timeout=2) as client:
                connection, _ = server.accept()
                connection.close()

    def test_port_is_reusable_immediately_after_server_connection_closes(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                connection, _ = server.accept()
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
                connection.close()
        self.assertEqual(runner.available_server_port(port), port)

    def test_notebook_cells_compile_and_bundle_current_sources(self):
        notebook = build_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        bundle = None
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"notebook-cell-{index}", "exec")
            self.assertEqual(cell["outputs"], [])
            if source.startswith("BUNDLED_SOURCES ="):
                bundle = ast.literal_eval(ast.parse(source).body[0].value)
        self.assertEqual(set(bundle), set(BUNDLED_FILES))
        for name in BUNDLED_FILES:
            self.assertEqual(bundle[name], (ROOT / name).read_text())
        committed = json.loads((ROOT / "notebooks/evaluate_ministral_3_8b_kaggle.ipynb").read_text())
        self.assertEqual(committed, notebook)

    def test_kaggle_build_avoids_missing_driver_target_and_reuses_clone(self):
        source = next("".join(cell["source"]) for cell in build_notebook()["cells"]
                      if "".join(cell["source"]).startswith("LLAMA_ROOT = WORK"))
        (self.root / "llama.cpp").mkdir()
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[0] == "cmake" and "-S" in command:
                if "-DGGML_CUDA_NO_VMM=ON" not in command:
                    raise subprocess.CalledProcessError(1, command, stderr="CUDA::cuda_driver target was not found")
            return Mock(returncode=0)

        namespace = {
            "WORK": self.root, "LLAMA_CPP_REF": "v0.6.0", "CUDA_ARCHS": "75",
            "subprocess": subprocess, "os": Mock(cpu_count=Mock(return_value=4)),
        }
        with patch.object(subprocess, "run", side_effect=run), \
                patch.object(subprocess, "check_output", return_value="pinned-commit\n"), \
                patch("builtins.print"):
            exec(source, namespace)
        self.assertEqual(len(commands), 3)
        self.assertIn("-DGGML_CUDA=ON", commands[0])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=75", commands[0])
        self.assertIn("--build", commands[1])
        self.assertEqual(commands[2][1], "--list-devices")
        self.assertFalse(any(command[0] == "git" for command in commands))


if __name__ == "__main__":
    unittest.main()
