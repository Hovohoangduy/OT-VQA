"""Generate a self-contained Kaggle notebook from the current evaluator sources.

Run: python3 -m scripts.build_kaggle_notebook
"""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
BUNDLED_FILES = (
    "scripts/evaluate_gqa_lmstudio.py", "scripts/evaluate_kaggle_ministral.py",
    "utils/metrics.py",
)


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": dedent(source).lstrip().splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    cells = [cell("markdown", """
        # GGUF vision models: zero-shot VQA evaluation on Kaggle, two GPUs

        Import this notebook into Kaggle. Set **Accelerator → GPU T4 x2**, enable
        **Internet**, and attach your dataset. Edit the configuration below, then
        run all cells. The notebook bundles the repository evaluator; no repository
        clone or upload is required.

        Defaults use **PlantExpertVQA** and its existing Kaggle dataset path.
        For **GQA**, set `DATASET = "gqa"` and change `QUESTIONS` and `IMAGES`.
        Questions accept repository CSV (`image,question,answer`) or official GQA
        question JSON. Answers are optional; unlabeled data cannot produce accuracy.

        **To switch models, change only `MODEL_DIR`.** Use a Hugging Face GGUF
        repository ID (or its URL) or a local Kaggle folder containing the GGUF
        weights and matching vision projector. The runner discovers file names,
        selects a quantization, and creates a separate results folder per model.
        The default is the [official Ministral GGUF repository](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/tree/main).
        Automatic quantization prefers Q4_K_M, then Q5_K_M, then Q8_0. A single
        available quantization is also accepted. For a specific HF quantization,
        use `MODEL_DIR = "owner/model-GGUF:Q8_0"`. Local folders must contain one
        model family and its projector (`mmproj` or `projector` in the file name).
        VQA requires a vision model supported by the selected llama.cpp version.
        [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/v0.6.0/tools/server/README.md)
        splits layers and KV cache across GPUs with `--split-mode layer --tensor-split 1,1`.
        This is one model distributed over two GPUs; questions run sequentially.
        A five-question smoke test checks the workflow before the full evaluation.
        It does not establish benchmark performance.
    """), cell("code", """
        from pathlib import Path
        import json
        import os
        import subprocess
        import sys

        DATASET = "plantexpert"  # "plantexpert" or "gqa"
        DATA_ROOT = Path("/kaggle/input/datasets/duyho0511chill/plantexpert-dataset/plantexpert_dataset")
        QUESTIONS = DATA_ROOT / "test.csv"
        IMAGES = DATA_ROOT / "images"
        IMAGE_SPLIT = "test"  # Used for flat image names in a split subfolder.
        MODEL_DIR = "mistralai/Ministral-3-8B-Instruct-2512-GGUF"  # Only model setting to change.
        # Local example: MODEL_DIR = "/kaggle/input/my-vision-model/gguf"
        HF_REVISION = "main"  # Prefer a fixed HF commit for repeated comparisons.
        LLAMA_CPP_REF = "v0.6.0"
        CONTEXT_SIZE = 8192
        MAX_TOKENS = 256 if DATASET == "plantexpert" else 64
        NO_BERTSCORE = False  # PlantExpert only: True skips the semantic metric.
        RESUME = False  # Set True to resume the FULL run with identical settings.
        FULL_LIMIT = None  # None = all questions; positive integer = first N.
        WORK = Path("/kaggle/working")
        EVAL_ROOT = WORK / "ot-vqa-eval"
        RESULTS_ROOT = WORK / "results"  # Model-specific folder is computed automatically.
        assert DATASET in {"gqa", "plantexpert"}
        assert QUESTIONS.is_file(), f"Set QUESTIONS to your attached dataset: {QUESTIONS}"
        assert IMAGES.is_dir(), f"Set IMAGES to your attached dataset: {IMAGES}"
        print("Model:", MODEL_DIR, "\\nQuestions:", QUESTIONS, "\\nImages:", IMAGES)
    """), cell("markdown", """
        ## Install evaluation dependencies and check both GPUs

        PlantExpert BERTScore runs on CPU to leave GPU memory for inference.
        GQA scoring needs only Python's standard library. The first BERTScore run
        downloads its scoring encoder; the GGUF download also needs Internet.
    """), cell("code", """
        packages = ["huggingface-hub>=0.34,<2"]
        if DATASET == "plantexpert" and not NO_BERTSCORE:
            packages.append("bert-score==0.3.13")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
        import torch
        assert torch.cuda.device_count() >= 2, "Select GPU T4 x2 in Kaggle settings."
        for index in range(2):
            device = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {device.name}, {device.total_memory / 2**30:.1f} GiB")
        CUDA_ARCHS = ";".join(sorted({
            "".join(map(str, torch.cuda.get_device_capability(index))) for index in range(2)
        }))
    """), cell("markdown", """
        ## Restore the bundled evaluation scripts

        These files are a snapshot of this repository, written into Kaggle's
        writable working directory. Regenerate the notebook after editing the
        evaluator with `python3 -m scripts.build_kaggle_notebook`.
    """)]
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in BUNDLED_FILES}
    cells.append(cell("code", "BUNDLED_SOURCES = " + repr(sources) + "\n" + dedent("""
        EVAL_ROOT.mkdir(parents=True, exist_ok=True)
        for name, source in BUNDLED_SOURCES.items():
            destination = EVAL_ROOT / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source, encoding="utf-8")
        for package in ("scripts", "utils"):
            (EVAL_ROOT / package / "__init__.py").write_text("", encoding="utf-8")
        if str(EVAL_ROOT) not in sys.path:
            sys.path.insert(0, str(EVAL_ROOT))
        from scripts.evaluate_kaggle_ministral import model_identifier
        print("Evaluation scripts:", EVAL_ROOT)
    """)))
    cells.extend([cell("markdown", """
        ## Build llama.cpp with CUDA

        Compilation can take several minutes and is reused when rerunning cells
        in the same session. The build targets your GPUs' compute capability.
        Build errors stop this cell; evaluation never silently switches to CPU.
    """), cell("code", """
        LLAMA_ROOT = WORK / "llama.cpp"
        if not LLAMA_ROOT.exists():
            subprocess.run([
                "git", "clone", "--depth", "1", "--branch", LLAMA_CPP_REF,
                "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_ROOT),
            ], check=True)
        actual_commit = subprocess.check_output(
            ["git", "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"], text=True).strip()
        requested_commit = subprocess.check_output(
            ["git", "-C", str(LLAMA_ROOT), "rev-parse", f"{LLAMA_CPP_REF}^{{commit}}"], text=True).strip()
        assert actual_commit == requested_commit, "Use a fresh working session after changing LLAMA_CPP_REF."
        subprocess.run([
            "cmake", "-S", str(LLAMA_ROOT), "-B", str(LLAMA_ROOT / "build"),
            "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHS}",
            "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_OPENSSL=OFF",
        ], check=True)
        subprocess.run([
            "cmake", "--build", str(LLAMA_ROOT / "build"), "--target", "llama-server",
            "--parallel", str(min(os.cpu_count() or 2, 4)),
        ], check=True)
        LLAMA_SERVER = LLAMA_ROOT / "build/bin/llama-server"
        subprocess.run([str(LLAMA_SERVER), "--list-devices"], check=True)
        print("Built llama.cpp commit:", actual_commit)
    """), cell("markdown", """
        ## Run five questions first

        The runner resolves local GGUF files or downloads them, starts a vision server, and
        shuts it down when evaluation finishes or fails. Check predictions and
        errors before relying on the full results. Smoke and full outputs use
        separate directories; rerunning a smoke test resumes saved successes.
        The first cell below defines command arguments. Run it even when skipping
        the smoke test during resume.
    """), cell("code", """
        def evaluation_command():
            command = [
                sys.executable, "-u", "-m", "scripts.evaluate_kaggle_ministral",
                "--llama-server", str(LLAMA_SERVER), "--model-dir", str(MODEL_DIR),
                "--cache-dir", str(WORK / "gguf-models"), "--hf-revision", HF_REVISION,
                "--context-size", str(CONTEXT_SIZE), "--dataset", DATASET,
                "--questions", str(QUESTIONS), "--images", str(IMAGES),
                "--image-split", IMAGE_SPLIT, "--max-tokens", str(MAX_TOKENS),
                "--bertscore-device", "cpu", "--timeout", "300",
            ]
            if NO_BERTSCORE:
                command.append("--no-bertscore")
            return command

        def evaluation_output():
            return RESULTS_ROOT / f"{model_identifier(str(MODEL_DIR))}_{DATASET}"
    """), cell("code", """
        smoke_output = evaluation_output() / "smoke"
        smoke = evaluation_command() + ["--limit", "5", "--output-dir", str(smoke_output),
                          "--server-log", str(WORK / "llama-smoke.log")]
        if (smoke_output / "run_config.json").is_file():
            smoke.append("--resume")
        subprocess.run(smoke, cwd=EVAL_ROOT, check=True)
        smoke_report = json.loads((smoke_output / "report.json").read_text())
        assert smoke_report["complete"] and smoke_report["errors"] == 0
        print((smoke_output / "predictions.csv").read_text())
    """), cell("markdown", """
        ## Evaluate the full split

        Each request uses only the image, question, and dataset instruction:
        zero-shot, temperature 0, seed 42. GQA reports exact-match accuracy and
        normalized accuracy; PlantExpert reports EM, Token-F1, BLEU-1/2,
        ROUGE-L and BERTScore-F1. Predictions flush to JSONL after every request.

        To resume, set `RESUME = True` and rerun the configuration, argument-setup
        and full-run cells with the same dataset, paths, quantization, revisions
        and settings. In a fresh session, run the dependency, bundled-script and
        llama.cpp build cells too.
        After a new Kaggle session, attach saved outputs and **copy the full run
        directory back to the original `/kaggle/working` location** before resuming.
        The smoke cell may be skipped when resuming. `run_config.json` records the
        resolved HF commit, llama.cpp version and GPU split and rejects changed settings.
    """), cell("code", """
        # Rebuild the arguments so configuration edits also apply when resuming.
        full_output = evaluation_output() / "full"
        full = evaluation_command() + ["--output-dir", str(full_output),
                         "--server-log", str(WORK / "llama-full.log")]
        if FULL_LIMIT is not None:
            full += ["--limit", str(FULL_LIMIT)]
        if RESUME:
            full.append("--resume")
        subprocess.run(full, cwd=EVAL_ROOT, check=True)
    """), cell("markdown", """
        ## Inspect and download results

        Save a Kaggle notebook version to preserve output files. The archive
        contains both runs' reports, CSV predictions, raw JSONL responses and
        settings. GQA additionally exports `gqa_predictions.json`.

        If inference fails, inspect `/kaggle/working/llama-full.log`. Responses
        ending at the token limit are counted in `truncated_responses`; a high
        count means a fresh run with a larger `MAX_TOKENS` may be needed.
    """), cell("code", """
        import shutil
        from IPython.display import FileLink, display

        report = json.loads((full_output / "report.json").read_text())
        print(json.dumps(report, indent=2))
        subprocess.run(["nvidia-smi"], check=True)  # Server has shut down; memory is released.
        archive = shutil.make_archive(str(WORK / f"{model_identifier(str(MODEL_DIR))}_vqa_results"),
                                      "zip", evaluation_output())
        display(FileLink(archive))
    """)])
    return {
        "cells": cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        }, "nbformat": 4, "nbformat_minor": 4,
    }


if __name__ == "__main__":
    destination = ROOT / "notebooks/evaluate_ministral_3_8b_kaggle.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_notebook(), indent=1) + "\n", encoding="utf-8")
    print(destination)
