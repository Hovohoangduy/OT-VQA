"""Generate a self-contained Kaggle notebook from the current evaluator sources.

Run: python3 -m scripts.build_kaggle_notebook
"""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
BUNDLED_FILES = (
    "scripts/evaluate_gqa_lmstudio.py", "scripts/evaluate_kaggle_ministral.py",
    "scripts/evaluate_kaggle_vlm.py",
    "scripts/keye_sdpa.py",
    "utils/metrics.py",
)


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": dedent(source).lstrip().splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    cells = [cell("markdown", """
        # Vision models: zero-shot VQA evaluation on Kaggle, two GPUs

        Import this notebook into Kaggle. Set **Accelerator → GPU T4 x2**, enable
        **Internet**, and attach your dataset. Edit the configuration below, then
        run all cells. The notebook bundles the repository evaluator; no repository
        clone or upload is required.

        Defaults use **PlantExpertVQA** and its existing Kaggle dataset path.
        For **GQA**, set `DATASET = "gqa"` and change `QUESTIONS` and `IMAGES`.
        Questions accept repository CSV (`image,question,answer`) or official GQA
        question JSON. Answers are optional; unlabeled data cannot produce accuracy.

        **To switch models, change only `MODEL_DIR`.** Use a Hugging Face
        repository ID (or its URL) or a local Kaggle model folder. The runner
        detects GGUF versus Transformers weights and creates a separate results
        folder per model. GGUF uses llama.cpp; native Transformers vision models
        use `AutoProcessor` and `AutoModelForImageTextToText` with Accelerate.
        The default is the [official Ministral GGUF repository](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/tree/main).
        Automatic quantization prefers Q4_K_M, then Q5_K_M, then Q8_0. A single
        available quantization is also accepted. For a specific HF quantization,
        use `MODEL_DIR = "owner/model-GGUF:Q8_0"`. Local folders must contain one
        model family and its projector (`mmproj` or `projector` in the file name).
        VQA requires a vision model supported by the selected llama.cpp version.
        For the original CapRL InternVL3.5 8B Safetensors weights, set
        `MODEL_DIR = "internlm/CapRL-InternVL3.5-8B"`. Its current repository
        provides the native [InternVL configuration](https://huggingface.co/internlm/CapRL-InternVL3.5-8B/blob/main/config.json).
        The llama.cpp build is skipped for Transformers models. FP16 is used on
        T4 GPUs; Accelerate balances weights across the two GPUs with a 12 GiB
        weight budget per GPU, leaving room for images and generation. Weights
        must fit on the GPUs. Native Transformers models require a multimodal
        chat processor; custom architectures may require additional integration.
        Custom models such as Keye-VL-1.5 require `TRUST_REMOTE_CODE = True`.
        Keye uses its registered `AutoModel` and separate image preprocessing.
        With `ATTENTION_IMPLEMENTATION = "auto"` or `"sdpa"`, the runner stages
        a local adaptation of Keye's custom code using PyTorch SDPA, including
        rotary embeddings and separate attention for each packed image. This
        path supports T4 x2 without installing FlashAttention. Original cached
        files and weights are unchanged, and adaptation hashes are saved for
        resume checks. Explicit `"flash_attention_2"` uses upstream code and
        requires Ampere-or-newer GPUs. Run the smoke test on your hardware first.
        [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/v0.6.0/tools/server/README.md)
        splits GGUF layers and KV cache across GPUs with `--split-mode layer --tensor-split 1,1`.
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
        # CapRL example: MODEL_DIR = "Jobaar/CapRL-InternVL3.5-8B-GGUF:Q4_K_M"
        # Original Safetensors example: MODEL_DIR = "internlm/CapRL-InternVL3.5-8B"
        # Keye on T4 x2: MODEL_DIR = "Kwai-Keye/Keye-VL-1_5-8B"; also set TRUST_REMOTE_CODE = True.
        BACKEND = "auto"  # Optional override: "gguf" or "transformers".
        TRUST_REMOTE_CODE = False  # Set True for a model that requires repository Python code.
        ATTENTION_IMPLEMENTATION = "auto"  # SDPA, including the Keye-VL-1.5 T4 adaptation.
        HF_REVISION = "main"  # Prefer a fixed HF commit for repeated comparisons.
        LLAMA_CPP_REF = "v0.6.0"
        CONTEXT_SIZE = 8192
        MAX_TOKENS = 256 if DATASET == "plantexpert" else 64
        NO_BERTSCORE = False  # PlantExpert only: True skips the semantic metric.
        RESUME = False  # Existing full runs resume automatically; True requires a saved run.
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
        downloads its scoring encoder; model downloads also need Internet.
    """), cell("code", """
        packages = ["huggingface-hub>=0.34,<1"]
        if DATASET == "plantexpert" and not NO_BERTSCORE:
            packages += ["bert-score==0.3.13", "transformers==4.57.1"]
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
        from scripts.evaluate_kaggle_vlm import (
            inspect_model, read_transformers_config, transformers_loading_plan,
            validate_attention_hardware, run_logged_evaluation,
        )
        print("Evaluation scripts:", EVAL_ROOT)
    """)))
    cells.extend([cell("markdown", """
        ## Select the backend

        Inspect model metadata before downloading weights or compiling llama.cpp.
        Re-run this cell after changing MODEL_DIR or BACKEND. When a repository
        contains both formats, auto prefers GGUF; use BACKEND to override this.
        Transformers models are loaded in a separate process so inference GPU
        memory is released when each run ends. No local HTTP server is needed.
    """), cell("code", """
        SELECTED_MODEL = inspect_model(str(MODEL_DIR), HF_REVISION, BACKEND)
        SELECTED_BACKEND = SELECTED_MODEL["backend"]
        SELECTED_MODEL_DIR = str(MODEL_DIR)
        SELECTED_HF_REVISION = HF_REVISION
        SELECTED_BACKEND_SETTING = BACKEND
        SELECTED_TRUST_REMOTE_CODE = TRUST_REMOTE_CODE
        SELECTED_ATTENTION_IMPLEMENTATION = ATTENTION_IMPLEMENTATION
        print("Backend:", SELECTED_BACKEND)
        if "revision" in SELECTED_MODEL:
            print("Pinned model revision:", SELECTED_MODEL["revision"])
        if SELECTED_BACKEND == "transformers":
            MODEL_CONFIG = read_transformers_config(SELECTED_MODEL, WORK / "hf-models")
            LOADING_PLAN = transformers_loading_plan(MODEL_CONFIG, TRUST_REMOTE_CODE, ATTENTION_IMPLEMENTATION)
            validate_attention_hardware(torch, LOADING_PLAN["attention"], check_package=False)
            packages = [
                "transformers==4.57.1", "accelerate>=1,<2", "pillow>=10", "safetensors>=0.4.5",
            ]
            if LOADING_PLAN["image_processing"] == "keye-vl-1.5":
                packages.append("einops>=0.8")
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
            if LOADING_PLAN["attention"] == "flash_attention_2":
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                                "flash-attn>=2.6,<3", "--no-build-isolation"], check=True)
    """), cell("markdown", """
        ## Build llama.cpp with CUDA

        This cell skips compilation when the selected backend is Transformers.
        Compilation can take several minutes and is reused when rerunning cells
        in the same session. The build targets your GPUs' compute capability.
        Build errors stop this cell; evaluation never silently switches to CPU.
        CUDA virtual memory management is disabled to avoid Kaggle's missing
        `CUDA::cuda_driver` CMake target. The CUDA backend and GPU splitting remain enabled.
    """), cell("code", """
        LLAMA_ROOT = WORK / "llama.cpp"
        LLAMA_SERVER = LLAMA_ROOT / "build/bin/llama-server"
        if SELECTED_BACKEND == "transformers":
            print("Transformers backend: skipping llama.cpp build.")
        else:
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
                "-DGGML_CUDA=ON", "-DGGML_CUDA_NO_VMM=ON", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHS}",
                "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_OPENSSL=OFF",
            ], check=True)
            subprocess.run([
                "cmake", "--build", str(LLAMA_ROOT / "build"), "--target", "llama-server",
                "--parallel", str(min(os.cpu_count() or 2, 4)),
            ], check=True)
            subprocess.run([str(LLAMA_SERVER), "--list-devices"], check=True)
            print("Built llama.cpp commit:", actual_commit)
    """), cell("markdown", """
        ## Run five questions first

        The runner loads local files or downloads weights and starts the selected
        backend. Check predictions and
        errors before relying on the full results. Smoke and full outputs use
        separate directories; rerunning a smoke test resumes saved successes.
        Each run selects a free local server port automatically, so the full run
        can start immediately after smoke. Resume accepts port changes while
        retaining the checks on model and evaluation settings.
        The first cell below defines command arguments. Run it even when skipping
        the smoke test during resume.
    """), cell("code", """
        def evaluation_command():
            assert (str(MODEL_DIR), HF_REVISION, BACKEND) == (
                SELECTED_MODEL_DIR, SELECTED_HF_REVISION, SELECTED_BACKEND_SETTING
            ), "Model settings changed. Re-run the backend selection and build cells."
            assert (TRUST_REMOTE_CODE, ATTENTION_IMPLEMENTATION) == (
                SELECTED_TRUST_REMOTE_CODE, SELECTED_ATTENTION_IMPLEMENTATION
            ), "Loading settings changed. Re-run the backend selection cell."
            command = [
                sys.executable, "-u", "-m", "scripts.evaluate_kaggle_vlm",
                "--backend", SELECTED_BACKEND,
                "--model-dir", str(MODEL_DIR),
                "--cache-dir", str(WORK / ("gguf-models" if SELECTED_BACKEND == "gguf" else "hf-models")),
                "--hf-revision", SELECTED_MODEL.get("revision", HF_REVISION),
                "--context-size", str(CONTEXT_SIZE), "--dataset", DATASET,
                "--questions", str(QUESTIONS), "--images", str(IMAGES),
                "--image-split", IMAGE_SPLIT, "--max-tokens", str(MAX_TOKENS),
                "--bertscore-device", "cpu", "--timeout", "300",
            ]
            if SELECTED_BACKEND == "gguf":
                command += ["--llama-server", str(LLAMA_SERVER)]
            else:
                command += ["--attention-implementation", ATTENTION_IMPLEMENTATION]
                if TRUST_REMOTE_CODE:
                    command.append("--trust-remote-code")
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
        run_logged_evaluation(smoke, EVAL_ROOT, WORK / "vlm-smoke.log")
        smoke_report = json.loads((smoke_output / "report.json").read_text())
        assert smoke_report["complete"] and smoke_report["errors"] == 0
        print((smoke_output / "predictions.csv").read_text())
    """), cell("markdown", """
        ## Evaluate the full split

        Each request uses only the image, question, and dataset instruction:
        zero-shot, temperature 0, seed 42. GQA reports exact-match accuracy and
        normalized accuracy; PlantExpert reports EM, Token-F1, BLEU-1/2,
        ROUGE-L and BERTScore-F1. Predictions flush to JSONL after every request.

        Existing full runs resume automatically when `run_config.json` is present.
        Rerun the configuration, argument-setup and full-run cells with the same
        dataset, paths, quantization, revisions
        and settings. In a fresh session, run the dependency, bundled-script and
        backend selection and build cells too (the build is skipped for Transformers).
        After a new Kaggle session, attach saved outputs and **copy the full run
        directory back to the original `/kaggle/working` location** before resuming.
        The smoke cell may be skipped when resuming. `run_config.json` records the
        resolved HF commit, backend versions and GPU placement and rejects changed settings.
    """), cell("code", """
        # Rebuild the arguments so configuration edits also apply when resuming.
        full_output = evaluation_output() / "full"
        saved_run = (full_output / "run_config.json").is_file()
        if full_output.exists() and any(full_output.iterdir()) and not saved_run:
            raise ValueError(
                f"{full_output} contains files but no run_config.json. "
                "Preserve those files and choose a new output directory before starting."
            )
        full = evaluation_command() + ["--output-dir", str(full_output),
                         "--server-log", str(WORK / "llama-full.log")]
        if FULL_LIMIT is not None:
            full += ["--limit", str(FULL_LIMIT)]
        if RESUME or saved_run:
            full.append("--resume")
            print("Resuming saved run:", full_output)
        else:
            print("Starting new run:", full_output)
        run_logged_evaluation(full, EVAL_ROOT, WORK / "vlm-full.log")
    """), cell("markdown", """
        ## Inspect and download results

        Save a Kaggle notebook version to preserve output files. The archive
        contains both runs' reports, CSV predictions, raw JSONL responses and
        settings. GQA additionally exports `gqa_predictions.json`.

        If inference fails, read the evaluator's `Error:` line above the traceback.
        Both stdout and stderr are streamed and saved to `vlm-smoke.log` or
        `vlm-full.log`; failures include the last 30 lines in the notebook exception.
        GGUF server failures additionally use `/kaggle/working/llama-full.log`. Responses
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
