"""Evaluate GGUF or native Transformers vision models on two Kaggle GPUs.

The backend is detected from model files. GGUF runs use the existing llama.cpp
runner; Transformers runs use AutoProcessor and AutoModelForImageTextToText.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from scripts import evaluate_kaggle_ministral as gguf
from scripts.evaluate_gqa_lmstudio import SYSTEM_PROMPT, evaluate, image_path, load_examples
from utils.metrics import normalize_llm_answer


def read_transformers_config(source: dict, cache_dir: Path) -> dict:
    """Read only config.json; never import custom code or download weights here."""
    if "directory" in source:
        path = Path(source["directory"]) / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=source["repo"], filename="config.json",
                                   revision=source["revision"], cache_dir=str(cache_dir)))
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The model's config.json must be a JSON object.")
    return config


def transformers_loading_plan(config: dict, trust_remote_code: bool,
                              attention: str = "auto") -> dict:
    """Choose the registered model class and known preprocessing requirements."""
    keye = str(config.get("model_type", "")).lower() == "keyevl1_5"
    auto_map = config.get("auto_map", {})
    loader = "AutoModelForImageTextToText"
    if keye and not trust_remote_code:
        raise ValueError(
            "Keye-VL-1.5 requires custom model/processor code. Set TRUST_REMOTE_CODE = True "
            "in the notebook, or pass --trust-remote-code. On T4 x2, keep "
            "ATTENTION_IMPLEMENTATION = 'auto' (or 'sdpa') to use the SDPA adaptation."
        )
    if trust_remote_code:
        for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            if name in auto_map:
                loader = name
                break
    if keye:
        # Match the model author's documented entry point, rather than a native auto class.
        if "AutoModel" not in auto_map:
            raise ValueError("Keye-VL-1.5 config is missing its AutoModel registration.")
        loader = "AutoModel"
        if attention not in {"auto", "sdpa", "flash_attention_2"}:
            raise ValueError("Keye-VL-1.5 supports the SDPA adaptation or upstream FlashAttention-2; use auto, sdpa, or flash_attention_2.")
    if attention == "auto":
        attention = "sdpa"
    return {"loader": loader, "attention": attention,
            "image_processing": "keye-vl-1.5" if keye else "native-chat-template"}


def validate_attention_hardware(torch, attention: str, check_package: bool = True):
    if attention != "flash_attention_2":
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two visible NVIDIA GPUs are required for this runner.")
    if any(torch.cuda.get_device_capability(i)[0] < 8 for i in range(2)):
        raise RuntimeError(
            "FlashAttention-2 requires Ampere-or-newer GPUs. Kaggle T4 x2 is unsupported "
            "for upstream FlashAttention-2. For Keye-VL-1.5, select auto or sdpa to use "
            "the SDPA adaptation. Installing flash-attn alone does not fix T4 compatibility."
        )
    if check_package and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("FlashAttention-2 is missing. Install flash-attn --no-build-isolation on compatible GPUs.")


def run_logged_evaluation(command: list[str], cwd: Path, log_path: Path):
    """Stream both subprocess outputs and include the actual error in notebook failures."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=30)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line)
                print(line, end="", flush=True)
            code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if code:
        raise RuntimeError(f"Evaluation failed (exit code {code}). Log: {log_path}\n" + "".join(tail))


def inspect_model(model_dir: str, revision: str = "main", backend: str = "auto",
                  quantization: str | None = None) -> dict:
    """Inspect local files or pinned HF metadata without downloading weights."""
    local = Path(model_dir).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError("MODEL_DIR must be a model folder or Hugging Face repository.")
        names = [path.relative_to(local).as_posix() for path in local.rglob("*") if path.is_file()]
        source = {"directory": str(local.resolve()), "names": names}
    else:
        if local.is_absolute() or str(model_dir).startswith((".", "~")):
            raise FileNotFoundError(f"Local MODEL_DIR does not exist: {local}")
        repo = str(model_dir).rstrip("/")
        if repo.startswith("https://huggingface.co/"):
            repo = urlparse(repo).path.strip("/")
        repo, separator, selected_quantization = repo.partition(":")
        if separator:
            if quantization and quantization.upper() != selected_quantization.upper():
                raise ValueError("Conflicting quantizations in MODEL_DIR and --quantization")
            quantization = selected_quantization
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError("MODEL_DIR must be a local model folder or HF owner/model repository.")
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError("Install huggingface-hub to inspect Hugging Face models.") from error
        api = HfApi()
        resolved = api.model_info(repo, revision=revision).sha
        names = api.list_repo_files(repo, revision=resolved)
        source = {"repo": repo, "revision": resolved, "names": names}
    has_gguf = any(name.lower().endswith(".gguf") for name in names)
    has_transformers = "config.json" in names and any(
        name.lower().endswith(".safetensors") or (Path(name).name.startswith("pytorch_model")
        and name.lower().endswith(".bin")) for name in names
    )
    if backend == "auto":
        backend = "gguf" if has_gguf else "transformers" if has_transformers else "auto"
    if backend == "gguf" and has_gguf:
        # Also validate projector pairing before compiling llama.cpp in the notebook.
        gguf.select_gguf_files(names, quantization)
    elif backend == "transformers" and has_transformers:
        if quantization:
            raise ValueError("The :Q4_K_M suffix selects GGUF quantization; omit it for Transformers weights.")
    else:
        raise ValueError(
            f"MODEL_DIR has no files for backend {backend!r}. Expected GGUF weights with a "
            "vision projector, or config.json with Transformers model weights."
        )
    return {**source, "backend": backend}


def resolve_transformers_model(source: dict, cache_dir: Path, trust_remote_code: bool) -> tuple[Path, dict]:
    """Pin the processor and all weight shards to the same revision."""
    suffixes = (".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".tiktoken")
    names = [name for name in source["names"] if name.lower().endswith(suffixes)
             or trust_remote_code and name.lower().endswith(".py")]
    if "directory" in source:
        directory = Path(source["directory"])
        checksums = {}
        for name in sorted(names):
            digest = hashlib.sha256()
            with (directory / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[name] = digest.hexdigest()
        identity = {"directory": str(directory), "files_sha256": checksums}
    else:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(
            repo_id=source["repo"], revision=source["revision"],
            cache_dir=str(cache_dir), allow_patterns=names,
        ))
        identity = {"repo": source["repo"], "revision": source["revision"]}
    return directory, identity


class TransformersClient:
    base_url = "transformers://local"

    def __init__(self, directory: Path, alias: str, dtype: str = "auto",
                 gpu_memory_gib: float = 12, context_size: int = 8192,
                 trust_remote_code: bool = False, attention: str = "auto"):
        try:
            import torch
            import accelerate
            import transformers
            from transformers import AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install transformers==4.57.1, accelerate, pillow, and safetensors.") from error
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Two visible NVIDIA GPUs are required; choose GPU T4 x2 in Kaggle.")
        config = read_transformers_config({"directory": str(directory)}, directory)
        plan = transformers_loading_plan(config, trust_remote_code, attention)
        validate_attention_hardware(torch, plan["attention"])
        if dtype == "auto":
            dtype = "bfloat16" if all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(2)) else "float16"
        if dtype == "bfloat16" and any(torch.cuda.get_device_capability(i)[0] < 8 for i in range(2)):
            raise ValueError("bfloat16 requires newer GPUs; use float16 for Kaggle T4 GPUs.")
        max_memory = {i: int(gpu_memory_gib * 2**30) for i in range(2)}
        max_memory["cpu"] = 0  # Fail if weights cannot fit, instead of silently offloading.
        self.torch = torch
        self.alias = alias
        self.context_size = context_size
        self.image_processing = plan["image_processing"]
        adapter_identity = {}
        if self.image_processing == "keye-vl-1.5" and plan["attention"] == "sdpa":
            from scripts.keye_sdpa import stage_keye_sdpa
            self._model_code_directory, directory, adapter_identity = stage_keye_sdpa(directory)
        self.processor = AutoProcessor.from_pretrained(
            str(directory), trust_remote_code=trust_remote_code, local_files_only=True,
        )
        loader = getattr(transformers, plan["loader"])
        self.model = loader.from_pretrained(
            str(directory), torch_dtype=getattr(torch, dtype), device_map="balanced",
            max_memory=max_memory, attn_implementation=plan["attention"],
            trust_remote_code=trust_remote_code, local_files_only=True,
        ).eval()
        placement = getattr(self.model, "hf_device_map", {})
        if not placement or any(str(device) not in {"0", "1", "cuda:0", "cuda:1"}
                                for device in placement.values()):
            raise RuntimeError("Model weights do not fit on the two GPUs. Use a smaller model or GGUF quantization.")
        self.backend_config = {
            "engine": "transformers", "version": transformers.__version__,
            "torch_version": torch.__version__, "accelerate_version": accelerate.__version__,
            "dtype": dtype, "device_map_strategy": "balanced",
            "device_map": {name: str(device) for name, device in placement.items()},
            "gpu_memory_gib": gpu_memory_gib, "context_size": context_size,
            "attention_implementation": plan["attention"], "trust_remote_code": trust_remote_code,
            **adapter_identity,
        }
        # Preserve the existing native manifest shape so saved CapRL runs still resume.
        if plan["loader"] != "AutoModelForImageTextToText" or self.image_processing != "native-chat-template":
            self.backend_config.update({"model_loader": plan["loader"],
                                        "image_processing": self.image_processing,
                                        "question_suffix": "/no_think" if self.image_processing == "keye-vl-1.5" else ""})

    def models(self) -> list[str]:
        return [self.alias]

    def answer(self, model, example, path, max_tokens, seed, system_prompt=SYSTEM_PROMPT) -> dict:
        if model != self.alias:
            raise ValueError(f"Loaded model is {self.alias}, not {model}.")
        start = time.perf_counter()
        # A local file path lets each processor perform its model-specific image preprocessing.
        keye = self.image_processing == "keye-vl-1.5"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image" if keye else "url": str(path.resolve())},
                {"type": "text", "text": example.question + ("/no_think" if keye else "")},
            ]},
        ]
        if keye:
            from PIL import Image
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # Keye's custom processor requires image inputs separately from the rendered prompt.
            with Image.open(path) as image:
                inputs = self.processor(text=[text], images=[image.convert("RGB")],
                                        padding=True, return_tensors="pt")
        else:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens + max_tokens > self.context_size:
            raise ValueError(
                f"Image/question uses {prompt_tokens} tokens plus {max_tokens} output tokens, "
                f"exceeding context-size {self.context_size}. Increase --context-size."
            )
        # Cast only floating tensors (pixel values), preserving integer token IDs.
        inputs = {key: (value.to(device=self.model.device, dtype=self.model.dtype)
                  if value.is_floating_point() else value.to(self.model.device))
                  if self.torch.is_tensor(value) else value
                  for key, value in inputs.items()}
        self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                          do_sample=False, num_beams=1, use_cache=True,
                                          return_dict_in_generate=False)
        generated = outputs[0, prompt_tokens:]
        raw = self.processor.decode(generated, skip_special_tokens=True)
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("The model returned no answer text. Check max-tokens and model settings.")
        # Reaching the token budget with EOS is still a completed response.
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        ended = len(generated) > 0 and generated[-1].item() in eos
        return {
            "raw_response": raw, "prediction": normalize_llm_answer(raw),
            "finish_reason": "length" if len(generated) >= max_tokens and not ended else "stop",
            "latency_seconds": time.perf_counter() - start,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(generated),
                      "total_tokens": prompt_tokens + len(generated)},
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--backend", choices=("auto", "gguf", "transformers"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--gpu-memory-gib", type=float, default=12,
                        help="Transformers weight budget per GPU; leaves memory for images and KV cache")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow a Transformers repository's custom Python code (default: disabled)")
    parser.add_argument("--attention-implementation", choices=("auto", "sdpa", "eager", "flash_attention_2"),
                        default="auto", help="Transformers attention backend; auto uses SDPA, including the Keye T4 adaptation")
    extra, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
    if extra.gpu_memory_gib <= 0:
        parser.error("gpu-memory-gib must be positive")
    options, args = gguf.parse_args(remaining)
    return extra, options, args, remaining


def main(argv=None) -> int:
    extra, options, args, remaining = parse_args(argv)
    try:
        # Validate data before inspecting repositories or downloading weights.
        examples = load_examples(args.questions)
        for example in examples[:args.limit]:
            image_path(args.images, example, args.image_split or args.questions.stem)
        source = inspect_model(options.model_dir, options.hf_revision, extra.backend,
                               options.quantization)
        print(f"Selected backend: {source['backend']}", flush=True)
        if source["backend"] == "gguf":
            if "revision" in source:
                remaining = [*remaining, "--hf-revision", source["revision"]]
            return gguf.main(remaining)
        if options.quantization:
            raise ValueError("--quantization selects GGUF weights; omit it for Transformers.")
        environment, devices = gguf.gpu_environment()
        # Torch is imported only after visibility is restricted to the selected two GPUs.
        os.environ["CUDA_VISIBLE_DEVICES"] = environment["CUDA_VISIBLE_DEVICES"]
        print("NVIDIA GPUs: " + "; ".join(devices), flush=True)
        config = read_transformers_config(source, options.cache_dir)
        plan = transformers_loading_plan(config, extra.trust_remote_code, extra.attention_implementation)
        if plan["attention"] == "flash_attention_2":
            import torch
            validate_attention_hardware(torch, plan["attention"])
        directory, identity = resolve_transformers_model(source, options.cache_dir, extra.trust_remote_code)
        client = TransformersClient(directory, args.model, extra.dtype, extra.gpu_memory_gib,
                                    options.context_size, extra.trust_remote_code, extra.attention_implementation)
        args.backend_config = {**client.backend_config, "model_source": identity,
                               "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"]}
        if args.list_models:
            print("\n".join(client.models()))
            return 0
        report = evaluate(args, client)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Results: {args.output_dir}", flush=True)
        return 0 if report["complete"] else 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun with identical settings and --resume.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, ImportError, AssertionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
