"""Zero-shot VQA evaluation with native Hugging Face vision-language checkpoints.

Run from the repository root: python3 -m scripts.evaluate_llm --help
No GGUF weights, local server, or model conversion is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from utils.metrics import (PAPER_METRICS, build_bertscore_scorer, lexical_scores,
                           normalize_llm_answer, score_pairs)

MODEL_PRESETS = {
    "llava-onevision-1.5-8b": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B-HF",
    "ministral-3-8b": "mistralai/Ministral-3-8B-Instruct-2512",
    "glm-4.6v-flash": "zai-org/GLM-4.6V-Flash",
    "gemma-4-e4b": "google/gemma-4-E4B-it",
}
MODEL_CLASSES = {
    "llavaonevision1_5": "AutoModelForCausalLM",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
    "internvl": "InternVLForConditionalGeneration",
    "mistral3": "Mistral3ForConditionalGeneration",
    "glm4v": "Glm4vForConditionalGeneration",
    "gemma4": "Gemma4ForConditionalGeneration",
}
ONEVISION_MODEL_TYPE = "llavaonevision1_5"
ONEVISION_AUTO_MAP = {
    "AutoConfig": "configuration_llavaonevision1_5.Llavaonevision1_5Config",
    "AutoModelForCausalLM": "modeling_llavaonevision1_5.LLaVAOneVision1_5_ForConditionalGeneration",
}


def normalize_model_config(config: dict) -> dict:
    """Identify OneVision 1.5 even when its published config omits model_type."""
    if not isinstance(config, dict):
        raise ValueError("Model config.json must contain a JSON object.")
    config = config.copy()
    auto_map = config.get("auto_map") or {}
    onevision = all(auto_map.get(key) == value for key, value in ONEVISION_AUTO_MAP.items())
    if onevision and not config.get("model_type"):
        config["model_type"] = ONEVISION_MODEL_TYPE
    if config.get("model_type") == ONEVISION_MODEL_TYPE and not onevision:
        raise ValueError("OneVision 1.5 requires the checkpoint's original AutoConfig/AutoModelForCausalLM mappings.")
    if config.get("model_type") not in MODEL_CLASSES:
        raise ValueError(f"Unsupported vision architecture: {config.get('model_type')!r}; "
                         f"architectures={config.get('architectures', [])}. Use --list-models. "
                         "InternVL3.5 requires OpenGVLab/InternVL3_5-8B-HF.")
    return config


def runtime_requirements(config: dict) -> list[str]:
    """OneVision's custom implementation needs the Transformers 4.x APIs."""
    if config["model_type"] == ONEVISION_MODEL_TYPE:
        return ["transformers==4.57.1", "huggingface-hub>=0.34,<1"]
    return ["transformers>=5.5,<6", "huggingface-hub>=1,<2"]

SYSTEM_PROMPT = (
    "Answer the question using the image. Return only the short answer, "
    "usually one word or a short phrase. For yes/no questions, return yes or no. "
    "Do not include an explanation, a full sentence, or an 'Answer:' prefix."
)
PLANTEXPERT_SYSTEM_PROMPT = (
    "Answer the plant-science question using the image. Give a concise, complete "
    "answer with the details requested by the question. For yes/no questions, "
    "answer yes or no and explain only if requested. Do not include an 'Answer:' "
    "prefix, unrelated information, or discussion of your reasoning process."
)
METADATA_FIELDS = ("question_type", "question_category", "crop", "disease")
ANSWER_NORMALIZATION = "whitespace_markdown_v1"
NORMALIZATION_DESCRIPTION = (
    "Predictions and references have actual/literal escaped newlines and tabs, "
    "repeated whitespace, and Markdown emphasis, code fences, headings, and bullets "
    "normalized before scoring. Meaningful punctuation and case are preserved."
)


@dataclass(frozen=True)
class Example:
    question_id: str
    image: str
    question: str
    answer: str | None
    metadata: dict[str, str] = field(default_factory=dict)


def load_examples(path: Path) -> list[Example]:
    """Accept repository CSVs or official GQA questionId -> question JSON."""
    examples = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"image", "question"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV must contain image and question columns; answer is optional.")
            for index, row in enumerate(reader):
                examples.append(Example(
                    row.get("question_id") or row.get("anno_id") or str(index),
                    row["image"], row["question"], row.get("answer"),
                    {key: row[key] for key in METADATA_FIELDS if key in row},
                ))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("GQA JSON must map question IDs to question objects.")
        for question_id, row in data.items():
            examples.append(Example(
                str(question_id), f"{row['imageId']}.jpg", row["question"],
                row.get("answer") or None,
            ))
    else:
        raise ValueError("Questions must be a .csv or .json file.")
    if not examples:
        raise ValueError("The question file is empty.")
    ids = set()
    for example in examples:
        if example.question_id in ids:
            raise ValueError(f"Duplicate question ID: {example.question_id}")
        ids.add(example.question_id)
        if not example.image or not example.question.strip():
            raise ValueError(f"Missing image or question for {example.question_id}")
    return examples


def image_path(root: Path, example: Example, split: str) -> Path:
    """Resolve image.jpg, split/image.jpg, and repository split-folder layouts."""
    direct = root / example.image
    if direct.is_file():
        return direct
    nested = root / split / example.image
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"Image not found: {direct} (also checked {nested})")


def normalize(text: str) -> str:
    return normalize_llm_answer(text).casefold()


def make_record(example: Example, result: dict | None = None, error: str = "",
                dataset: str = "gqa") -> dict:
    record = {
        "question_id": example.question_id, "image": example.image,
        "question": example.question, "reference": example.answer,
        "prediction": "", "raw_response": "", "error": error,
    }
    record.update(example.metadata)
    record.update(result or {})
    record["prediction"] = normalize_llm_answer(record["prediction"])
    labeled = example.answer is not None
    reference = normalize_llm_answer(example.answer) if labeled else None
    if dataset == "plantexpert":
        if not labeled:
            scores = {metric: None for metric in PAPER_METRICS[:-1]}
        elif error:
            scores = {metric: 0.0 for metric in PAPER_METRICS[:-1]}
        else:
            scores = lexical_scores(reference, record["prediction"])
        record.update(scores)
        return record
    record["correct"] = (
        bool(not error and record["prediction"] == reference) if labeled else None
    )
    record["normalized_correct"] = (
        bool(not error and normalize(record["prediction"]) == normalize(example.answer))
        if labeled else None
    )
    return record


def save_outputs(output: Path, records: dict[str, dict], config: dict,
                 selected_count: int, bert_scorer=None) -> dict:
    rows = list(records.values())
    labeled = [row for row in rows if row["reference"] is not None]
    successful = [row for row in rows if not row["error"]]
    report = {
        "config": config,
        "examples_selected": selected_count,
        "examples_attempted": len(rows),
        "examples_succeeded": len(successful),
        "errors": len(rows) - len(successful),
        "labeled_examples_attempted": len(labeled),
        "complete": len(successful) == selected_count,
        "truncated_responses": sum(row.get("finish_reason") == "length" for row in successful),
        "mean_latency_seconds": (
            sum(row["latency_seconds"] for row in successful) / len(successful)
            if successful else None
        ),
    }
    plantexpert = config.get("dataset") == "plantexpert"
    if plantexpert:
        report["generated_metrics"] = {
            metric: sum(row[metric] for row in labeled) / len(labeled) if labeled else None
            for metric in PAPER_METRICS[:-1]
        }
        report["generated_metrics"]["bertscore_f1"] = None
        report["bertscore"] = config["bertscore"].copy()
        report["scoring"] = {
            "metrics": "Repository PlantExpert metrics; scores are fractions, rescaled BERTScore can be negative.",
            "errors": "Failed labeled requests receive zero for every metric; unattempted questions are excluded.",
        }
    else:
        report.update({
            "accuracy": sum(row["correct"] for row in labeled) / len(labeled) if labeled else None,
            "normalized_accuracy": (
                sum(row["normalized_correct"] for row in labeled) / len(labeled) if labeled else None
            ),
            "scoring": {
                "accuracy": "Exact match after answer formatting normalization; one point per correct answer.",
                "normalized_accuracy": "Exact match after answer formatting normalization, also ignoring case.",
                "errors": "Failed labeled requests count as incorrect; unattempted questions are excluded.",
            },
        })
    report["scoring"]["answer_normalization"] = NORMALIZATION_DESCRIPTION
    # Save a report before semantic scoring, so scoring failures preserve inference results.
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if plantexpert:
        for row in rows:
            failed_label = row["error"] and row["reference"] is not None
            row["bertscore_f1"] = 0.0 if bert_scorer is not None and failed_label else None
        if bert_scorer is not None:
            scored = [row for row in labeled if not row["error"]]
            if scored:
                scores = score_pairs([normalize_llm_answer(row["reference"]) for row in scored],
                                     [row["prediction"] for row in scored], bert_scorer)
                for row, metrics in zip(scored, scores):
                    row["bertscore_f1"] = metrics["bertscore_f1"]
            report["generated_metrics"]["bertscore_f1"] = (
                sum(row["bertscore_f1"] for row in labeled) / len(labeled) if labeled else None
            )
            report["bertscore"]["model_hash"] = bert_scorer.hash
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    return report


def evaluate(args: argparse.Namespace, client: TransformersClient) -> dict:
    examples = load_examples(args.questions)
    if args.limit is not None:
        examples = examples[:args.limit]
    # Validate the selected files before starting a potentially long inference run.
    paths = {example.question_id: image_path(args.images, example, args.image_split or args.questions.stem)
             for example in examples}
    plantexpert = args.dataset == "plantexpert"
    system_prompt = PLANTEXPERT_SYSTEM_PROMPT if plantexpert else SYSTEM_PROMPT
    config = {
        "questions": str(args.questions.resolve()),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "images": str(args.images.resolve()),
        "image_split": args.image_split or args.questions.stem,
        "model": args.model, "base_url": client.base_url,
        "limit": args.limit, "seed": args.seed,
        "temperature": 0, "max_tokens": args.max_tokens,
        "system_prompt": system_prompt, "protocol": "zero-shot",
        "answer_normalization": ANSWER_NORMALIZATION,
    }
    if getattr(args, "backend_config", None) is not None:
        config["backend"] = args.backend_config
    if plantexpert:
        config.update({"dataset": "plantexpert", "bertscore": {
            "enabled": not args.no_bertscore, "model": args.bertscore_model,
            "device": args.bertscore_device, "batch_size": args.bertscore_batch_size,
            "rescale_with_baseline": args.bertscore_rescale,
        }})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_config.json"
    journal = output / "predictions.jsonl"
    records = {}
    if args.resume:
        saved_config = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
        manifest_config = saved_config.copy() if saved_config is not None else None
        # Older journals retain raw predictions; rescore them with current cleanup.
        if saved_config is not None:
            saved_config.setdefault("answer_normalization", ANSWER_NORMALIZATION)
        if saved_config != config:
            raise ValueError("Resume configuration differs or is missing. Use the original flags or a new output-dir.")
        if journal.is_file():
            examples_by_id = {example.question_id: example for example in examples}
            for line in journal.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                records[row["question_id"]] = make_record(
                    examples_by_id[row["question_id"]], row, error=row["error"], dataset=args.dataset,
                )
    else:
        if any(output.iterdir()):
            raise ValueError(f"Output directory is not empty: {output}. Use --resume or a new output-dir.")
    bert_scorer = None
    if plantexpert and not args.no_bertscore and any(example.answer is not None for example in examples):
        print("Loading BERTScore before inference...", flush=True)
        bert_scorer = build_bertscore_scorer(
            model_type=args.bertscore_model, device=args.bertscore_device,
            batch_size=args.bertscore_batch_size, rescale_with_baseline=args.bertscore_rescale,
        )
    if not args.resume:
        manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    elif "answer_normalization" not in manifest_config:
        manifest_config["answer_normalization"] = ANSWER_NORMALIZATION
        manifest.write_text(json.dumps(manifest_config, indent=2) + "\n", encoding="utf-8")
    print(f"Model: {args.model}; selected questions: {len(examples)}; zero-shot", flush=True)
    try:
        with journal.open("a", encoding="utf-8") as handle:
            for index, example in enumerate(examples, 1):
                previous = records.get(example.question_id)
                if previous is not None and not previous["error"]:
                    continue
                failure = None
                try:
                    prompt_args = {"system_prompt": system_prompt} if plantexpert else {}
                    result = client.answer(args.model, example, paths[example.question_id],
                                           args.max_tokens, args.seed, **prompt_args)
                    record = make_record(example, result, dataset=args.dataset)
                except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
                    failure = error
                    record = make_record(example, error=str(error), dataset=args.dataset)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records[example.question_id] = record
                print(f"[{index}/{len(examples)}] {example.question_id}: "
                      f"{record['prediction']!r} | reference={example.answer!r}"
                      + (f" | ERROR: {failure}" if failure else ""), flush=True)
                if failure and not args.continue_on_error:
                    raise RuntimeError("Stopped after an error; predictions are saved. Fix it and use --resume.") from failure
    finally:
        report = save_outputs(output, records, config, len(examples), bert_scorer)
    return report



def inspect_model(model: str, revision: str = "main", cache_dir: Path | None = None) -> tuple[dict, dict]:
    """Read config/metadata first and pin remote weights and processor together."""
    model = MODEL_PRESETS.get(model.lower(), model)
    local = Path(model).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError("--model must be a native checkpoint folder or Hugging Face repository, not a GGUF file.")
        names = [path.relative_to(local).as_posix() for path in local.rglob("*") if path.is_file()]
        source = {"directory": str(local.resolve()), "names": names}
        config_path = local / "config.json"
    else:
        if local.is_absolute() or model.startswith((".", "~")):
            raise FileNotFoundError(f"Local model folder does not exist: {local}")
        if model.startswith("https://huggingface.co/"):
            model = urlparse(model).path.strip("/")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model) or "gguf" in model.lower():
            raise ValueError("Use a native Hugging Face owner/model repository; GGUF and quantization suffixes are unsupported.")
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        commit = api.model_info(model, revision=revision).sha
        if not commit:
            raise ValueError("Cannot resolve the model revision to a commit.")
        names = api.list_repo_files(model, revision=commit)
        source = {"repo": model, "revision": commit, "names": names}
        # Reject GGUF-only repositories before downloading anything.
        validate_model_files(names)
        config_path = Path(hf_hub_download(repo_id=model, filename="config.json", revision=commit,
                                          cache_dir=str(cache_dir) if cache_dir else None))
    validate_model_files(names)
    config = normalize_model_config(json.loads(config_path.read_text(encoding="utf-8")))
    return source, config


def validate_model_files(names: list[str]):
    if "config.json" not in names or not any(
        name in names for name in ("model.safetensors", "model.safetensors.index.json",
                                   "pytorch_model.bin", "pytorch_model.bin.index.json")
    ):
        raise ValueError("Expected config.json and native Transformers weights (Safetensors or pytorch_model.bin). GGUF is unsupported.")


def resolve_model(source: dict, cache_dir: Path | None) -> tuple[Path, dict]:
    """Download one Transformers weight set, excluding GGUF and duplicate Mistral consolidated weights."""
    names = source["names"]
    safe = "model.safetensors" in names or "model.safetensors.index.json" in names
    weight_suffix = ".safetensors" if safe else ".bin"
    assets = [name for name in names if name.endswith((".json", ".model", ".txt", ".jinja", ".tiktoken", ".py"))
              or name.startswith("model" if safe else "pytorch_model") and name.endswith(weight_suffix)]
    if "directory" in source:
        directory = Path(source["directory"])
        checksums = {}
        for name in sorted(assets):
            digest = hashlib.sha256()
            with (directory / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[name] = digest.hexdigest()
        identity = {"directory": str(directory), "files_sha256": checksums}
    else:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(repo_id=source["repo"], revision=source["revision"],
                                          cache_dir=str(cache_dir) if cache_dir else None,
                                          allow_patterns=assets))
        identity = {"repo": source["repo"], "revision": source["revision"]}
    return directory, identity


def validate_runtime(config: dict, dtype: str):
    try:
        import torch
        import accelerate
        import transformers
    except ImportError as error:
        raise RuntimeError("Install evaluation dependencies: " + " ".join(runtime_requirements(config))
                           + " accelerate pillow safetensors") from error
    from packaging.specifiers import SpecifierSet
    requirement = runtime_requirements(config)[0]
    if transformers.__version__ not in SpecifierSet(requirement.removeprefix("transformers")):
        raise RuntimeError(f"{config['model_type']} needs {requirement}; installed {transformers.__version__}. "
                           "Run: pip install -U " + " ".join(f"'{item}'" for item in runtime_requirements(config)))
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError("An NVIDIA CUDA GPU is required. On Kaggle select GPU T4 x2.")
    bf16 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(count))
    if dtype == "auto":
        dtype = "bfloat16" if bf16 else "float16"
    if dtype == "bfloat16" and not bf16:
        raise ValueError("bfloat16 requires Ampere or newer GPUs; select float16 for T4.")
    model_class = MODEL_CLASSES[config["model_type"]]
    loader = getattr(transformers, model_class, None)
    if loader is None:
        raise RuntimeError(f"Installed Transformers {transformers.__version__} lacks {model_class}; upgrade transformers.")
    loading = {}
    quantization = config.get("quantization_config") or {}
    if quantization:
        if quantization.get("quant_method") != "fp8":
            raise ValueError("Use the original native checkpoint; only its FP8-to-16-bit loading is supported.")
        # Official Ministral weights are FP8. Dequantize during loading for T4.
        loading["quantization_config"] = transformers.FineGrainedFP8Config(dequantize=True)
    return torch, accelerate, transformers, count, dtype, loader, loading


class TransformersClient:
    base_url = "transformers://local"

    def __init__(self, directory: Path, alias: str, config: dict, dtype: str = "auto",
                 gpu_memory_gib: float = 12, context_size: int = 8192, attention: str = "sdpa"):
        torch, accelerate, transformers, count, dtype, loader, loading = validate_runtime(config, dtype)
        model_class = MODEL_CLASSES[config["model_type"]]
        quantization = config.get("quantization_config") or {}
        self.torch = torch
        self.alias = alias
        self.context_size = context_size
        self.onevision = config["model_type"] == ONEVISION_MODEL_TYPE
        # Only this explicitly supported custom architecture executes checkpoint code.
        # resolve_model downloads its Python files at the same pinned commit as the weights.
        self.processor = transformers.AutoProcessor.from_pretrained(
            str(directory), trust_remote_code=self.onevision, local_files_only=True,
            **({"use_fast": False} if self.onevision else {}),
        )
        self.model = loader.from_pretrained(
            str(directory), dtype=getattr(torch, dtype), device_map="balanced",
            max_memory={**{i: int(gpu_memory_gib * 2**30) for i in range(count)}, "cpu": 0},
            attn_implementation=attention, trust_remote_code=self.onevision, local_files_only=True,
            **loading,
        ).eval()
        placement = getattr(self.model, "hf_device_map", {})
        allowed = {str(i) for i in range(count)} | {f"cuda:{i}" for i in range(count)}
        if not placement or any(str(device) not in allowed for device in placement.values()):
            raise RuntimeError("Weights do not fit in the GPU budget. Increase --gpu-memory-gib or use GPUs with more memory.")
        self.backend_config = {
            "engine": "transformers", "version": transformers.__version__,
            "torch_version": torch.__version__, "accelerate_version": accelerate.__version__,
            "model_class": model_class, "dtype": dtype, "attention_implementation": attention,
            "device_map": {name: str(device) for name, device in placement.items()},
            "gpu_memory_gib": gpu_memory_gib, "context_size": context_size,
            "fp8_dequantized": bool(quantization), "enable_thinking": False,
        }
        if self.onevision:
            self.backend_config.update({"trust_remote_code": True,
                                        "model_class": type(self.model).__name__})

    def answer(self, model, example, path, max_tokens, seed, system_prompt=SYSTEM_PROMPT) -> dict:
        if model != self.alias:
            raise ValueError(f"Loaded model is {self.alias}, not {model}.")
        start = time.perf_counter()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "url": str(path.resolve())},
                {"type": "text", "text": example.question},
            ]},
        ]
        if self.onevision:
            # The official checkpoint uses Qwen2.5-VL's processor with temporal_patch_size=1.
            # Feed the PIL image directly so its own resize/normalization settings are used.
            from PIL import Image
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
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
            raise ValueError(f"Image/question uses {prompt_tokens} tokens plus {max_tokens} output tokens, "
                             f"exceeding --context-size {self.context_size}.")
        # Preserve integer token IDs, masks, grids, and other processor metadata.
        inputs = {key: (value.to(device=self.model.device, dtype=self.model.dtype)
                        if value.is_floating_point() else value.to(self.model.device))
                  if self.torch.is_tensor(value) else value for key, value in inputs.items()}
        self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                          num_beams=1, use_cache=True, return_dict_in_generate=False)
        generated = outputs[0, prompt_tokens:]
        raw = self.processor.decode(generated, skip_special_tokens=True)
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("The model returned no answer text. Check max-tokens and model settings.")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-dir", default=MODEL_PRESETS["qwen3-vl-8b"],
                        help="Preset alias, native HF repository ID/URL, or local checkpoint folder")
    parser.add_argument("--list-models", action="store_true", help="Show native model presets without downloading")
    parser.add_argument("--hf-revision", default="main", help="HF revision; resolved to a commit for reproducibility")
    parser.add_argument("--cache-dir", type=Path, help="HF cache directory (default: standard HF cache)")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--gpu-memory-gib", type=float, default=12, help="Weight budget per visible GPU; reserve room for generation")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--context-size", type=int, default=8192, help="Maximum prompt plus output tokens")
    parser.add_argument("--dataset", choices=("gqa", "plantexpert"), default="gqa")
    parser.add_argument("--questions", type=Path, help="Default: data/<dataset>_dataset/test.csv")
    parser.add_argument("--images", type=Path, help="Default: data/<dataset>_dataset/images")
    parser.add_argument("--image-split", help="Optional split subfolder for flat image names")
    parser.add_argument("--output-dir", type=Path, help="Default: results/<model>_<dataset>_zero_shot")
    parser.add_argument("--limit", type=int, help="First N questions (default: all)")
    parser.add_argument("--max-tokens", type=int, help="Default: 64 for GQA, 256 for PlantExpert")
    parser.add_argument("--no-bertscore", action="store_true", help="PlantExpert: skip the semantic metric")
    parser.add_argument("--bertscore-model", default="bert-base-uncased")
    parser.add_argument("--bertscore-device", default="cpu")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--no-bertscore-rescale", dest="bertscore_rescale", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reuse saved successes and retry failed questions")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failures as incorrect and continue")
    args = parser.parse_args(argv)
    args.model = MODEL_PRESETS.get(args.model.lower(), args.model)
    args.questions = args.questions or Path(f"data/{args.dataset}_dataset/test.csv")
    args.images = args.images or Path(f"data/{args.dataset}_dataset/images")
    model_name = Path(args.model.rstrip("/")).name
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_") or "model"
    args.output_dir = args.output_dir or Path(f"results/{slug}_{args.dataset}_zero_shot")
    if args.max_tokens is None:
        args.max_tokens = 256 if args.dataset == "plantexpert" else 64
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if min(args.max_tokens, args.context_size, args.bertscore_batch_size) <= 0:
        parser.error("max-tokens, context-size, and bertscore-batch-size must be positive")
    if not math.isfinite(args.gpu_memory_gib) or args.gpu_memory_gib <= 0:
        parser.error("gpu-memory-gib must be a finite positive number")
    if args.max_tokens >= args.context_size:
        parser.error("max-tokens must leave room for the prompt in context-size")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_models:
        for alias, repo in MODEL_PRESETS.items():
            print(f"{alias}: {repo}")
        return 0
    try:
        # Stop on bad data or an occupied output directory before model downloads.
        examples = load_examples(args.questions)[:args.limit]
        for example in examples:
            image_path(args.images, example, args.image_split or args.questions.stem)
        if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {args.output_dir}. Use --resume or a new output-dir.")
        source, config = inspect_model(args.model, args.hf_revision, args.cache_dir)
        validate_runtime(config, args.dtype)
        directory, identity = resolve_model(source, args.cache_dir)
        client = TransformersClient(directory, args.model, config, args.dtype, args.gpu_memory_gib,
                                    args.context_size, args.attention_implementation)
        args.backend_config = {**client.backend_config, "model_source": identity}
        report = evaluate(args, client)
        if args.dataset == "plantexpert":
            print("PlantExpert metrics: " + ", ".join(
                f"{key}={value:.4f}" if value is not None else f"{key}=not computed"
                for key, value in report["generated_metrics"].items()))
        elif report["accuracy"] is not None:
            print(f"Accuracy: {report['accuracy']:.2%}; normalized: {report['normalized_accuracy']:.2%}")
        else:
            print("No reference answers supplied; predictions saved without scoring.")
        print(f"Results: {args.output_dir / 'report.json'}")
        return 0 if report["complete"] else 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun with identical settings and --resume.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
