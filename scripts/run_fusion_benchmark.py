#!/usr/bin/env python3
"""Execute fusion benchmark with a single BERT & ViT load for all 15 configurations."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from configs.config import Config
from model.features_extraction import ImageEmbedding, QuestionEmbedding
from utils.data_processing import load_dataframe
from utils.device import resolve_device
from utils.feature_cache import file_fingerprint, write_feature_cache
from utils.vqa_dataset import VQADataset, resolve_image_root


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def fusion_name(method: str, transport: str) -> str:
    if transport == "none":
        return method
    if transport == "uot":
        return f"uot_{method}"
    if transport == "balanced":
        return f"balanced_ot_{method}"
    raise ValueError(f"Unsupported transport: {transport}")


def validate_method(method: str) -> None:
    valid = {"san", "ban", "mutan", "cross_attention", "qformer"}
    if method not in valid:
        raise ValueError(f"Unsupported method: {method}; expected one of {sorted(valid)}")


def validate_transport(transport: str) -> None:
    valid = {"none", "balanced", "uot"}
    if transport not in valid:
        raise ValueError(f"Unsupported transport: {transport}; expected one of {sorted(valid)}")


def precompute_split(csv_path: str | Path, img_path: str | Path, output_dir: str | Path,
                     image_encoder: ImageEmbedding, question_encoder: QuestionEmbedding,
                     device: torch.device, batch_size: int = 4, split_name: str = "train") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = load_dataframe(csv_path)
    resolved_img_root = resolve_image_root(frame, img_path, split_name)
    dataset = VQADataset(frame, Config.transforms, resolved_img_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    samples = []
    with torch.no_grad():
        for anno_ids, images, questions, answers in loader:
            image_features, _ = image_encoder(images.to(device), anno_ids)
            question_features, question_mask, question_ids = question_encoder.encode_tokens(questions)
            valid_lengths = question_ids.ne(question_encoder.tokenizer.pad_token_id).sum(1)
            for row in range(len(questions)):
                length = int(valid_lengths[row])
                samples.append({
                    "anno_id": anno_ids[row].item() if torch.is_tensor(anno_ids) else anno_ids[row],
                    "image_features": image_features[row].detach().cpu().half(),
                    "question_features": question_features[row, :length].detach().cpu().half(),
                    "question_padding_mask": question_mask[row, :length].detach().cpu(),
                    "question": questions[row],
                    "answer": answers[row],
                })

    manifest = {
        "dataset_path": str(Path(csv_path).resolve()),
        "dataset_fingerprint": file_fingerprint(csv_path),
        "text_model": question_encoder.tokenizer.name_or_path,
        "image_model": image_encoder.process.name_or_path if hasattr(image_encoder.process, "name_or_path") else Config.image_model,
        "text_revision": getattr(question_encoder.text_encoder.config, "_commit_hash", None),
        "image_revision": getattr(image_encoder.model.config, "_commit_hash", None),
        "question_max_length": Config.MAX_LEN_QUES,
        "tokenizer": {
            "class": type(question_encoder.tokenizer).__name__,
            "vocab_size": len(question_encoder.tokenizer),
            "padding_side": question_encoder.tokenizer.padding_side,
            "truncation_side": question_encoder.tokenizer.truncation_side,
        },
        "image_processor": {
            "class": type(image_encoder.process).__name__,
            "size": _json_value(image_encoder.process.size),
            "crop_size": _json_value(getattr(image_encoder.process, "crop_size", None)),
            "image_mean": _json_value(image_encoder.process.image_mean),
            "image_std": _json_value(image_encoder.process.image_std),
        },
        "special_token_policy": (
            "question boundary tokens masked; visual CLS/distillation prefix tokens "
            "removed in fusion"
        ),
        "visual_source_shape": list(samples[0]["image_features"].shape) if samples else None,
        "question_hidden_size": samples[0]["question_features"].shape[-1] if samples else None,
        "annotation_ids": [str(sample["anno_id"]) for sample in samples],
    }
    write_feature_cache(output_path, samples, manifest)
    print(f"  [Cache] Cached {len(samples)} samples to {output_path}")


def ensure_feature_cache(cache_dir: Path, train_csv: str | Path, dev_csv: str | Path,
                         img_path: str | Path, text_model: str, image_model: str,
                         device: torch.device, batch_size: int = 4) -> Path:
    train_cache = cache_dir / "train"
    dev_cache = cache_dir / "dev"
    embeddings_file = cache_dir / "embeddings.pt"

    train_valid = (train_cache / "manifest.json").is_file() and (train_cache / "features.pt").is_file()
    dev_valid = (dev_cache / "manifest.json").is_file() and (dev_cache / "features.pt").is_file()

    if train_valid and dev_valid and embeddings_file.is_file():
        print("=====================================================================")
        print(f"[Benchmark] Using existing feature cache in {cache_dir}.")
        print("[Benchmark] BOTH ViT and BERT backbones will NOT be loaded (0 ViT / 0 BERT weights loaded).")
        print("[Benchmark] All benchmark runs will consume cached features directly.")
        print("=====================================================================")
        return cache_dir

    print("=====================================================================")
    print(f"[Benchmark] Loading BOTH ViT ({image_model}) and BERT ({text_model}) ONCE...")
    print("[Benchmark] Precomputing frozen visual and linguistic features for benchmark...")
    print("=====================================================================")

    image_encoder = ImageEmbedding(image_model).to(device).eval()
    question_encoder = QuestionEmbedding(model_name=text_model).to(device).eval()

    cache_dir.mkdir(parents=True, exist_ok=True)
    precompute_split(train_csv, img_path, train_cache, image_encoder, question_encoder, device, batch_size, "train")
    precompute_split(dev_csv, img_path, dev_cache, image_encoder, question_encoder, device, batch_size, "val")

    # Save token embeddings state dict so train.py models initialize without Hugging Face AutoModel
    torch.save(question_encoder.text_encoder.embeddings.state_dict(), embeddings_file)
    torch.save(question_encoder.text_encoder.embeddings.state_dict(), train_cache / "embeddings.pt")
    torch.save(question_encoder.text_encoder.embeddings.state_dict(), dev_cache / "embeddings.pt")

    # Free memory
    del image_encoder
    del question_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    print("=====================================================================")
    print(f"[Benchmark] Feature precomputation complete! Saved to {cache_dir}")
    print("[Benchmark] Heavy ViT & BERT backbones released from memory.")
    print("[Benchmark] ZERO ViT and ZERO BERT reloads across all benchmark modes.")
    print("=====================================================================")
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("DEVICE", "mps"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "50")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "2")))
    parser.add_argument("--methods", nargs="+",
                        default=os.getenv("METHODS", "san ban mutan cross_attention qformer").split())
    parser.add_argument("--transports", nargs="+",
                        default=os.getenv("TRANSPORTS", "none balanced uot").split())
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[int(s) for s in os.getenv("SEEDS", "1105 1106 1107").split()])
    parser.add_argument("--train_csv", default=os.getenv("TRAIN_CSV", "data/gqa_dataset/train.csv"))
    parser.add_argument("--dev_csv", default=os.getenv("DEV_CSV", "data/gqa_dataset/val.csv"))
    parser.add_argument("--img_path", default=os.getenv("IMG_PATH", "data/gqa_dataset/images"))
    parser.add_argument("--text_model", default=os.getenv("TEXT_MODEL", "bert-base-uncased"))
    parser.add_argument("--image_model", default=os.getenv("IMAGE_MODEL", "google/vit-base-patch16-224-in21k"))
    parser.add_argument("--ot_profile", default=os.getenv("OT_PROFILE", None))
    parser.add_argument("--run_id", default=os.getenv("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--run_root", default=os.getenv("RUN_ROOT", None))
    parser.add_argument("--feature_cache", default=os.getenv("FEATURE_CACHE", "data/gqa_cache"))
    parser.add_argument("--diagnostics", type=int, default=int(os.getenv("DIAGNOSTICS", "1")))
    parser.add_argument("--preflight_only", type=int, default=int(os.getenv("PREFLIGHT_ONLY", "0")))
    parser.add_argument("--overwrite", action="store_true",
                        default=bool(int(os.getenv("OVERWRITE", "0"))),
                        help="Allow overwriting an existing non-empty RUN_ROOT directory")

    # Model architecture options matching small-data regularized profile
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--ffn_hidden", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=8)

    args = parser.parse_args()

    # Preflight validation
    for method in args.methods:
        validate_method(method)
    for transport in args.transports:
        validate_transport(transport)

    if args.preflight_only:
        print("Fusion benchmark preflight passed.")
        return

    # Resolve default OT profile
    device_name = args.device.lower()
    if args.ot_profile is None:
        if device_name == "cpu":
            ot_profile = "configs/ot_cpu.json"
        elif device_name == "mps":
            ot_profile = "configs/ot_mps.json"
        elif device_name == "cuda":
            ot_profile = "configs/ot_gpu.json"
        else:
            ot_profile = "configs/ot_cpu.json"
    else:
        ot_profile = args.ot_profile

    run_root = Path(args.run_root if args.run_root else f"results/fusion_benchmark/{args.run_id}")
    if run_root.exists() and any(run_root.iterdir()):
        if args.overwrite:
            import shutil
            print(f"[Benchmark] OVERWRITE active: cleaning existing directory {run_root}")
            shutil.rmtree(run_root)
        else:
            raise RuntimeError(
                f"Run directory already exists and is non-empty: {run_root}\n"
                f"To overwrite this directory, set OVERWRITE=1 or pass --overwrite:\n"
                f"  OVERWRITE=1 RUN_ROOT={run_root} bash scripts/run_fusion_benchmark.sh\n"
                f"Or remove it first:\n"
                f"  rm -rf {run_root}\n"
                f"Or choose a new RUN_ROOT or leave RUN_ROOT unset to auto-generate a timestamped directory."
            )
    run_root.mkdir(parents=True, exist_ok=True)

    # Write benchmark.env
    env_content = (
        f"run_id={args.run_id}\n"
        f"device={args.device}\n"
        f"epochs={args.epochs}\n"
        f"batch_size={args.batch_size}\n"
        f"methods={' '.join(args.methods)}\n"
        f"transports={' '.join(args.transports)}\n"
        f"seeds={' '.join(str(s) for s in args.seeds)}\n"
        f"ot_profile={ot_profile}\n"
        f"text_model={args.text_model}\n"
        f"image_model={args.image_model}\n"
    )
    (run_root / "benchmark.env").write_text(env_content, encoding="utf-8")

    # Ensure feature cache: load BERT & ViT ONCE here
    device = resolve_device(args.device)
    cache_path = Path(args.feature_cache)
    ensure_feature_cache(
        cache_path, args.train_csv, args.dev_csv, args.img_path,
        args.text_model, args.image_model, device,
    )

    # Benchmark loop: All modes run using feature cache with ZERO BERT loads
    python_bin = os.getenv("PYTHON_BIN", sys.executable)
    total_runs = len(args.methods) * len(args.transports) * len(args.seeds)
    current_run = 0

    print(f"\n[Benchmark] Launching {total_runs} runs across {len(args.methods)} methods and {len(args.transports)} transport configurations...")

    for method in args.methods:
        for transport in args.transports:
            fusion = fusion_name(method, transport)
            for seed in args.seeds:
                current_run += 1
                run_dir = run_root / method / transport / f"seed_{seed}"
                model_dir = run_dir / "model"
                run_dir.mkdir(parents=True, exist_ok=True)

                command = [
                    python_bin, "train.py",
                    "--device", args.device,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--fusion", fusion,
                    "--train_csv_path", str(args.train_csv),
                    "--dev_csv_path", str(args.dev_csv),
                    "--img_path", str(args.img_path),
                    "--text_model", str(args.text_model),
                    "--image_model", str(args.image_model),
                    "--feature_cache", str(cache_path),
                    "--d_model", str(args.d_model),
                    "--ffn_hidden", str(args.ffn_hidden),
                    "--num_layers", str(args.num_layers),
                    "--num_heads", str(args.num_heads),
                    "--drop_prob", str(args.drop_prob),
                    "--freeze_answer_embeddings",
                    "--weight_decay", str(args.weight_decay),
                    "--gradient_clip", str(args.gradient_clip),
                    "--label_smoothing", str(args.label_smoothing),
                    "--early_stopping_patience", str(args.early_stopping_patience),
                    "--seed", str(seed),
                    "--model_path", str(model_dir),
                ]

                if transport != "none":
                    command.extend(["--ot_profile", ot_profile])
                if fusion.endswith("_san") and fusion != "san":
                    command.extend([
                        "--ot_san_layers", "1",
                        "--ot_san_hidden_dim", "128",
                        "--ot_san_dropout", "0.2",
                        "--ot_san_gate_init", "-2.0",
                    ])
                if args.diagnostics:
                    command.append("--diagnostics")

                # Save command.txt
                (run_dir / "command.txt").write_text(" ".join(shlex.quote(c) for c in command) + "\n", encoding="utf-8")

                print(f"\n[{current_run}/{total_runs}] Starting method={method} transport={transport} seed={seed}")

                # Stream to both stdout and train.log
                log_file = run_dir / "train.log"
                with log_file.open("w", encoding="utf-8") as handle:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        handle.write(line)
                        handle.flush()
                    process.wait()
                    if process.returncode != 0:
                        raise RuntimeError(f"Run failed with exit code {process.returncode}: {' '.join(command)}")

    # Summarize runs
    print(f"\n[Benchmark] Aggregating run metrics into {run_root}...")
    summarize_cmd = [python_bin, "scripts/summarize_fusion_benchmark.py", str(run_root)]
    subprocess.run(summarize_cmd, check=True)
    print(f"\nBenchmark complete: {run_root}")


if __name__ == "__main__":
    main()
