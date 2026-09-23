#!/usr/bin/env python3
"""Run the matched OT evidence-routing pilot described in the research plan."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--dev_csv", required=True)
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--output_root", default="results/ot_evidence_routing")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--routing_query_diversity_weight", type=float, default=0.0)
    parser.add_argument("--counterfactual_weight", type=float, default=0.0)
    parser.add_argument("--effective_batch_size", type=int, default=32)
    parser.add_argument("--routing_tau", type=float, default=0.1)
    parser.add_argument("--routing_tau_warmup_epochs", type=int, default=5)
    parser.add_argument("--routing_tau_ramp_epochs", type=int, default=5)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="cuda")
    parser.add_argument("--gpus", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--mixed_precision", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=[
            "cross_attention", "softmax_evidence_routing",
            "ot_evidence_routing", "ot_tau_zero",
            "softmax_evidence_routing_v2", "ot_evidence_routing_v2",
        ],
        default=[
            "cross_attention", "softmax_evidence_routing_v2",
            "ot_evidence_routing_v2",
        ],
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def command(args, seed, model):
    prefix = [sys.executable]
    if args.gpus == 2:
        prefix = ["torchrun", "--standalone", "--nproc_per_node=2"]
    destination = Path(args.output_root) / model / f"seed_{seed}"
    fusion = "ot_evidence_routing" if model == "ot_tau_zero" else model
    accumulation = max(
        1, math.ceil(args.effective_batch_size / (args.batch_size * args.gpus))
    )
    result = prefix + [
        "train.py",
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed),
        "--fusion", fusion,
        "--alignment_mode", "none",
        "--train_csv_path", args.train_csv,
        "--dev_csv_path", args.dev_csv,
        "--img_path", args.img_path,
        "--model_path", str(destination),
        "--diagnostics",
        "--gradient_accumulation_steps", str(accumulation),
    ]
    if args.feature_cache:
        result.extend(["--feature_cache", args.feature_cache])
    if args.lr is not None:
        result.extend(["--lr", str(args.lr)])
    if args.mixed_precision:
        result.append("--mixed_precision")
    if fusion != "cross_attention":
        result.extend([
            "--routing_query_diversity_weight",
            str(args.routing_query_diversity_weight),
        ])
    if fusion.endswith("_v2"):
        result.extend([
            "--routing_tau", str(args.routing_tau),
            "--routing_tau_warmup_epochs", str(args.routing_tau_warmup_epochs),
            "--routing_tau_ramp_epochs", str(args.routing_tau_ramp_epochs),
            "--fusion_lr", "3e-4",
            "--decoder_lr", "1e-4",
            "--warmup_ratio", "0.05",
            "--lr_schedule", "cosine",
            "--weight_decay", "0.01",
            "--counterfactual_weight", str(args.counterfactual_weight),
        ])
    if model == "ot_tau_zero":
        result.extend(["--routing_tau", "0"])
    return result


def main():
    args = parse_args()
    if args.gpus == 2 and args.device not in {"cuda", "auto"}:
        raise ValueError("Two-rank execution requires CUDA or auto device selection")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    for seed in args.seeds:
        for model in args.models:
            current = command(args, seed, model)
            print(" ".join(shlex.quote(part) for part in current), flush=True)
            if not args.dry_run:
                subprocess.run(current, check=True, env=environment)


if __name__ == "__main__":
    main()
