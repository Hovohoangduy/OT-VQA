#!/usr/bin/env python3
"""Summarize best validation rows from matched OT evidence-routing runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


MODELS = (
    "cross_attention",
    "softmax_evidence_routing",
    "ot_evidence_routing",
    "ot_tau_zero",
)


def best_row(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "val_f1" in row and "val_loss" in row:
                rows.append(row)
    if not rows:
        raise ValueError(f"No VQA validation rows in {path}")
    return max(rows, key=lambda row: (row["val_f1"], -row["val_loss"]))


def parse_seed(path: Path):
    name = path.parent.name
    if not name.startswith("seed_"):
        raise ValueError(f"Expected seed directory, received {path.parent}")
    return int(name.removeprefix("seed_"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/ot_routing_pilot")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    records = []
    for model in MODELS:
        for path in sorted((root / model).glob("seed_*/metrics.jsonl")):
            row = best_row(path)
            records.append({
                "model": model,
                "seed": parse_seed(path),
                "best_epoch": row["epoch"],
                "best_val_f1": row["val_f1"],
                "best_val_em": row["val_em"],
                "best_val_loss": row["val_loss"],
                "train_f1_at_best": row["train_f1"],
                "train_val_f1_gap": row["train_f1"] - row["val_f1"],
                "latency_ms_per_example": row.get("val_latency_ms_per_example"),
                "routing_null": row.get("val_fusion_routing_null"),
                "routing_converged": row.get("val_fusion_routing_converged"),
            })
    if not records:
        raise ValueError(f"No matched run metrics found below {root}")

    grouped = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)
    summary = {}
    for model, rows in grouped.items():
        f1 = [row["best_val_f1"] for row in rows]
        summary[model] = {
            "runs": len(rows),
            "seeds": [row["seed"] for row in rows],
            "mean_best_val_f1": statistics.fmean(f1),
            "std_best_val_f1": statistics.stdev(f1) if len(f1) > 1 else 0.0,
            "mean_best_val_loss": statistics.fmean(
                row["best_val_loss"] for row in rows
            ),
        }

    by_model_seed = {
        (row["model"], row["seed"]): row for row in records
    }
    paired = []
    for seed in sorted({row["seed"] for row in records}):
        ot = by_model_seed.get(("ot_evidence_routing", seed))
        softmax = by_model_seed.get(("softmax_evidence_routing", seed))
        if ot is not None and softmax is not None:
            paired.append({
                "seed": seed,
                "ot_minus_softmax_f1": ot["best_val_f1"] - softmax["best_val_f1"],
            })
    report = {
        "runs": records,
        "summary": summary,
        "paired_ot_vs_softmax": paired,
        "mean_paired_f1_difference": (
            statistics.fmean(row["ot_minus_softmax_f1"] for row in paired)
            if paired else None
        ),
    }
    output = Path(args.output) if args.output else root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output} and {csv_path}")


if __name__ == "__main__":
    main()
