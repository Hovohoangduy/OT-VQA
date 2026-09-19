"""Summarize fusion benchmark metrics into run-level and aggregate files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


COMPARISONS = (
    ("balanced_vs_none", "none", "balanced"),
    ("uot_vs_none", "none", "uot"),
    ("uot_vs_balanced", "balanced", "uot"),
)


RUN_FIELDS = [
    "method", "transport", "seed", "epochs_completed", "best_epoch",
    "best_train_f1", "best_val_loss", "best_val_em", "best_val_f1",
    "train_val_f1_gap", "final_epoch", "final_train_f1", "final_val_f1",
    "total_parameters", "trainable_parameters", "latency_ms_per_example",
]


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    return rows


def summarize_run(root: Path, metrics_path: Path) -> dict:
    relative = metrics_path.relative_to(root)
    parts = relative.parts
    if len(parts) != 5 or parts[3:] != ("model", "metrics.jsonl"):
        raise ValueError(f"Unexpected benchmark path: {relative}")
    method, transport, seed_folder = parts[:3]
    if not seed_folder.startswith("seed_"):
        raise ValueError(f"Unexpected seed folder: {seed_folder}")
    rows = read_rows(metrics_path)
    best = max(rows, key=lambda row: (row["val_f1"], -row["val_loss"]))
    final = rows[-1]
    return {
        "method": method,
        "transport": transport,
        "seed": int(seed_folder.removeprefix("seed_")),
        "epochs_completed": len(rows),
        "best_epoch": best["epoch"],
        "best_train_f1": best["train_f1"],
        "best_val_loss": best["val_loss"],
        "best_val_em": best["val_em"],
        "best_val_f1": best["val_f1"],
        "train_val_f1_gap": best["train_f1"] - best["val_f1"],
        "final_epoch": final["epoch"],
        "final_train_f1": final["train_f1"],
        "final_val_f1": final["val_f1"],
        "total_parameters": best.get("total_parameters"),
        "trainable_parameters": best.get("trainable_parameters"),
        "latency_ms_per_example": best.get("val_latency_ms_per_example"),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["transport"]), []).append(row)
    result = []
    for (method, transport), group in sorted(groups.items()):
        f1_values = [row["best_val_f1"] for row in group]
        loss_values = [row["best_val_loss"] for row in group]
        gap_values = [row["train_val_f1_gap"] for row in group]
        parameter_values = [
            row["trainable_parameters"] for row in group
            if row["trainable_parameters"] is not None
        ]
        latency_values = [
            row["latency_ms_per_example"] for row in group
            if row["latency_ms_per_example"] is not None
        ]
        result.append({
            "method": method,
            "transport": transport,
            "runs": len(group),
            "mean_best_val_f1": statistics.fmean(f1_values),
            "std_best_val_f1": statistics.stdev(f1_values) if len(group) > 1 else 0.0,
            "mean_best_val_loss": statistics.fmean(loss_values),
            "mean_train_val_f1_gap": statistics.fmean(gap_values),
            "mean_trainable_parameters": (
                statistics.fmean(parameter_values) if parameter_values else None
            ),
            "mean_latency_ms_per_example": (
                statistics.fmean(latency_values) if latency_values else None
            ),
        })
    return result


def paired_comparisons(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    indexed = {
        (row["method"], row["transport"], row["seed"]): row
        for row in rows
    }
    pairs = []
    methods_and_seeds = sorted({(row["method"], row["seed"]) for row in rows})
    for method, seed in methods_and_seeds:
        for comparison, baseline_name, candidate_name in COMPARISONS:
            baseline = indexed.get((method, baseline_name, seed))
            candidate = indexed.get((method, candidate_name, seed))
            if baseline is None or candidate is None:
                continue
            pairs.append({
                "comparison": comparison,
                "method": method,
                "seed": seed,
                "baseline_transport": baseline_name,
                "candidate_transport": candidate_name,
                "baseline_best_val_f1": baseline["best_val_f1"],
                "candidate_best_val_f1": candidate["best_val_f1"],
                "candidate_minus_baseline_f1": (
                    candidate["best_val_f1"] - baseline["best_val_f1"]
                ),
                "baseline_best_val_loss": baseline["best_val_loss"],
                "candidate_best_val_loss": candidate["best_val_loss"],
                "candidate_minus_baseline_loss": (
                    candidate["best_val_loss"] - baseline["best_val_loss"]
                ),
            })

    grouped: dict[tuple[str, str], list[dict]] = {}
    for pair in pairs:
        grouped.setdefault((pair["comparison"], pair["method"]), []).append(pair)
    paired_aggregates = []
    for (comparison, method), group in sorted(grouped.items()):
        deltas = [row["candidate_minus_baseline_f1"] for row in group]
        paired_aggregates.append({
            "comparison": comparison,
            "method": method,
            "baseline_transport": group[0]["baseline_transport"],
            "candidate_transport": group[0]["candidate_transport"],
            "paired_seeds": len(group),
            "mean_candidate_minus_baseline_f1": statistics.fmean(deltas),
            "std_candidate_minus_baseline_f1": (
                statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            ),
            "candidate_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "candidate_losses": sum(delta < 0 for delta in deltas),
        })
    return pairs, paired_aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    metrics_files = sorted(root.glob("*/*/seed_*/model/metrics.jsonl"))
    if not metrics_files:
        raise ValueError(f"No benchmark metrics found below {root}")

    runs = [summarize_run(root, path) for path in metrics_files]
    runs.sort(key=lambda row: (row["method"], row["transport"], row["seed"]))
    aggregates = aggregate(runs)
    pairs, paired_aggregates = paired_comparisons(runs)
    aggregate_fields = [
        "method", "transport", "runs", "mean_best_val_f1", "std_best_val_f1",
        "mean_best_val_loss", "mean_train_val_f1_gap",
        "mean_trainable_parameters", "mean_latency_ms_per_example",
    ]
    paired_fields = [
        "comparison", "method", "seed", "baseline_transport",
        "candidate_transport", "baseline_best_val_f1", "candidate_best_val_f1",
        "candidate_minus_baseline_f1", "baseline_best_val_loss",
        "candidate_best_val_loss", "candidate_minus_baseline_loss",
    ]
    paired_aggregate_fields = [
        "comparison", "method", "baseline_transport", "candidate_transport",
        "paired_seeds", "mean_candidate_minus_baseline_f1",
        "std_candidate_minus_baseline_f1", "candidate_wins", "ties",
        "candidate_losses",
    ]

    write_csv(root / "runs.csv", RUN_FIELDS, runs)
    write_csv(root / "aggregate.csv", aggregate_fields, aggregates)
    write_csv(root / "paired_deltas.csv", paired_fields, pairs)
    write_csv(
        root / "paired_aggregate.csv", paired_aggregate_fields, paired_aggregates
    )
    (root / "runs.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "aggregate.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "paired_deltas.json").write_text(
        json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "paired_aggregate.json").write_text(
        json.dumps(paired_aggregates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Summarized {len(runs)} runs in {root}")


if __name__ == "__main__":
    main()
