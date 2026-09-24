"""Paired SAN-versus-OT answer comparison on the same held-out examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path):
    frame = pd.read_csv(path, keep_default_na=False, dtype={"anno_id": str})
    required = {"anno_id", "question", "reference", "em", "f1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["anno_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate anno_id values")
    return frame.set_index("anno_id").sort_index()


def compare_runs(baseline_paths, ot_paths, bootstrap_samples=5000, seed=1105):
    if len(baseline_paths) != len(ot_paths) or not baseline_paths:
        raise ValueError("Provide the same positive number of SAN and OT prediction files")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    per_seed = []
    example_differences = {"em": [], "f1": []}
    reference_index = reference_questions = reference_answers = None
    question_types = None
    for baseline_path, ot_path in zip(baseline_paths, ot_paths):
        baseline, ot = _read(baseline_path), _read(ot_path)
        if not baseline.index.equals(ot.index):
            raise ValueError("SAN and OT files must contain the same anno_id values")
        if not baseline["question"].equals(ot["question"]) or not baseline["reference"].equals(ot["reference"]):
            raise ValueError("Paired files disagree on questions or reference answers")
        if reference_index is None:
            reference_index = baseline.index
            reference_questions = baseline["question"]
            reference_answers = baseline["reference"]
            if "question_type" in baseline:
                question_types = baseline["question_type"]
        elif (not reference_index.equals(baseline.index) or
              not reference_questions.equals(baseline["question"]) or
              not reference_answers.equals(baseline["reference"])):
            raise ValueError("Every seed must use the same held-out examples")
        if question_types is not None and ("question_type" not in baseline or
                                           not question_types.equals(baseline["question_type"])):
            raise ValueError("Question types differ across seed files")
        row = {"baseline": str(baseline_path), "ot": str(ot_path)}
        for metric in ("em", "f1"):
            base = baseline[metric].to_numpy(dtype=float)
            candidate = ot[metric].to_numpy(dtype=float)
            row[f"san_{metric}"] = float(base.mean())
            row[f"ot_{metric}"] = float(candidate.mean())
            row[f"delta_{metric}"] = float(candidate.mean() - base.mean())
            example_differences[metric].append(candidate - base)
        per_seed.append(row)
    rng = np.random.default_rng(seed)
    count = len(reference_index)
    if not count:
        raise ValueError("Prediction files are empty")
    summary = {"examples": count, "seeds": len(per_seed), "per_seed": per_seed}
    for metric in ("em", "f1"):
        differences = np.mean(np.stack(example_differences[metric]), axis=0)
        samples = np.empty(bootstrap_samples)
        for sample_index in range(bootstrap_samples):
            sampled = rng.integers(0, count, size=count)
            samples[sample_index] = differences[sampled].mean()
        summary[metric] = {
            "mean_delta": float(differences.mean()),
            "seed_delta_std": float(np.std([row[f"delta_{metric}"] for row in per_seed], ddof=1))
            if len(per_seed) > 1 else None,
            "paired_bootstrap_95_ci": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        }
    if question_types is not None:
        em_differences = np.mean(np.stack(example_differences["em"]), axis=0)
        summary["question_type_em_delta"] = {
            str(kind): {"examples": int((question_types.to_numpy() == kind).sum()),
                        "delta": float(em_differences[question_types.to_numpy() == kind].mean())}
            for kind in sorted(question_types.unique())
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--san", nargs="+", required=True, help="SAN prediction CSVs, one per seed")
    parser.add_argument("--ot", nargs="+", required=True, help="OT prediction CSVs, same seed order")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = compare_runs(args.san, args.ot, args.bootstrap_samples, args.seed)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
