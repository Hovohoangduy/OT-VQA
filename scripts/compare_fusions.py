"""Train and evaluate fusion methods under identical dataset and run settings.

Example:
  python -m scripts.compare_fusions --output_dir results/fusion_comparison \
    --device cuda --epochs 10 --batch_size 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from model.fusion import FUSION_METHODS
from configs.arg_parser import get_args
from utils.device import resolve_device
from utils.metrics import PAPER_METRICS


def collect_results(output_dir, methods, seeds, split):
    """Read completed reports; missing reports remain visible as incomplete runs."""
    rows = []
    for seed in seeds:
        for method in methods:
            run_dir = output_dir / f"{method}_seed{seed}"
            report_path = run_dir / f"{split}_report.json"
            if not report_path.is_file():
                rows.append({"fusion": method, "seed": seed, "status": "incomplete"})
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            if report["fusion"] != method or report["split"] != split:
                raise ValueError(f"Report does not match requested run: {report_path}")
            row = {"fusion": method, "seed": seed, "status": "complete",
                   "split": split, "checkpoint": report["checkpoint"],
                   "train_examples": manifest["train_examples"],
                   "dev_examples": manifest["dev_examples"],
                   "train_csv_sha256": manifest["train_csv_sha256"],
                   "dev_csv_sha256": manifest["dev_csv_sha256"],
                   "evaluation_csv_sha256": report["evaluation_csv_sha256"],
                   "loss": report["loss"],
                   "trainable_parameters": report["trainable_parameters"],
                   "examples": report["performance"]["examples"],
                   "examples_per_second": report["performance"]["examples_per_second"],
                   "milliseconds_per_example": report["performance"]["milliseconds_per_example"],
                   "peak_cuda_bytes": report["performance"]["peak_cuda_bytes"]}
            row.update(report["generated_metrics"])
            rows.append(row)
    return rows


def write_comparison(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row["status"] == "complete"]
    for key in ("train_csv_sha256", "dev_csv_sha256", "evaluation_csv_sha256",
                "train_examples", "dev_examples", "examples", "split"):
        if len({row[key] for row in completed}) > 1:
            raise ValueError(f"Cannot compare runs with different {key}")
    columns = ("fusion", "seed", "status", "split", "checkpoint", "train_examples",
               "dev_examples", "train_csv_sha256", "dev_csv_sha256",
               "evaluation_csv_sha256", "trainable_parameters", "loss",
               *PAPER_METRICS, "examples", "examples_per_second",
               "milliseconds_per_example", "peak_cuda_bytes")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in dict.fromkeys(row["fusion"] for row in rows):
        group = [row for row in completed if row["fusion"] == method]
        item = {"fusion": method, "runs": len(group)}
        for key in ("loss", *PAPER_METRICS, "milliseconds_per_example"):
            item[f"mean_{key}"] = statistics.mean(row[key] for row in group) if group else None
            item[f"std_{key}"] = (statistics.stdev(row[key] for row in group)
                                    if len(group) > 1 else None)
        summary.append(item)
    summary_columns = ("fusion", "runs", *(f"{stat}_{key}"
                       for key in ("loss", *PAPER_METRICS, "milliseconds_per_example")
                       for stat in ("mean", "std")))
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Fusion comparison", "",
             "Each run uses its lowest validation-loss checkpoint. Metrics come from generated answers.",
             "", "## Per-seed results", "",
             "| Fusion | Seed | Status | Trainable parameters | Loss | EM | Token F1 | BLEU-1 | BLEU-2 | ROUGE-L | BERTScore F1 | ms/example |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["fusion"], str(row["seed"]), row["status"]]
        values.append(f"{row['trainable_parameters']:,}" if "trainable_parameters" in row else "—")
        values += [f"{row[key]:.4f}" if key in row else "—" for key in
                   ("loss", *PAPER_METRICS, "milliseconds_per_example")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Across seeds", "",
              "Sample standard deviations require at least two completed seeds.", "",
              "| Fusion | Runs | EM mean ± SD | Token F1 mean ± SD | BERTScore F1 mean ± SD | ms/example mean ± SD |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        values = [item["fusion"], str(item["runs"])]
        for key in ("em", "token_f1", "bertscore_f1", "milliseconds_per_example"):
            mean = item[f"mean_{key}"]
            std = item[f"std_{key}"]
            values.append("—" if mean is None else f"{mean:.4f} ± {std:.4f}" if std is not None
                          else f"{mean:.4f} ± —")
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=FUSION_METHODS,
                        default=list(FUSION_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1105])
    parser.add_argument("--output_dir", type=Path, default=Path("results/fusion_comparison"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--train_csv_path", default="data/gqa_dataset/train.csv")
    parser.add_argument("--dev_csv_path", default="data/gqa_dataset/val.csv")
    parser.add_argument("--test_csv_path", default="data/gqa_dataset/test.csv")
    parser.add_argument("--img_path", default="data/gqa_dataset/images")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bertscore_model", default="bert-base-uncased")
    parser.add_argument("--bertscore_device", default="cpu")
    parser.add_argument("--bertscore_batch_size", type=int, default=16)
    parser.add_argument("--bertscore_rescale", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--force", action="store_true", help="Retrain completed runs")
    args, extra_train_args = parser.parse_known_args(argv)
    if any(option.split("=")[0] in {"--fusion", "--model_path", "--seed", "--resume"}
           for option in extra_train_args):
        parser.error("fusion, model_path, seed and resume are controlled by this runner")
    project_root = Path(__file__).resolve().parents[1]
    common = ["--train_csv_path", args.train_csv_path, "--dev_csv_path", args.dev_csv_path,
              "--test_csv_path", args.test_csv_path, "--img_path", args.img_path,
              "--device", args.device, "--batch_size", str(args.batch_size),
              "--bertscore_model", args.bertscore_model,
              "--bertscore_device", args.bertscore_device,
              "--bertscore_batch_size", str(args.bertscore_batch_size),
              "--bertscore_rescale" if args.bertscore_rescale else "--no-bertscore_rescale"]
    output_dir = args.output_dir.resolve()
    for seed in args.seeds:
        for method in args.methods:
            run_dir = output_dir / f"{method}_seed{seed}"
            checkpoint = run_dir / "best.pt"
            report = run_dir / f"{args.split}_report.json"
            train_command = [sys.executable, str(project_root / "train.py"),
                             *common, "--fusion", method, "--seed", str(seed),
                             "--epochs", str(args.epochs), "--model_path", str(run_dir),
                             *extra_train_args]
            if checkpoint.is_file() and not args.force:
                manifest_path = run_dir / "run_config.json"
                if not manifest_path.is_file() or json.loads(
                        manifest_path.read_text(encoding="utf-8"))["arguments"] != vars(
                            get_args(train_command[2:])):
                    raise ValueError(f"Existing checkpoint has different training settings: {run_dir}. "
                                     "Use --force or a new output directory.")
            if args.force or not checkpoint.is_file():
                print("Training", method, "seed", seed, flush=True)
                subprocess.run(train_command, cwd=project_root, check=True)
            if report.is_file() and not args.force:
                saved = json.loads(report.read_text(encoding="utf-8"))
                eval_csv = args.dev_csv_path if args.split == "dev" else args.test_csv_path
                csv_digest = hashlib.sha256((project_root / eval_csv).read_bytes()).hexdigest()
                bert = saved.get("bertscore", {})
                if (saved.get("fusion") != method or saved.get("evaluation_csv_sha256") != csv_digest
                        or bert.get("model") != args.bertscore_model
                        or bert.get("rescale_with_baseline") != args.bertscore_rescale
                        or bert.get("batch_size") != args.bertscore_batch_size
                        or bert.get("device") != str(resolve_device(args.bertscore_device))):
                    raise ValueError(f"Existing report has different evaluation settings: {report}. "
                                     "Use --force or a new output directory.")
            if args.force or not report.is_file():
                eval_command = [sys.executable, str(project_root / "test.py"),
                                *common, "--split", args.split, "--checkpoint", str(checkpoint),
                                "--report_json", str(report),
                                "--predictions_csv", str(run_dir / f"{args.split}_predictions.csv")]
                print("Evaluating", method, "seed", seed, flush=True)
                subprocess.run(eval_command, cwd=project_root, check=True)
            write_comparison(output_dir, collect_results(output_dir, args.methods,
                                                         args.seeds, args.split))
    print(f"Comparison written to {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
