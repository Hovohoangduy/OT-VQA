"""Create fixed dev screening manifests or paired figure/paper bootstrap reports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from utils.metrics import scivqa_score_pairs
from utils.quality_evidence import guardrail_metrics, paired_cluster_interval


def aligned_predictions(gold, path):
    predictions = pd.read_csv(path, keep_default_na=False, dtype=str)
    if predictions.anno_id.duplicated().any() or gold.anno_id.duplicated().any():
        raise ValueError("Duplicate annotation IDs")
    if set(predictions.anno_id) != set(gold.anno_id):
        raise ValueError("Prediction IDs must cover exactly the selected dev manifest")
    return gold[["anno_id"]].merge(predictions[["anno_id", "prediction"]], on="anno_id", validate="one_to_one").prediction.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev_csv", default="data/scivqa_dataset/val.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--screen_figures", type=int, default=80)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    frame = pd.read_csv(args.dev_csv, keep_default_na=False, dtype=str)
    if not args.baseline and not args.candidate:
        # Proportional stratification by figure type; complete figures only.
        figures = frame.drop_duplicates("image")
        if not 1 <= args.screen_figures <= len(figures):
            raise ValueError("Invalid screening figure count")
        rng = np.random.default_rng(args.seed)
        groups = list(figures.groupby("figure_type", sort=True))
        allocations = [len(g)*args.screen_figures/len(figures) for _, g in groups]
        quotas = np.floor(allocations).astype(int)
        for i in np.argsort(-(np.array(allocations)-quotas), kind="stable")[:args.screen_figures-sum(quotas)]:
            quotas[i] += 1
        images = []
        for (_, group), count in zip(groups, quotas):
            images.extend(rng.choice(group.image.tolist(), count, replace=False).tolist())
        selected = frame[frame.image.isin(images)]
        output = {"seed": args.seed, "images": images, "anno_ids": selected.anno_id.tolist(),
                  "dev_csv_sha256": hashlib.sha256(Path(args.dev_csv).read_bytes()).hexdigest()}
    else:
        if not args.baseline or not args.candidate:
            parser.error("Comparison requires both baseline and candidate")
        baseline, candidate = [aligned_predictions(frame, path) for path in (args.baseline, args.candidate)]
        score = lambda preds: np.array([r["rouge_l_f1"] for r in scivqa_score_pairs(frame.answer.tolist(), preds, compute_bertscore=False)])
        a, b = score(baseline), score(candidate)
        output = {"examples": len(frame), "baseline": float(a.mean()), "candidate": float(b.mean()),
                  "figure_interval": paired_cluster_interval(b-a, frame.image, args.seed),
                  "paper_interval": paired_cluster_interval(b-a, frame.paper_id, args.seed),
                  "groups": {kind: {"baseline": float(a[idx].mean()), "candidate": float(b[idx].mean())}
                             for kind, idx in frame.groupby("qa_pair_type").indices.items()},
                  "guardrails": guardrail_metrics(frame.answer.tolist(), candidate)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"output": str(path), "examples": output.get("examples", len(output.get("anno_ids", []))) }))


if __name__ == "__main__":
    main()
