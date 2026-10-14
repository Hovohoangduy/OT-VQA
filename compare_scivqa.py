"""Compare SciVQA predictions with the paper using official ROUGE settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rouge_score.rouge_scorer import RougeScorer


PAPER_ROUGE_L_F1 = {"dev": 0.729, "test": 0.740}


def compare(gold_path, predictions_path, split):
    gold = pd.read_csv(gold_path, keep_default_na=False, dtype=str)
    predictions = pd.read_csv(predictions_path, keep_default_na=False, dtype=str)
    for name, frame, required in (("gold", gold, {"anno_id", "answer"}),
                                  ("predictions", predictions, {"anno_id", "prediction"})):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
        if frame["anno_id"].duplicated().any():
            raise ValueError(f"{name} CSV contains duplicate anno_id values")
    if set(gold.anno_id) != set(predictions.anno_id):
        missing = len(set(gold.anno_id) - set(predictions.anno_id))
        extra = len(set(predictions.anno_id) - set(gold.anno_id))
        raise ValueError(f"Prediction IDs do not match gold: {missing} missing, {extra} extra")
    merged = gold.merge(predictions[["anno_id", "prediction"]], on="anno_id",
                        how="left", validate="one_to_one", sort=False)
    scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(reference, prediction)
              for reference, prediction in zip(merged.answer, merged.prediction)]
    count = len(scores)
    if not count:
        raise ValueError("Cannot compare an empty split")
    rouge_1 = sum(score["rouge1"].fmeasure for score in scores) / count
    rouge_l = sum(score["rougeL"].fmeasure for score in scores) / count
    target = PAPER_ROUGE_L_F1[split]
    return {"split": split, "examples": count,
            "rouge_1_f1": rouge_1, "rouge_l_f1": rouge_l,
            "paper_internvl3_rouge_l_f1": target,
            "rouge_l_gap_to_paper": rouge_l - target,
            "rouge": {"implementation": "rouge-score", "use_stemmer": True,
                      "aggregation": "mean of per-answer scores"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--gold_csv", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()
    gold = args.gold_csv or Path("data/scivqa_dataset") / (
        "val.csv" if args.split == "dev" else "test.csv")
    result = compare(gold, args.predictions_csv, args.split)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
