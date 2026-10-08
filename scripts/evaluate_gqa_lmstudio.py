"""Zero-shot GQA evaluation through LM Studio; Python standard library only.

Run from the repository root:
    python3 -m scripts.evaluate_gqa_lmstudio --limit 5 --output-dir results/qwen_smoke
    python3 -m scripts.evaluate_gqa_lmstudio

Each request contains only an image and question, with no demonstrations,
reference answers, scene graphs, or chat history. Images retain their original
resolution; LM Studio handles the model's image preprocessing.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


SYSTEM_PROMPT = (
    "Answer the question using the image. Return only the short answer, "
    "usually one word or a short phrase. For yes/no questions, return yes or no. "
    "Do not include an explanation, a full sentence, or an 'Answer:' prefix."
)


@dataclass(frozen=True)
class Example:
    question_id: str
    image: str
    question: str
    answer: str | None


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
                    row["image"], row["question"], row.get("answer") or None,
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


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0]
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image format: {path}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class LMStudioClient:
    def __init__(self, base_url: str, timeout: float, retries: int):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.timeout = timeout
        self.retries = retries

    def request(self, endpoint: str, payload: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        # Set this only when authentication is enabled in LM Studio.
        if os.environ.get("LM_STUDIO_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["LM_STUDIO_API_KEY"]
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"LM Studio HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Cannot reach {self.base_url}: {error}. "
                        "Check that LM Studio's server is running and the vision model is loaded."
                    ) from error
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Request exhausted its retry attempts.")

    def models(self) -> list[str]:
        return [item["id"] for item in self.request("/models")["data"]]

    def answer(self, model: str, example: Example, path: Path,
               max_tokens: int, seed: int) -> dict:
        start = time.perf_counter()
        response = self.request("/chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(path)}},
                    {"type": "text", "text": example.question},
                ]},
            ],
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": False,
        })
        choice = response["choices"][0]
        raw = choice["message"].get("content")
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("The model returned no answer text. Check max-tokens and model settings.")
        return {
            "raw_response": raw,
            "prediction": raw.strip(),
            "finish_reason": choice.get("finish_reason"),
            "latency_seconds": time.perf_counter() - start,
            "usage": response.get("usage", {}),
        }


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def make_record(example: Example, result: dict | None = None, error: str = "") -> dict:
    record = {
        "question_id": example.question_id, "image": example.image,
        "question": example.question, "reference": example.answer,
        "prediction": "", "raw_response": "", "error": error,
    }
    record.update(result or {})
    labeled = example.answer is not None
    record["correct"] = (
        bool(not error and record["prediction"].strip() == example.answer.strip()) if labeled else None
    )
    record["normalized_correct"] = (
        bool(not error and normalize(record["prediction"]) == normalize(example.answer))
        if labeled else None
    )
    return record


def save_outputs(output: Path, records: dict[str, dict], config: dict,
                 selected_count: int) -> dict:
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
        "accuracy": sum(row["correct"] for row in labeled) / len(labeled) if labeled else None,
        "normalized_accuracy": (
            sum(row["normalized_correct"] for row in labeled) / len(labeled) if labeled else None
        ),
        "truncated_responses": sum(row.get("finish_reason") == "length" for row in successful),
        "mean_latency_seconds": (
            sum(row["latency_seconds"] for row in successful) / len(successful)
            if successful else None
        ),
        "scoring": {
            "accuracy": "Exact match after trimming response whitespace; one point per correct answer.",
            "normalized_accuracy": "Exact match ignoring case and repeated whitespace.",
            "errors": "Failed labeled requests count as incorrect; unattempted questions are excluded.",
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["question_id", "image", "question", "reference", "prediction", "correct",
              "normalized_correct", "latency_seconds", "finish_reason", "error"]
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    # Official GQA prediction structure. Repository anno_id values are image IDs,
    # so use official question JSON if exporting predictions for the GQA evaluator.
    (output / "gqa_predictions.json").write_text(json.dumps([
        {"questionId": row["question_id"], "prediction": row["prediction"]}
        for row in successful
    ], indent=2) + "\n", encoding="utf-8")
    return report


def evaluate(args: argparse.Namespace, client: LMStudioClient) -> dict:
    examples = load_examples(args.questions)
    if args.limit is not None:
        examples = examples[:args.limit]
    # Validate the selected files before starting a potentially long inference run.
    paths = {example.question_id: image_path(args.images, example, args.image_split or args.questions.stem)
             for example in examples}
    config = {
        "questions": str(args.questions.resolve()),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "images": str(args.images.resolve()),
        "image_split": args.image_split or args.questions.stem,
        "model": args.model, "base_url": client.base_url,
        "limit": args.limit, "seed": args.seed,
        "temperature": 0, "max_tokens": args.max_tokens,
        "system_prompt": SYSTEM_PROMPT, "protocol": "zero-shot",
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_config.json"
    journal = output / "predictions.jsonl"
    records = {}
    if args.resume:
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration differs or is missing. Use the original flags or a new output-dir.")
        if journal.is_file():
            for line in journal.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                records[row["question_id"]] = row
    else:
        if any(output.iterdir()):
            raise ValueError(f"Output directory is not empty: {output}. Use --resume or a new output-dir.")
        manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Model: {args.model}; selected questions: {len(examples)}; zero-shot", flush=True)
    try:
        with journal.open("a", encoding="utf-8") as handle:
            for index, example in enumerate(examples, 1):
                previous = records.get(example.question_id)
                if previous is not None and not previous["error"]:
                    continue
                failure = None
                try:
                    result = client.answer(args.model, example, paths[example.question_id],
                                           args.max_tokens, args.seed)
                    record = make_record(example, result)
                except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
                    failure = error
                    record = make_record(example, error=str(error))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records[example.question_id] = record
                print(f"[{index}/{len(examples)}] {example.question_id}: "
                      f"{record['prediction']!r} | reference={example.answer!r}"
                      + (f" | ERROR: {failure}" if failure else ""), flush=True)
                if failure and not args.continue_on_error:
                    raise RuntimeError("Stopped after an error; predictions are saved. Fix it and use --resume.") from failure
    finally:
        report = save_outputs(output, records, config, len(examples))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen3-vl-2b-instruct", help="Exact identifier from --list-models")
    parser.add_argument("--list-models", action="store_true", help="Print model identifiers and exit")
    parser.add_argument("--questions", type=Path, default=Path("data/gqa_dataset/test.csv"))
    parser.add_argument("--images", type=Path, default=Path("data/gqa_dataset/images"))
    parser.add_argument("--image-split", help="Optional split subfolder for flat image names")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_vl_2b_gqa_zero_shot"))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions (default: all)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="LM Studio sampling seed")
    parser.add_argument("--timeout", type=float, default=180, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Additional attempts for transient API failures")
    parser.add_argument("--resume", action="store_true", help="Reuse saved successes and retry failed questions")
    parser.add_argument("--continue-on-error", action="store_true", help="Record errors and continue; errors count as wrong")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_tokens <= 0 or args.timeout <= 0 or args.retries < 0:
        parser.error("max-tokens and timeout must be positive; retries must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = LMStudioClient(args.base_url, args.timeout, args.retries)
    try:
        models = client.models()
        if args.list_models:
            print("\n".join(models))
            return 0
        if args.model not in models:
            raise ValueError(f"Model {args.model!r} was not found. Available IDs: {', '.join(models)}")
        report = evaluate(args, client)
    except KeyboardInterrupt:
        print("\nInterrupted. Saved completed predictions; rerun with --resume.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if report["accuracy"] is not None:
        print(f"Accuracy: {report['accuracy']:.2%}; normalized: {report['normalized_accuracy']:.2%}")
    else:
        print("No reference answers supplied; predictions saved without accuracy scoring.")
    print(f"Results: {args.output_dir / 'report.json'}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
