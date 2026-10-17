# SciVQA LLM/VLM evaluation

The existing `scripts.evaluate_llm` entry point supports all six native model
presets on SciVQA. Use the same prompt/context settings across models. This is
zero-shot inference, without training examples, paper retrieval, or ensembling.

## Dataset audit (2026-10-17)

The local CSVs were checked against all records in the cached official July
2025 JSON files. The 16 source fields are retained, with `instance_id` renamed
to `anno_id`, `image_file` to `image`, and the source question retained as
`original_question`. Options use JSON serialization; reference answers have
exporter whitespace stripping. No source-field mismatches were found under
those conversion rules. The local `metadata.json` records `revision: main`,
not an immutable Hub commit; `run_config.json` additionally records the actual
CSV SHA-256 and resolved model commit for each evaluation.

| Local split | QA pairs | Unique images | Pairs with caption | Pairs with options | Compound-figure pairs | Missing image files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 15,120 | 2,160 | 15,120 | 4,320 | 5,803 | 0 |
| val | 1,680 | 240 | 1,680 | 480 | 616 | 0 |
| test | 4,200 | 600 | 4,200 | 1,200 | 1,778 | 0 |

Each split contains all seven QA types equally: 2,160, 240, and 600 questions
per type, respectively. Annotation IDs and image filenames are disjoint across
splits. Source publications are not disjoint: train/val share 72 paper IDs,
train/test share 167, and val/test share 22. These are the supplied official
splits; do not describe them as a paper-held-out evaluation. The audit checked
file existence and metadata, without running inference on test or decoding all
images for visual integrity.

Local CSV SHA-256 values:

```text
train  5d77bc6670f05f4d57f0c87369d4fb3a1768bc18a01ac0e1361e52ed73785613
val    6322dc4b28569278e2ad7d070c7f7ea8fdcca108c36ab792c8c137e7cc8593ce
test   a774652181cc89ebbc98c2b5ddbbb5658350d05ebda5feba669e26ceda2d1cfd
```

## Model context

The figure context follows `utils/vqa_dataset.py` on branch `sci-vqa`
(inspected at `6f2b681`, `scivqa-v2` prompt construction). It is implemented in
the lightweight `utils/scivqa_prompt.py`, so native evaluation does not import
the training dataset or change the OT model. The local exporter used
`include_caption=false`; the evaluator reconstructs context directly from the
retained fields, without requiring re-export.

| Field | Inference use |
| --- | --- |
| `image` | Full figure through each model's native image processor. |
| `original_question` | Source question, before exported context was appended. |
| `answer_options` | Every labeled option, retaining source order and text, including multiple-correct questions. |
| `caption` | Full figure caption; no manual character/token truncation. |
| `figure_type` | Figure type and reading guidance for axes, labels, colors, trends, etc. |
| `compound`, `figs_numb` | Whether the figure contains multiple panels and how many; follow the panel named in the question. `figs_numb` is not a panel index. |
| `qa_pair_type` | Only binary versus non-binary answer-format guidance. Visual/non-visual and unanswerable labels are not disclosed. Retained for report grouping. |
| `answer` | Scoring only. Never used to construct the prompt or rewrite predictions. |
| `anno_id`, `figure_id`, `paper_id`, `pdf_url`, `venue`, `categories`, `source_dataset` | Traceability/grouping metadata, not added to the prompt; no publication downloads. |

The official data contains figure captions and metadata, not a separate OCR
transcript or full publication text. The native VLM reads labels from the image.
The dataset card defines answerable questions using the image or image plus
caption; external article retrieval would introduce a different protocol.

All questions receive the same system instruction, including a fixed abstention
format: `It is not possible to answer this question based only on the provided
data.` The model decides whether information is sufficient. The gold
`unanswerable` annotation never triggers abstention or prediction rewriting.
Task guidance requests Yes/No or option letters as appropriate; general answers
retain the values, units, and details needed by the question.

`--scivqa-context full` is the default. `question-only` retains the question,
options, and answer-format guidance, excluding caption, type, and panel guidance.
It is an optional controlled ablation, not evidence that either mode improves
accuracy. Both modes require context-preserving CSVs from `utils.download_scivqa`.
Malformed options or missing required columns fail before model downloads.

## Commands

Install `requirements.txt` for the native models, or
`requirements-llava-onevision.txt` in a separate environment for OneVision 1.5.
Real inference requires the evaluator's supported NVIDIA CUDA environment.

```bash
# CPU-only prompt inspection; no model download or metric dependencies required.
python3 -m scripts.evaluate_llm --dataset scivqa --preview-prompts 7

# Smoke run on test, followed by full test in a separate folder.
python3 -m scripts.evaluate_llm --dataset scivqa --model qwen3-vl-8b \
  --limit 7 --output-dir results/qwen3_scivqa_test_smoke
python3 -m scripts.evaluate_llm --dataset scivqa --model qwen3-vl-8b

# Another model, same full-context test protocol.
python3 -m scripts.evaluate_llm --dataset scivqa --model internvl3.5-8b

# Optional context ablation on validation.
python3 -m scripts.evaluate_llm --dataset scivqa --model qwen3-vl-8b \
  --split dev --scivqa-context question-only

# Resume test with identical settings; use dev explicitly for development.
python3 -m scripts.evaluate_llm --dataset scivqa --model qwen3-vl-8b --resume
python3 -m scripts.evaluate_llm --dataset scivqa --model qwen3-vl-8b --split dev
```

SciVQA defaults to `test.csv` (`--split test`, all 4,200 questions), 256 output tokens, greedy decoding,
and an 8,192-token input-plus-output budget. Image tokens count toward the budget.
Overlong examples fail explicitly rather than silently losing caption/question
context. Adjust `--context-size` only within model and hardware capacity.
Use `--questions` and `--images` for custom paths; the actual CSV path and its
hash identify the evaluated data, even when explicitly overriding `--split`.

For Kaggle, use `notebooks/evaluate_llm_kaggle.ipynb`. It bundles the evaluator,
context builder, and metrics. Set `DATA_ROOT` to the attached dataset folder.
Its defaults select SciVQA, `SPLIT="test"`, `FULL_LIMIT=None`, and `SCIVQA_CONTEXT="full"`.
Both CLI and notebook support the same six model presets and their existing
architecture-specific runtimes.

## Metrics and outputs

`report.json` contains nine macro-averaged metrics: ROUGE-1, ROUGE-L, and
BERTScore, each with precision, recall, and F1. `by_qa_pair_type` repeats these
metrics for each labeled QA type. This uses the repository's existing SciVQA
scorer: ROUGE's lowercase alphanumeric tokenizer without stemming, and
`roberta-large` BERTScore without baseline rescaling. These are explicit project
settings, not a claim of identical tokenization/encoder settings to every paper.
`--no-bertscore` computes six ROUGE values and leaves three BERTScore values null.

Predictions and references use the existing LLM whitespace/Markdown cleanup,
preserving option labels, case, punctuation, and numerical content. No
gold-dependent answer normalization or choice reordering is added. Failed
labeled requests contribute zero to computed metrics; unlabeled examples are
excluded, and unattempted questions are excluded from partial-run averages.

`predictions.jsonl` preserves the effective user prompt in `question`, original
question/context/provenance metadata, raw response, prediction, reference,
latency, and lexical scores. It is an append-only inference journal; BERTScore
is computed during report aggregation. Resume uses the latest row for each ID,
skips successful inference, retries failures, and recomputes aggregate scores.
`run_config.json` records context mode, prompt version/hash, input CSV hash,
system instruction, model identity, backend, and scorer settings. Changing any
of these prevents resume into an incompatible run. The default SciVQA output
folder includes split and context mode to keep experiments separate.

Sources: [official dataset card](https://huggingface.co/datasets/katebor/SciVQA),
[SciVQA prompting and evaluation paper](https://arxiv.org/html/2507.06183v1).

## Implementation verification

The following checks ran locally for this change:

```bash
/tmp/ot-vqa-scivqa-eval-venv/bin/python -m unittest discover -s tests -v
python3 -m scripts.evaluate_llm --help
python3 -m scripts.evaluate_llm --dataset scivqa --preview-prompts 7
git diff --check
```

All 27 tests passed. The temporary environment installed `rouge-score` and
`packaging`; inference/runtime and BERTScore model calls were mocked. Tests cover
context assembly, option validation, no gold/answerability leakage, caption
deduplication, CLI and notebook settings, scoring failures/unlabeled examples,
all three semantic components, and resume/retry/configuration rejection.
The real-data preview and whitespace check passed. Full native model inference
and real BERTScore model loading have not been run locally; no VQA accuracy gain
is claimed.
