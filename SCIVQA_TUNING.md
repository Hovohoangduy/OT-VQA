# Improving OT on SciVQA

These are implemented experiments, not measured improvements from retraining.
The local environment has CPU PyTorch and no CUDA, so the two-GPU training runs
must be performed in Kaggle. Start fresh runs in separate output directories.

## Evidence and comparison

The latest supplied training log peaks at validation ROUGE-L F1 **0.4626 at
epoch 5**. Training ROUGE-L reaches **0.7580** by epoch 13, while validation
ROUGE-L falls to **0.4259** and validation loss rises from its epoch-6 minimum
of **1.1333** to **1.2873**. Early stopping selects epoch 5. This indicates
overfitting; increasing the epoch limit alone will not address it. Training
loss uses label smoothing and dropout;
validation loss does not, so their absolute values are not directly comparable.

The existing `results/scivqa/test_report.json` reports test ROUGE-L **0.4557** and
BERTScore F1 **0.9241**, from `data/weights/OT_SciVQA.pt`. That checkpoint records
best epoch 19, so it is not necessarily the epoch-13 model from the supplied log.
The already available predictions show the following breakdown. These stored
test results are diagnostic context; use validation to choose new experiments.

| Question group | Examples | Existing test ROUGE-L F1 |
| --- | ---: | ---: |
| Binary, non-visual | 600 | 0.6017 |
| Binary, visual | 600 | 0.5633 |
| Multiple-choice, non-visual | 600 | 0.4063 |
| Multiple-choice, visual | 600 | 0.3435 |
| Infinite answer set, non-visual | 600 | 0.0977 |
| Infinite answer set, visual | 600 | 0.2206 |
| Unanswerable | 600 | 0.9567 |

A seeded 64-example **validation** diagnostic on that existing checkpoint
changed only **5/64 predictions (7.8%)** when replacing every image with a
different figure, compared with **59/64 (92.2%)** when replacing questions.
This small sample suggests weak image dependence; it is not a full-validation
quality estimate. The report is saved in
`results/scivqa/validation_bottleneck_review.json`. The diagnostic now avoids
replacing a figure with itself, which the previous within-batch roll could do
because SciVQA has seven adjacent questions per figure.

The second review expanded the image-replacement test to **all 1,680 validation
examples** using cached frozen image features and the same ROUGE scorer used
by training. Original ROUGE-1/L F1 was **0.4541**; replacing each image with the
next distinct figure produced **0.4482**. Only **208/1,680 predictions (12.4%)**
changed. The small 0.0059 score drop supports weak useful visual grounding in
this checkpoint, although one replacement test does not prove the image branch
is completely unused. Infinite-answer visual/non-visual groups scored
**0.1605 / 0.0957**, versus **0.9583** on unanswerable questions. Full details
are in `results/scivqa/full_validation_visual_review.json`. These measurements
evaluate the existing checkpoint, not the new padded-image training recipe.

The attached paper, *Enhancing Scientific Visual Question Answering through
Multimodal Reasoning and Ensemble Modeling*, reports validation ROUGE-L 0.729
for InternVL3 (Table 1), and test ROUGE-L 0.740 with CoT or 0.727 without CoT
(Table 2). Those models use 5B–8B pretrained multimodal systems, captions,
task-specific prompts and post-processing. This repository instead combines a
frozen ImageNet MobileViTV2, MiniLM and a newly initialized answer decoder.
The local SciVQA export has `include_caption: false`. The 50.2M number in the
log counts trainable parameters, not the entire model (the stored model totals
about 79.5M). The paper does not identify all BERTScore settings, so numerical
parity for BERTScore is unverified.
The [official SciVQA evaluator](https://github.com/esborisova/SciVQA/blob/main/src/eval/scoring.py)
uses ROUGE stemming. New runs now match that setting. Re-scoring the stored
4,200 test predictions with stemming leaves their ROUGE-L F1 at **0.4557**, so
this metric detail does not explain the gap to the paper. Existing JSON reports
still record the earlier no-stemming setting; they were not overwritten.

Across all 2,160 training figures, the old `Resize(288)` + `CenterCrop(256)`
retains a median **50.1%** of image area; the widest figure retains **4.6%**.
This is a geometry audit, not a score estimate. In the generated comparison
`results/scivqa/figure_preprocessing_comparison.png`, the crop removes the USA
panel title and y-axis values and clips the lower panel's x-axis labels.
The full-figure padded version preserves them. The audit data is saved in
`results/scivqa/training_data_preflight.json`.

Paper: https://arxiv.org/html/2507.06183v1

## Implemented changes

- After the second review, SciVQA starts with decoder/fusion LR `1e-5` and
  5% warmup. Dropout `0.2`, weight decay `0.05`, and label smoothing `0.1`
  retain the supplied command's regularization. The earlier `1e-4` default
  and simultaneous reduction in regularization were not supported by an LR
  comparison and have been withdrawn. The encoder LR `2e-6` applies only when
  question encoder fine-tuning is enabled. Other dataset defaults are unchanged.
- Fresh SciVQA runs now freeze the pretrained MiniLM question encoder as well
  as the image encoder and answer embeddings. This removes the largest source
  of trainable parameters from a 15,120-example training set and keeps its
  dropout disabled during training. The OT projection, fusion, and decoder
  remain trainable. `--no-freeze_question_encoder` restores fine-tuning for a
  controlled comparison at encoder LR `2e-6`. This change is an overfitting
  hypothesis; its validation effect has not been measured.
- Fresh SciVQA runs now default to `--loss_normalization example`, giving
  every answer equal weight after averaging its valid target-token losses.
  The training split has 2,160
  unanswerable examples out of 15,120 (14.3%), but they contribute 34,560 of
  71,846 non-padding targets (48.1%, counting EOS and excluding BOS). Each
  unanswerable target has 16 tokens; binary targets have only 2. Legacy
  token-averaged loss consequently weights a long answer more heavily than a
  short one, whereas reported ROUGE is a macro mean over questions. This is a
  loss/evaluation mismatch worth testing, not proof of causation. Use
  `--loss_normalization token` for a separate legacy control run.
  Validation CE remains token-averaged, and checkpoint resume preserves the
  saved training objective.
- New SciVQA runs select `best.pt` and early stopping by validation
  `rouge_l_f1`. Resumed checkpoints keep their original selection metric.
- `--image_preprocessing pad` preserves the entire figure and its aspect ratio
  on a white square. `--image_size 512` sends 512 pixels to MobileViTV2 and
  yields a 16×16 spatial grid instead of 8×8. Its processor cannot silently
  reduce that prepared batch back to 256 pixels. Higher resolution costs more
  compute and still uses an ImageNet encoder rather than an OCR model.
  New SciVQA training uses these image settings by default. Evaluation and
  resume inherit saved preprocessing; changing the preset cannot silently
  alter an existing checkpoint's inputs. Other datasets retain their defaults.
- `--ot_visual_memory` reuses the reverse direction of the same partial OT plan
  to condition spatial image tokens on question tokens. The decoder attends to
  global memory, OT-grounded question tokens, and these image tokens. The visual
  residual path preserves details that question-side barycentric averaging can
  lose. A learned 2D coordinate projection supplies positions. OT remains the
  fusion mechanism; this is an optional architecture experiment.
- Numerical output spacing is repaired (`0. 5` → `0.5`, `- 2. 75` → `-2.75`).
  This uses only generated text and preserves the digits. ROUGE's existing
  tokenizer already gives these decimal variants identical lexical tokens, so
  this formatting fix does **not** solve the low ROUGE scores.
- Validation history and evaluation reports include metrics by `qa_pair_type`.
  These annotations are used for reporting, not supplied to the model or used
  to decide whether an answer is unanswerable.
- Checkpoints store image size, preprocessing and visual-memory settings;
  training, evaluation, prediction and diagnostics use them consistently.
  Older checkpoints retain the legacy 256-pixel crop and original architecture.

## First run: full figure with frozen question encoder

Use the updated code. This is run C in the comparison table below: preserve the
complete figure at 512 pixels, freeze the pretrained question encoder, and keep the
original regularization and OT memory architecture. The leading `!` is for a
Kaggle notebook. This is a candidate experiment, not a guaranteed improvement.

```bash
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset scivqa --device cuda --fusion ot \
  --text_model sentence-transformers/all-MiniLM-L12-v2 \
  --epochs 30 --batch_size 8 \
  --d_model 384 --ffn_hidden 1024 --num_layers 2 --num_heads 4 \
  --lr 1e-5 --warmup_ratio 0.05 \
  --weight_decay 0.05 --label_smoothing 0.1 --drop_prob 0.2 \
  --loss_normalization example \
  --freeze_answer_embeddings --freeze_question_encoder --gradient_clip 1.0 \
  --early_stopping_patience 8 --selection_metric rouge_l_f1 \
  --max_question_tokens 256 --max_answer_tokens 64 \
  --train_eval_samples 512 --num_workers 2 \
  --image_size 512 --image_preprocessing pad --no-ot_visual_memory \
  --train_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/train.csv \
  --dev_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/val.csv \
  --img_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/images \
  --model_path /kaggle/working/scivqa_ot_pad512_frozen_question
```

Compare this run's best validation ROUGE-L F1 with **0.4626** from the supplied
log. Use its `best.pt`, not `last.pt`. For a control that isolates question
encoder freezing, repeat the command with `--no-freeze_question_encoder
--encoder_lr 2e-6` and a different `--model_path`. Do not resume the supplied
checkpoint into either run.

## Figure context run against the paper

The cited paper supplies the figure caption and figure type to its VLM prompts.
The local SciVQA CSVs already contain both fields, but the run above does not
read them into the model input. To test their contribution while keeping OT as
the fusion mechanism, repeat the command above with these replacements:

```bash
--figure_context --max_question_tokens 512 \
--model_path /kaggle/working/scivqa_ot_figure_context
```

With the current MiniLM tokenizer, the longest question plus figure context
in the local train/dev/test CSVs is 423 tokens including special tokens; 512
preserves all of them. At 256 tokens, 74 train, 2 dev, and 18 test examples
would be truncated.

Keep the same train/dev CSVs, seed, model, and other settings. Start fresh: the
input text has changed, so `--resume` rejects a conflicting context setting.
If the longer text exhausts Kaggle GPU memory, reduce `--batch_size` to 4 on
both comparison runs and record the resulting global batch size of 8.
The dataset puts the question before the figure type and caption, and avoids
appending a caption already present in an older export. `best.pt` records the
context setting so `test.py` and diagnostics use the same input at validation
and test time. For single-image inference, `predict.py` requires `--caption`
and `--figure_type` with this checkpoint.

The paper also provides QA pair type to its prompt. This run does not use that
label because the `unanswerable` type directly reveals an answer category.
Report that input difference when comparing scores. A higher score than the
paper's InternVL3 test ROUGE-L F1 of 0.740 is a target, not an expected result
of adding context; the paper uses a much larger, chart-capable VLM.

## Next runs: figure preservation and OT visual memory

Reuse the command above, changing only the listed flags and output directory.
Choose runs by the full validation metrics, with particular attention to
infinite-answer-set questions. Do not keep using the test split to pick settings.

After run C, compare it with an identical run using
`--loss_normalization token` and a separate output directory such as
`/kaggle/working/scivqa_ot_token_loss`. Keep whichever objective performs
better on full-validation ROUGE-L and the numerical question groups, then keep
that choice fixed when testing visual memory in D. A and B are optional controls
to measure preprocessing and resolution changes, using the same learning rates
and loss objective. Crop is retained as an explicit legacy control, not the
recommended SciVQA preprocessing. Do not switch objectives on resume.

| Run | Image preprocessing | Image size | Visual memory | Output directory suffix |
| --- | --- | ---: | --- | --- |
| A | crop | 256 | `--no-ot_visual_memory` | `ot_optimizer` |
| B | pad | 256 | `--no-ot_visual_memory` | `ot_pad256` |
| C | pad | 512 | `--no-ot_visual_memory` | `ot_pad512` |
| D | pad | 512 | `--ot_visual_memory` | `ot_visual512` |

For D, replace the image flags with:

```bash
--image_size 512 --image_preprocessing pad --ot_visual_memory
```

If D runs out of GPU memory, use `--batch_size 4`; that changes the global batch
from 16 to 8 and should be recorded. Alternatively test `--image_size 384`.
Do not resume the old checkpoint into a changed architecture.

After finding a useful visual configuration, compare decoder LRs `5e-6`,
`1e-5`, `3e-5` while keeping the question encoder frozen. If the fine-tuning
control performs better, compare encoder LRs `1e-6`, `2e-6`, `5e-6`
separately. Do not infer an optimal LR from the epoch
of the validation peak alone. Repeat the winner with seeds 1105, 42,
and 2026. OT epsilon, dustbin mass and iteration count remain at the original
0.05/0.2/20 for these experiments. Increase Sinkhorn iterations only if the
diagnostic residuals justify it; the existing checkpoint's residuals are already
below 2e-7 on the 64-example validation diagnostic.

Caption context is another independent experiment. The existing downloader
supports `--include-caption`; export all splits to a new directory and set
`--max_question_tokens 512` after checking truncation. Keep caption usage fixed
across train/validation/test and report it when comparing with the paper.

To evaluate D on validation:

```bash
!python test.py --dataset scivqa --device cuda --split dev --batch_size 8 \
  --checkpoint /kaggle/working/scivqa_ot_visual512/best.pt \
  --dev_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/val.csv \
  --img_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/images \
  --report_json /kaggle/working/scivqa_ot_visual512/val_report.json \
  --predictions_csv /kaggle/working/scivqa_ot_visual512/val_predictions.csv
```

Image settings are loaded from the checkpoint; no image flags are needed for
evaluation. Reevaluate the baseline with the same decoding and metric code.
After saving validation predictions, compare them with the paper's 0.729
InternVL3 validation ROUGE-L using the official ROUGE stemming setting:

```bash
python compare_scivqa.py --split dev \
  --gold_csv /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/val.csv \
  --predictions_csv /kaggle/working/scivqa_ot_figure_context/val_predictions.csv \
  --output_json /kaggle/working/scivqa_ot_figure_context/paper_comparison.json
```

The script joins predictions to gold by `anno_id`, rejects missing or duplicate
IDs, and prints the ROUGE-L gap. For the held-out test comparison, use `--split
test` with the test CSV and predictions; the paper's InternVL3 CoT target is
0.740. Do not select settings using test results.
For final test evaluation, change `--split dev` to `--split test`, supply
`--test_csv_path` ending in `test.csv`, and use distinct test report filenames.

## What a ROUGE-F1 target above 0.8 means

The supplied log's rising validation loss after epoch 6 and increasing training
ROUGE are consistent with overfitting, but do not identify an excessive LR as
the cause. That run's peak LRs were `1e-4` for the decoder/fusion and `1e-5`
for the text encoder; current SciVQA runs use `1e-5` for decoder/fusion and
freeze the text encoder by default. Fine-tuning controls use `2e-6` for it.
Lower rates can slow memorization or stabilize adaptation, but cannot provide missing
chart labels or numerical-reading pretraining.

A target above 0.8 is a research objective, not an expected outcome of this
recipe. Using the full-validation category scores as an illustration: even if
all five finite-answer/unanswerable groups reached 1.0, leaving the two
infinite-answer groups at 0.0957 and 0.1605 would cap overall ROUGE-L at
`(5 + 0.0957 + 0.1605) / 7 = 0.7509`. Numerical reading must improve, and the
current other-category scores also need substantial gains.

If visual dependence and numerical accuracy remain weak, a more credible
route toward 0.8 is a chart/OCR-pretrained vision encoder and a pretrained
generative decoder, with OT retained as the alignment adapter. For example,
[UniChart's official model card](https://huggingface.co/ahmed-masry/unichart-base-960)
describes chart pretraining and chart QA support. It is a candidate backbone,
not a verified 0.8 SciVQA model, and it cannot be selected by simply replacing
`--image_model` in the current MiniLM/MobileViTV2 architecture. Such a change
needs a new adapter, input processor, decoder interface and training validation.
The current patch does not implement that larger architecture change.

## Local verification

Run `python -m pytest tests/test_scivqa_improvements.py -q` in an environment
with the repository dependencies. Regression checks cover image-border
preservation, legacy transforms, processor resolution, padded-token isolation,
finite OT gradients, legacy state loading, numerical spacing and distinct-figure
counterfactuals, equal-answer loss weighting, padding isolation and resume
objective validation. Generation projects only the newest decoder position
into the vocabulary; a regression check verifies parity with full-sequence
projection. A real MiniLM/MobileViTV2 512-pixel forward/backward check was
also performed locally. None of these checks measures retrained answer quality.
