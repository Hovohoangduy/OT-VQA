# OT-VQA

Visual question answering with a frozen SigLIP vision encoder, MiniLM question tokens,
question-conditioned partial optimal transport fusion, and autoregressive answer generation. The
image encoder remains configurable, and checkpoints retain their encoder configuration.
For a code-level walkthrough with diagrams and an interactive transport example,
open [the architecture guide](docs/optimal_transport_architecture.html).

## Setup

```bash
pip install -r requirements.txt
```

The default paths expect GQA CSV files under `data/gqa_dataset` and images under
`data/gqa_dataset/images`. Run `python train.py --help` to see path overrides.
The current default image encoder is `google/siglip-base-patch16-384`; the text
encoder is `sentence-transformers/all-MiniLM-L12-v2`. Image size and preprocessing
are dataset- and checkpoint-controlled. The current SciVQA preset pads the full
figure to 384×384, while older checkpoints may use 256-pixel center-crop
preprocessing. The image encoder supplies spatial features, and MiniLM supplies
token features plus a mean-pooled question vector. Existing checkpoints retain
their saved encoder names and preprocessing settings.

To download random GQA subsets of 10,000 training, 1,000 validation, and 2,000
test images, with one question–answer pair per image:

```bash
python -m utils.download_gqa
```

Sampling uses `--seed 42` by default; change the seed for a different subset.
Override sizes with `--train-images`, `--val-images`, and `--test-images`.
Validation and test are disjoint random subsets of the labeled GQA validation split.

Keep the held-out test split for the final comparison. Dataset downloads require
network access and can be large; choose counts that fit your compute budget.
The downloader spaces dataset viewer requests, honors HTTP 429 retry delays, and
saves metadata pages in `<output>/.row_cache`. Rerun with the same `--output` to
reuse completed pages. By default, metadata uses one worker and starts requests
at least one second apart. HTTP 429 responses pause all metadata workers for
the server's requested cooldown before retrying. If rate limiting persists, set `HF_TOKEN` in your shell
and try `--metadata-workers 1 --request-interval 2`. `--workers` controls image
downloads separately.

To download random PlantExpertVQA subsets of 10,000 training, 1,000 validation,
and 2,000 test question–answer pairs, with their images:

```bash
python -m utils.download_plantexpert
```

The output is `data/plantexpert_dataset/train.csv`, `val.csv`, `test.csv`, and `images/`.
The script streams each source CSV once and samples reproducible individual
question–answer rows (`--seed 42` by default). It does
not use the rate-limited dataset viewer API. The image downloader reads only
the required files from the source ZIP archives. You can change the counts
with `--train-pairs`, `--val-pairs`, and `--test-pairs`. Selected rows are cached in
`<output>/.selection_cache`, and rerunning with the same output reuses them
and any downloaded images. Set `HF_TOKEN` in your shell if you have a Hugging
Face token.
For training, use `python train.py --dataset plantexpert`. This automatically
selects the Python defaults in `configs/config.py`, including dataset paths and answer metrics.

To download and format [SciVQA](https://huggingface.co/datasets/katebor/SciVQA):

```bash
python -m utils.download_scivqa
```

This preserves all official labeled splits and writes `train.csv`, `val.csv`,
`test.csv`, `images/`, and `metadata.json` under `data/scivqa_dataset`.
The CSVs use `anno_id`, `image`, `question`, and `answer` for this project's
loader, and retain the original questions, answer options, captions, and figure
metadata. Multiple-choice questions include labeled options; answers retain
their original labels (including multiple labels). Add `--include-caption` to
include captions in the model input; captions are always retained in the CSV.

Optional `--train-pairs 10000 --val-pairs 1000 --test-pairs 2000` selects
reproducible subsets with `--seed 42`; each count defaults to `0` (all pairs).
The script downloads each complete source image ZIP even for subsets. Source
files are cached under `<output>/.download_cache`, and reruns reuse downloaded
files and verified images. Use `--output` to change the destination or
`--revision` to choose a Hugging Face commit. `HF_TOKEN` is supported.

Train the OT model with the SciVQA preset:

```bash
python train.py --dataset scivqa
```

The SciVQA preset includes figure type and caption context and allows up to 512
question tokens. To compare without figure context, use `--no-figure_context`
and a fresh `--model_path`. The setting is stored in the checkpoint and reused
for validation, test evaluation, and diagnostics. The [SciVQA experiment guide](SCIVQA_TUNING.md)
has a two-GPU Kaggle command and a comparison against the cited paper.

[`configs/config.py`](configs/config.py) supplies local dataset paths and
training defaults. For Kaggle, override the CSV/image paths and `--model_path`
with your input dataset and a directory under `/kaggle/working`.
The preset uses
OT fusion, MiniLM/SigLIP backbones, batch size 8 per GPU, up to 30 epochs,
and freezes both the image and question encoders. AdamW trains the new
decoder/fusion layers at `1e-5`, with 5% warmup, weight decay `0.05`, label smoothing
`0.1`, dropout `0.25`, gradient clipping at `1.0`, and patience 8. Training
answer metrics use a fixed 512-example subset; image loading uses two workers
per GPU. Use `--no-freeze_question_encoder` for a separate fine-tuning control;
that run uses the preset encoder LR of `2e-6`. These are starting tuning choices,
not a measured score improvement.
The reference
[SciVQA paper](https://arxiv.org/html/2507.06183v1) primarily evaluates prompting
and ensembles; Appendix D mentions LoRA on 5,000 Bespoke MiniChart examples
but provides no full OT training recipe. This preset trains on the complete
15,120/1,680 train/validation splits and reserves the 4,200 test examples for
final evaluation. SciVQA checkpoints maximize validation ROUGE-L F1 by default;
pass `--selection_metric loss` to select by validation loss instead.

The preset allows 512 question tokens and 64 answer tokens, including special
tokens. The downloaded splits reach 210/37 tokens with the default MiniLM
tokenizer, preserving the choice context and unanswerable targets. If exporting
with `--include-caption`, use `--max_question_tokens 512` and check lengths for
your dataset revision. New SciVQA runs use full-figure padding at 384 pixels with the frozen SigLIP
vision encoder; this does not reproduce the paper's VLMs.

`--dataset scivqa` and `--dataset plantexpert` select built-in Python defaults
from `Config.DATASET_DEFAULTS` for both training and evaluation. Each selects
its own paths, token limits, and evaluation metrics. Other training settings
use shared defaults, except for SciVQA's training overrides described above.
Explicit CLI flags override these settings; for example,
add `--device cuda --batch_size 2` to reduce GPU memory. Edit `configs/config.py`
to change the dataset defaults.
Training records the resolved arguments and all nine SciVQA metrics in
`run_config.json` and `metrics.jsonl` under `--model_path`.
The resolved model configuration records image size and preprocessing. Each
validation history row also includes `val_metrics_by_qa_pair_type` so numerical
answers and abstention can be inspected separately.

Evaluate the best checkpoint on validation or test:

```bash
python test.py --dataset scivqa --split dev \
  --report_json results/scivqa/dev_report.json \
  --predictions_csv results/scivqa/dev_predictions.csv

python test.py --dataset scivqa --split test \
  --report_json results/scivqa/test_report.json \
  --predictions_csv results/scivqa/test_predictions.csv
```

For **Qwen3-VL-2B + OT + LoRA** on Kaggle, use the direct two-process command
below. Install the shared dependencies in your Kaggle cell first:

```python
%pip install -r requirements.txt
```

`requirements.txt` contains the runtime, evaluation, OT/LoRA and test dependencies
with the reviewed versions pinned. NF4's bitsandbytes dependency installs on
Linux/Windows; macOS CPU/MPS uses the same file without bitsandbytes.

The [Kaggle runbook](.agents/plans/scivqa-kaggle-final-review.md) includes the matching
pilot, full training and resume commands. Run from the updated repository and
keep a fresh output directory.

```bash
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset scivqa --device cuda --fusion ot --model_family qwen3_vl \
  --vlm_model Qwen/Qwen3-VL-2B-Instruct \
  --vlm_revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --text_model sentence-transformers/all-MiniLM-L12-v2 \
  --query_revision a50ef00143b4d5391434df20ae11632588ac25be \
  --quantize_4bit --amp fp16 --amp_init_scale 1024 --gradient_checkpointing \
  --epochs 3 --batch_size 1 --gradient_accumulation_steps 8 --num_workers 2 \
  --lora_rank 16 --lora_lr 2e-5 --lr 1e-4 --warmup_ratio 0.05 \
  --weight_decay 0.05 --label_smoothing 0 --loss_normalization example \
  --gradient_clip 1.0 --early_stopping_patience 2 \
  --ot_num_heads 1 --cost_dim 128 --ot_loss_weight 0 \
  --ot_epsilon 0.05 --ot_iterations 20 --ot_dustbin_mass 0.2 --ot_dustbin_cost 1.0 \
  --adapter_control ot --plan_control learned \
  --figure_context --prompt_version scivqa-v2 \
  --query_tokens 256 --max_answer_tokens 64 --total_sequence_tokens 2048 \
  --visual_tokens 256 --selection_metric rouge_l_f1 \
  --train_eval_samples 0 --no-compute_bertscore --log_every_updates 25 --seed 1105 \
  --train_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/train.csv \
  --dev_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/val.csv \
  --img_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/images \
  --model_path /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105
```

Batch 1/GPU × accumulation 8 × 2 GPUs gives **effective batch 16**.
Before the full run, execute the same command with `--max_updates 100
--profile_stages` and change `--model_path` to
`/kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105_pilot`. Keep `--epochs 3`
and the other settings unchanged. Then run:

```bash
!python scripts/check_t4_pilot.py \
  --run /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105_pilot --lora_rank 16
```

The direct `torchrun` command does not enforce the launcher gate automatically;
inspect the pilot report, gradient checks and worst shapes before the full run.
`scripts/kaggle_train.py` remains available as an optional validated launcher.
These commands start a quality experiment; they do not establish an accuracy gain.

The MiniLM/SigLIP settings described above and below apply to the **legacy**
model family. Its batch size 8/GPU gives effective batch 16 with accumulation 1.
`--lr` trains its fusion/decoder; `--encoder_lr` applies to unfrozen text/answer
embeddings. The native path uses `--lr 1e-4` for OT and `--lora_lr 2e-5` for LoRA,
with both vision and MiniLM frozen. Native checkpoints store adapters and
training state with references to pinned base weights.

For an independent OT visual-memory experiment, add `--ot_visual_memory` and
use a fresh output directory. The SciVQA preset already uses
`--image_preprocessing pad --image_size 384`. Padding preserves figure edges
and aspect ratio; SigLIP receives the prepared 384-pixel tensor without its
processor recropping it to a smaller default size.
The decoder receives the original OT-grounded question tokens plus spatial
image tokens conditioned using the reverse direction of the same OT plan,
with a visual residual path and learned 2D position projection. With the SciVQA SigLIP preset, this adds 576 spatial image tokens (a 24×24
patch grid). The image encoder remains frozen. Reduce batch size or image
resolution if memory is tight. The default retains
question-only OT memory. Use explicit `--image_size 256 --image_preprocessing crop`
only for a comparison with the legacy preprocessing. Cropping is not a validated
SciVQA improvement; a training-figure geometry audit found that the legacy crop
retains only 50.1% of image area at the median, and 4.6% for the widest figure.
Those numbers describe area, not useful information or accuracy.

See [the SciVQA experiment guide](SCIVQA_TUNING.md) for the diagnosis,
copyable Kaggle commands, and a validation-only ablation sequence.

`--selection_metric rouge_l_f1` maximizes full-validation ROUGE-L F1 for both
`best.pt` and early stopping. Use `bertscore_f1` if that is your primary objective.
SciVQA defaults to ROUGE-L F1; other datasets minimize validation loss.
`--train_eval_samples 512` scores a fixed, seeded random training subset each
epoch; it does not reduce the examples
used for optimization or validation. Use `0` to skip training-answer scoring, or
`-1` to score the full training set (the default for other datasets). Subset training metrics should only
be compared across runs with the same seed and sample count. `--num_workers`
controls image-loading processes per GPU. BERTScore remains on CPU by default;
`--bertscore_device cuda --bertscore_batch_size 4` can speed it up if the primary
GPU has enough memory for both models.

Compare fresh runs with decoder learning rates `5e-6`, `1e-5`, and `3e-5`, keeping
the encoder LR and remaining settings fixed. Choose on validation, repeat the
winner with multiple seeds, then evaluate the held-out test split once. A longer
200-epoch limit alone does not guarantee better answers. Legacy 256-pixel center-crop checkpoints may limit reading small chart labels or
content near figure edges; learning-rate tuning cannot restore missing pixels.
The current 384-pixel padded SciVQA preset preserves figure geometry. It does
not add OCR or the pretrained numerical reasoning capabilities of a large VLM.

Fresh SciVQA runs default to `--loss_normalization example`: average token CE
within each answer, then average across answers. In the local SciVQA training
data, unanswerable questions make up 14.3% of examples but 48.1%
of non-padding targets under legacy token averaging. Equal-answer averaging
reduces this length imbalance without changing targets or evaluation metrics.
Use `--loss_normalization token` in a separate fresh run as the legacy control.
Other datasets keep token averaging. Validation loss always uses token averaging.
Logs and checkpoints record the training objective, and resume inherits it
unless an explicit conflicting override is supplied. The effect on validation
quality has not yet been measured.

For resuming, omit `--epochs` to inherit the saved total epoch target and schedule.
To continue a run originally targeting 3 epochs up to a total of 20, pass
`--resume path/to/last.pt --epochs 20`. If the checkpoint completed epoch 3,
training starts at epoch 4 and can run 17 more epochs. Decreasing the saved
target is rejected; all other recipe checks still apply. Extending the target
retains model weights, optimizer moments, AMP scaler, RNG, best metrics and
early-stopping history. Early stopping can therefore end training before epoch 20.
The linear learning-rate schedule is recalculated for the new total steps,
including warmup at the saved ratio, and applied at the saved optimizer-update
index before the first resumed update. Learning rates can increase, including
from zero after the old schedule ended. This continuation changes the schedule
used by future updates; it is not equivalent to training 20 epochs from scratch.
New checkpoints save the extended target and resume it normally. `best.pt`
continues from its selected epoch; use `last.pt` for the latest saved epoch.
The Kaggle launcher also accepts `--stage resume --resume path/to/last.pt --epochs 20`.

For older runs without warmup, pass `--warmup_ratio 0`. Without an epoch extension,
optimizer state restores its saved learning rates. A checkpoint
with separate encoder/decoder groups recreates those groups automatically; an
older single-LR checkpoint needs a fresh run to use `--encoder_lr`. The stored
selection metric is inherited on resume, and switching it is rejected.
Image preprocessing, resolution, and visual-memory settings are also inherited
from the checkpoint; conflicting overrides require a fresh run. Old checkpoints
without these fields retain 256-pixel crop preprocessing and question-only OT
memory. Evaluation, single-image prediction, and diagnostics use the checkpoint's
image transform automatically.

To train the OT model on this PlantExpertVQA subset:

```bash
python train.py --dataset plantexpert --device auto --batch_size 2
```

The run uses the training default of 10 epochs. Across the local 10,000 train,
1,000 validation, and 2,000 test rows, the longest MiniLM-tokenized question
is 32 tokens including special tokens. The longest answer is 112 tokens
including start and end tokens. The defaults cover all 13,000 rows without
truncation. `--max_question_tokens` and `--max_answer_tokens` can override
these limits for other datasets; both limits are saved with each checkpoint.
It saves
`best.pt`, `last.pt`, and `metrics.jsonl` in `data/plantexpert_model`. Use
`--resume data/plantexpert_model/last.pt` to continue an interrupted run, with
`--epochs` set to the desired total epoch count. `--device auto` chooses CUDA,
then Apple MPS, then CPU. To choose explicitly, replace it with `--device cpu`
or `--device cuda` (NVIDIA GPU), or `--device mps` (Apple GPU). The current
`ot-vqa` environment reports CPU only.

After training, evaluate the PlantExpert validation split with:

```bash
python test.py --dataset plantexpert --split dev \
  --predictions_csv data/plantexpert_model/val_predictions.csv \
  --report_json data/plantexpert_model/val_report.json
```

## Train

Training uses OT fusion with the configured image encoder (SigLIP for the current
SciVQA preset), MiniLM question encoder, and autoregressive answer decoder. OT settings are saved in each
checkpoint and restored for evaluation and resumed training. `--fusion ot`
remains accepted for existing commands; OT is the only supported method.

For NVIDIA GPU training on Windows, install a CUDA-enabled `torch` and matching
`torchvision` in your active environment using the
[official PyTorch installer](https://pytorch.org/get-started/locally/). Installing
`requirements.txt` alone does not guarantee a CUDA build. Check your installation
in PowerShell:

```powershell
conda activate ot
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must print `True`. Then train with CUDA explicitly:

```powershell
python train.py --device cuda --train_csv_path data/gqa_dataset/train.csv --dev_csv_path data/gqa_dataset/val.csv --img_path data/gqa_dataset/images --model_path data/gqa_model
```

Training prints `Training on device: cuda`. If CUDA is unavailable, `--device cuda`
stops with an error instead of training on CPU. Reduce `--batch_size` from its
default of 4 if the GPU runs out of memory. `--device auto` also uses CUDA when
available, but can fall back to another device.

Training writes `last.pt`, the best checkpoint under the validation selection
metric as `best.pt` (ROUGE-L F1 for SciVQA, loss for other datasets), a JSONL
metric history, an evaluation plot, and `run_config.json` with dataset hashes.
Add `--save_every_epoch` to retain `epoch_0001.pt`, `epoch_0002.pt`, and so on.
Each epoch file is a full resumable checkpoint, so 100 files can use substantial disk space.
After each epoch, GQA runs report generated-answer binary exact-match `accuracy` for both
the training and validation splits. SciVQA reports ROUGE-1/L and BERTScore
precision, recall, and F1. PlantExpertVQA and unrecognized dataset paths report EM, token F1,
BLEU-1/2, ROUGE-L, and BERTScore. The history stores
these as `train_*` and `val_*` fields; the plot shows both curves. Scoring the
full training split adds an evaluation pass each epoch.
Resume with `--resume path/to/last.pt`.

For a Kaggle notebook with two enabled GPUs, run:

```bash
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --device cuda --epochs 100 --batch_size 32 \
  --early_stopping_patience 0 \
  --train_csv_path /kaggle/input/datasets/duyho0511chill/plantexpert-dataset/plantexpert_dataset/train.csv \
  --dev_csv_path /kaggle/input/datasets/duyho0511chill/plantexpert-dataset/plantexpert_dataset/val.csv \
  --img_path /kaggle/input/datasets/duyho0511chill/plantexpert-dataset/plantexpert_dataset/images \
  --model_path /kaggle/working/plantexpert_model
```

`--batch_size 32` means 32 examples per GPU (64 total). Checkpoints and
validation metrics are written once by rank 0; validation runs on one GPU.
Use `/kaggle/working` for output because Kaggle input datasets are read-only.
Resume with the same command plus `--resume /kaggle/working/plantexpert_model/last.pt`.
This command updates `last.pt` after every epoch without accumulating 100 large
files. Add `--save_every_epoch` only if you need a separate file for each epoch.
The default early-stopping patience is 8 epochs; this command disables it to
run all 100 epochs.
Set `--ot_dustbin_mass 0` for a full-OT ablation. Partial OT defaults to mass
`0.2`, Sinkhorn epsilon `0.05`,
20 iterations, and dustbin-to-dustbin cost `1.0`. These values require
validation on your dataset.

## Evaluate

### Evaluate a trained OT model

```bash
python test.py --checkpoint data/gqa_model/best.pt --split dev \
  --predictions_csv data/gqa_model/dev_predictions.csv \
  --report_json data/gqa_model/dev_report.json
```

To evaluate `data/weights/OT_fusion_0410.pt` on the 2,000-example PlantExpertVQA
test split, run from the repository root:

```bash
conda activate ot-vqa

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u test.py \
  --checkpoint data/weights/OT_fusion_0410.pt \
  --split test \
  --test_csv_path data/plantexpert_dataset/test.csv \
  --img_path data/plantexpert_dataset/images \
  --device auto \
  --batch_size 4 \
  --predictions_csv results/OT_fusion_0410_test_predictions.csv \
  --report_json results/OT_fusion_0410_test_report.json
```

Use the dataset the checkpoint was trained on. If it was trained on GQA,
replace both `plantexpert_dataset` paths with `gqa_dataset`.
The command prints loss and the dataset's answer metrics described below, plus
throughput and latency. It saves per-example predictions to CSV and an
evaluation report to JSON, creating the output directory automatically.

Metric selection defaults to `--dataset auto`: a CSV path containing a `gqa`
or `scivqa` component (such as `data/scivqa_dataset/val.csv`) selects that
dataset's scoring. Use `--dataset gqa` or `--dataset scivqa` for custom paths,
or `--dataset plantexpert` to select the six PlantExpertVQA metrics explicitly.
The same option works for
`train.py` and `test.py`.

GQA uses [binary exact-match accuracy](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html):
each generated answer receives `1` if it matches the single reference answer
and `0` otherwise. The dataset's `accuracy` is the mean of these scores, ranging
from `0` to `1` (multiply by 100 for a percentage). Matching trims leading and
trailing whitespace; case, punctuation, and internal whitespace remain
significant. It does not extract answers from explanations or award partial
credit for synonyms.
Training logs and plots use `train_accuracy` and `val_accuracy`;
evaluation JSON and prediction CSV exports use `accuracy`. GQA runs do not
compute the six PlantExpertVQA metrics or load a BERTScore model. Checkpoint
selection and early stopping still use validation loss.

This replaces the earlier `vqa_accuracy` consensus score that awarded only
`1/3` for a correct single-reference answer. Existing checkpoints can be
evaluated with the new metric without retraining. Old reports and history
entries retain their original scores; rerun evaluation to obtain exact-match
accuracy rather than renaming or multiplying old scores, because matching
rules also changed. When resuming training, newly appended epochs use the
`train_accuracy`/`val_accuracy` fields.

SciVQA uses the nine metrics in Tables 1 and 2 of the
[reference paper](https://arxiv.org/html/2507.06183v1):

| Paper column | JSON/CSV metric |
| --- | --- |
| R1-F1 / R1-P / R1-R | `rouge_1_f1` / `rouge_1_precision` / `rouge_1_recall` |
| RL-F1 / RL-P / RL-R | `rouge_l_f1` / `rouge_l_precision` / `rouge_l_recall` |
| BS-F1 / BS-P / BS-R | `bertscore_f1` / `bertscore_precision` / `bertscore_recall` |

Scores are macro means of per-answer values; F1 is averaged directly rather
than recomputed from average precision/recall. ROUGE uses `rouge-score` with
stemming enabled, matching the [official SciVQA evaluator](https://github.com/esborisova/SciVQA/blob/main/src/eval/scoring.py).
SciVQA defaults
to `roberta-large` BERTScore without baseline rescaling; change these choices
with `--bertscore_model` and `--bertscore_rescale`. The paper does not specify
its BERTScore encoder, so the implementation records this choice and the
BERTScore model hash rather than claiming identical published scores.
ROUGE follows the library's zero score for empty token sequences. BERTScore
assigns 1 when both answers are empty and 0 when only one is empty, as an
explicit project convention. Answers retain their original multiple-choice
labels, numbers, and abstention text; scoring does not infer abstention from
gold annotations or rewrite model responses to reproduce the paper's
post-processing. Prediction exports retain `qa_pair_type` and `figure_type`.
Generated answers repair WordPiece numerical spacing (`0. 5` becomes `0.5`)
using prediction text alone. Digits and answer choices are preserved; reports
record `answer_postprocessing=wordpiece_numeric_spacing_v1`. This fixes
presentation and can affect BERTScore/exact matching; the existing ROUGE
tokenizer already ignores decimal punctuation and spaces. Reevaluate old
checkpoints with the same code when comparing experiments.
Training logs use `train_<metric>` and `val_<metric>`;
history diagnostics accept the same nine-score schema.

For PlantExpertVQA and unrecognized dataset paths, evaluation reports the six PlantExpertVQA paper metrics: case-insensitive exact
match, SQuAD-style token F1, BLEU-1, BLEU-2, ROUGE-L F1, and BERTScore-F1.
Scores are fractions; baseline-rescaled BERTScore can be negative. BLEU uses
unsmoothed per-answer modified precision and a brevity penalty. The default
BERTScore setup is `bert-base-uncased`, English baseline rescaling, and CPU;
change it with `--bertscore_model`, `--no-bertscore_rescale`, or
`--bertscore_device`. The evaluation report records the model hash. The paper
does not define empty-answer scoring; this implementation assigns BERTScore 1
when both answers are empty and 0 when only one is empty. The paper does not
specify every tokenizer and BERTScore setting, so these scores should
not be treated as numerically identical to its published results. The local
5,000/1,000 train/validation subset also differs from the paper's full test
set. Throughput and peak CUDA memory describe VQA generation and loss only.
If your CSV has a `question_type` column, the prediction export
includes it. The paper's results are for image-text retrieval; VQA quality
must be established with this repository's generated-answer evaluation.

## Predict

```bash
python predict.py \
  --checkpoint data/gqa_model/best.pt \
  --image path/to/image.jpg \
  --question "What color is the car?"
```

## Diagnose training

`diagnose_training.py` reports overfitting, output collapse, and sensitivity to
shuffled images or questions.

```bash
python diagnose_training.py \
  --checkpoint data/gqa_model/best.pt \
  --dev_csv_path data/gqa_dataset/val.csv \
  --dev_img_path data/gqa_dataset/images
```

## Codex project guidance

Codex reads repository instructions from [`AGENTS.md`](AGENTS.md) and discovers project skills under [`.agents/skills/`](.agents/skills/). The original `.agent/` directory is retained for the existing Gemini setup. The Codex guidance treats question-conditioned partial Optimal Transport as the core image-question fusion path; claimed VQA gains still require measured validation results and controlled comparisons.

## SciVQA evidence-plan implementation (2026-10-15)

The legacy OT model remains the default. Fresh runs use `scivqa-v2` prompts
(panel **count**, no guessed target panel) and `task-v2` decoding (no numeric
hard mask). Old checkpoints inherit `scivqa-v1` / `task-v1` for reproducibility.
To evaluate a correctness intervention on an old checkpoint, explicitly pass
`--prompt_version scivqa-v2 --decode_version task-v2`; this is an input/decode
intervention, not a retrained accuracy result.

The new `--model_family qwen3_vl` path preserves the native processor, image
placeholders, M-RoPE, KV cache and all DeepStack levels. Query-only frozen
MiniLM representations condition merged visual features through partial OT.
`--adapter_control native|pooled` and `--plan_control independent` are explicit
research controls; OT is the default. Gate zero gives native no-op parity;
random output projections let the gates learn on the first update.

See [the implementation and T4 runbook](.agents/plans/scivqa-implementation-runbook.md)
for the pinned environment, pilot commands, controlled comparisons and remaining
acceptance gates. **No >80% accuracy claim is implied by implementation or smoke
tests.** CUDA/NF4, two-T4 memory and quality must be measured on that hardware.

Run persistent regression tests with `python -m pytest tests -q` after installing
`transformers==5.5.0`, `peft==0.18.1`, `accelerate==1.13.0` and pytest. Tests use
small native Qwen models, CPU DDP, and available device checks; they do not
require downloading 2B weights or evaluating the held-out test split.
