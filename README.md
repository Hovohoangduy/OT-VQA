# OT-VQA

Visual question answering with a frozen MobileViTV2 image encoder, MiniLM question tokens,
question-conditioned partial optimal transport fusion, and autoregressive answer generation.
For a code-level walkthrough with diagrams and an interactive transport example,
open [the architecture guide](docs/optimal_transport_architecture.html).

## Setup

```bash
pip install -r requirements.txt
```

The default paths expect GQA CSV files under `data/gqa_dataset` and images under
`data/gqa_dataset/images`. Run `python train.py --help` to see path overrides.
The default encoders are `apple/mobilevitv2-2.0-imagenet1k-256` for images and
`sentence-transformers/all-MiniLM-L12-v2` for text. The legacy pipeline resizes
to a 288-pixel shortest edge, then center crops to 256×256. New SciVQA runs
preserve the complete figure with aspect-ratio-preserving padding at 512×512;
the image encoder supplies spatial features, and MiniLM supplies token features
plus a mean-pooled question vector. Existing OT checkpoints retain their saved
encoder names and can still be loaded.

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

For a fresh run with the figure caption and figure type supplied alongside the
question, add `--figure_context --max_question_tokens 512` and use a new
`--model_path`. The setting is stored in the checkpoint and reused for
validation, test evaluation, and diagnostics. The [SciVQA experiment guide](SCIVQA_TUNING.md)
has a two-GPU Kaggle command and a comparison against the cited paper.

[`configs/config.py`](configs/config.py) supplies local dataset paths and
training defaults. For Kaggle, override the CSV/image paths and `--model_path`
with your input dataset and a directory under `/kaggle/working`.
The preset uses
OT fusion, MiniLM/MobileViTV2 backbones, batch size 8 per GPU, up to 30 epochs,
and freezes both the image and question encoders. AdamW trains the new
decoder/fusion layers at `1e-5`, with 5% warmup, weight decay `0.05`, label smoothing
`0.1`, dropout `0.2`, gradient clipping at `1.0`, and patience 8. Training
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

The preset allows 256 question tokens and 64 answer tokens, including special
tokens. The downloaded splits reach 210/37 tokens with the default MiniLM
tokenizer, preserving the choice context and unanswerable targets. If exporting
with `--include-caption`, use `--max_question_tokens 512` and check lengths for
your dataset revision. New SciVQA runs use full-figure padding at 512 pixels
with the frozen MobileViTV2 weights; this does not reproduce the paper's VLMs.

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

For a SciVQA optimization experiment on two Kaggle GPUs, start a fresh run with:

```bash
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --dataset scivqa
```

Use the updated repository for these flags. Remove the leading `!` outside a
notebook. These values are a starting experiment, not a measured improvement.
With two GPUs, batch size 8 means 16 examples per optimizer update.
`--lr` trains the new fusion, projections, decoder, and vocabulary head;
`--encoder_lr` trains the pretrained question encoder and any unfrozen answer
embeddings. The image encoder remains frozen. Warmup takes 5% of the planned
optimization steps, followed by linear decay. Batch size is per GPU.

For an independent OT visual-memory experiment, add `--ot_visual_memory` and
use a fresh output directory. The SciVQA preset already uses
`--image_preprocessing pad --image_size 512`. Padding preserves figure edges
and aspect ratio; MobileViTV2
receives the full 512-pixel tensor without its processor recropping to 256.
The decoder receives the original OT-grounded question tokens plus spatial
image tokens conditioned using the reverse direction of the same OT plan,
with a visual residual path and learned 2D position projection. This increases
visual token count from 64 to 256. The image encoder remains frozen. Try 384
pixels or reduce batch size if memory is tight. The default retains
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
200-epoch limit alone does not guarantee better answers. The current frozen
256-pixel image encoder and center crop may limit reading small chart labels or
content near figure edges; learning-rate tuning cannot restore missing pixels.
The new image options address cropping and resolution; they do not add OCR or
the pretrained numerical reasoning capabilities of a large VLM.

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

For resuming, keep the schedule settings and total epoch target from the
original run; for older runs without warmup, pass `--warmup_ratio 0`. Optimizer
state restores its saved learning rates. A checkpoint
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

Training uses OT fusion with the MobileViTV2 image encoder, MiniLM question
encoder, and autoregressive answer decoder. OT settings are saved in each
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
