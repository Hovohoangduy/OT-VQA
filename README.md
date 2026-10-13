# OT-VQA

Visual question answering with a frozen MobileViTV2 image encoder, MiniLM question tokens,
selectable multimodal fusion, and autoregressive answer generation.
For a code-level walkthrough with diagrams and an interactive transport example,
open [the architecture guide](docs/optimal_transport_architecture.html).

## Setup

```bash
pip install -r requirements.txt
```

The default paths expect GQA CSV files under `data/gqa_dataset` and images under
`data/gqa_dataset/images`. Run `python train.py --help` to see path overrides.
The default encoders are `apple/mobilevitv2-2.0-imagenet1k-256` for images and
`sentence-transformers/all-MiniLM-L12-v2` for text. Images are resized to a
288-pixel shortest edge, then center cropped to 256×256;
the image encoder supplies spatial features, and MiniLM supplies token features
plus a mean-pooled question vector. Existing checkpoints retain their saved
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

[`configs/config.py`](configs/config.py) supplies the split/image paths,
CUDA device, and output directory for the Kaggle SciVQA dataset. Edit
`Config.SCIVQA_DATA_ROOT` to change the dataset folder; checkpoints and logs
default to `/kaggle/working/scivqa_model_tuned`. For a local run, override the
CSV/image paths, `--model_path`, and `--device` to match your machine.
The preset uses
OT fusion, MiniLM/MobileViTV2 backbones, batch size 16 per GPU, up to 60 epochs,
AdamW learning rates `1e-4` for the new decoder/fusion layers and `1e-5` for
the pretrained text encoder, 5% warmup, weight decay `0.01`, label smoothing
`0.05`, dropout `0.1`, gradient clipping at `1.0`, and patience 10. Training
answer metrics use a fixed 512-example subset; image loading uses two workers
per GPU. These are starting tuning choices, not a measured score improvement.
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
your dataset revision. The preset uses the existing image preprocessing and
256-pixel vision backbone; it does not reproduce the paper's high-resolution VLMs.

`--dataset scivqa` and `--dataset plantexpert` select built-in Python defaults
from `Config.DATASET_DEFAULTS` for both training and evaluation. Each selects
its own paths, token limits, and evaluation metrics. Other training settings
use the shared defaults. Explicit CLI flags override these settings; for example,
add `--device cuda --batch_size 2` to reduce GPU memory. Edit `configs/config.py`
to change the dataset defaults.
Training records the resolved arguments and all nine SciVQA metrics in
`run_config.json` and `metrics.jsonl` under `/kaggle/working/scivqa_model_tuned`.

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
With two GPUs, batch size 16 means 32 examples per optimizer update.
`--lr` trains the new fusion, projections, decoder, and vocabulary head;
`--encoder_lr` trains the pretrained question encoder and any unfrozen answer
embeddings. The image encoder remains frozen. Warmup takes 5% of the planned
optimization steps, followed by linear decay. The smaller batch provides twice
as many updates per epoch as batch size 32 on the same two GPUs.

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

Compare fresh runs with decoder learning rates `5e-5`, `1e-4`, and `2e-4`, keeping
the encoder LR and remaining settings fixed. Choose on validation, repeat the
winner with multiple seeds, then evaluate the held-out test split once. A longer
200-epoch limit alone does not guarantee better answers. The current frozen
256-pixel image encoder and center crop may limit reading small chart labels or
content near figure edges; learning-rate tuning cannot restore missing pixels.

For resuming, keep the schedule settings and total epoch target from the
original run; for older runs without warmup, pass `--warmup_ratio 0`. Optimizer
state restores its saved learning rates. A checkpoint
with separate encoder/decoder groups recreates those groups automatically; an
older single-LR checkpoint needs a fresh run to use `--encoder_lr`. The stored
selection metric is inherited on resume, and switching it is rejected.

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

Choose a fusion method with `--fusion ot|san|ban|cross_attention|qformer`.
The default is `ot`. SAN uses stacked question-guided image attention;
BAN uses low-rank bilinear token-patch attention; cross attention uses
question-to-image multihead attention; and Q-Former uses learned query tokens
that interact with question tokens before attending to image patches. These
are compact baselines in this shared VQA architecture, not reproductions of
every component in the original SAN, BAN, or BLIP-2 systems. All methods use
the same MobileViTV2, MiniLM, decoder, training loss, data splits, and answer metrics.
The fusion method and its settings are saved in each checkpoint. `test.py`
loads them automatically; an optional `--fusion` there checks that the
checkpoint matches the requested method. `--resume` also checks the method.

`--fusion_glimpses` sets SAN/BAN attention steps (default 2).
`--fusion_queries` and `--fusion_layers` set Q-Former learned queries (8)
and blocks (2). OT options apply only to `--fusion ot`.

To train and evaluate all five methods on the same GQA splits and seed:

```bash
python -m scripts.compare_fusions --output_dir results/fusion_comparison \
  --device cuda --epochs 10 --batch_size 4 --seeds 1105 1106 1107
```

The runner trains each method into its own directory, selects `best.pt` by
lowest validation loss, evaluates the held-out test split, and writes
`comparison.csv`, `summary.csv`, and `comparison.md`. It also saves per-run predictions,
reports, manifests, and checkpoints. Use `--split dev` for a development-only
comparison. Training flags not defined by the runner (such as `--d_model`,
`--lr`, `--fusion_glimpses`, and `--text_model`) are passed to `train.py`.
Completed checkpoints and reports are reused when rerunning the command;
`--force` retrains and reevaluates the selected runs. Ensure identical flags
and data when reusing an output directory. Throughput measurements depend on
the machine and are not model-quality scores. The repository does not include
trained checkpoints for these five methods, so the table is generated after
the comparison run; the older CSVs in `results/` are separate experiments.

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

### Zero-shot native LLM/VLM evaluation

Use `scripts/evaluate_llm.py` for all pretrained vision-language models. It loads
native Hugging Face Transformers checkpoints directly with the model's own
processor. One evaluator handles GQA and PlantExpertVQA, with no server or GGUF
backend. The old GGUF/LM Studio runners, Keye adapter, notebook generator, and
bundled GGUF notebook have been removed.

Install or update evaluation dependencies:

```bash
pip install -U -r requirements.txt
python3 -m scripts.evaluate_llm --list-models
```

Change only `--model` to select a preset alias or its native repository ID:

| Alias | Native checkpoint |
| --- | --- |
| `qwen3-vl-8b` (default) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| `internvl3.5-8b` | [OpenGVLab/InternVL3_5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) |
| `ministral-3-8b` | [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) |
| `glm-4.6v-flash` | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) |
| `gemma-4-e4b` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |

Use InternVL's `-HF` checkpoint for the native Transformers architecture. GLM's
9B Flash model uses the repository ID above. Gemma requires accepting Google's
model license on Hugging Face and providing `HF_TOKEN` in your environment.
Native checkpoint folders are also accepted through `--model` (or `--model-dir`).
GGUF files, GGUF-only repositories, and `:Q4_K_M` suffixes are rejected. When a
folder contains both formats, only native weights are loaded.

Run a five-question GQA smoke evaluation, then evaluate the full split in a
separate output folder:

```bash
python3 -m scripts.evaluate_llm --model Qwen/Qwen3-VL-8B-Instruct \
  --limit 5 --output-dir results/qwen_smoke
python3 -m scripts.evaluate_llm --model qwen3-vl-8b
python3 -m scripts.evaluate_llm --model qwen3-vl-8b --resume
```

Defaults use `data/gqa_dataset/test.csv` and `data/gqa_dataset/images`.
For PlantExpertVQA:

```bash
python3 -m scripts.evaluate_llm --model internvl3.5-8b --dataset plantexpert
```

For Kaggle, upload [evaluate_llm_kaggle.ipynb](notebooks/evaluate_llm_kaggle.ipynb),
select **Settings → Accelerator → GPU T4 x2**, enable **Internet**, and attach your
dataset. Edit `MODEL`, `DATASET`, `DATA_ROOT`, `QUESTIONS`, and `IMAGES` in the
configuration cell, then **Run All**. Defaults use Qwen3-VL-8B-Instruct and the
PlantExpert Kaggle path below. The notebook bundles the evaluator and metrics;
no repository checkout is needed. It installs evaluation dependencies, checks
exactly two visible GPUs, runs five smoke questions, then evaluates the full
split. Model weights are balanced across both GPUs, and each run executes in a
subprocess that releases memory when it exits. Separate smoke/full output folders
live under `/kaggle/working/results`, with automatic resume for matching settings
and reuse of the saved model commit. Restore saved results to the same path after
starting a new Kaggle session. Set `RUN_SMOKE=False` to resume only the full run.
For Gemma, add an `HF_TOKEN` Kaggle secret and enable `USE_KAGGLE_HF_TOKEN` after
accepting the model license. Download the three result files from Kaggle's Output
tab after saving a version. This is the only evaluation notebook; no separate
notebook generator is retained.

If you already have a repository checkout in Kaggle, the equivalent direct
command for the attached PlantExpert dataset is:

```bash
!python -u -m scripts.evaluate_llm \
  --model Qwen/Qwen3-VL-8B-Instruct --dataset plantexpert \
  --questions /kaggle/input/datasets/duyho0511chill/plantexpert-dataset/plantexpert_dataset/test.csv \
  --images /kaggle/input/datasets/duyho0511chill/plantexpert-dataset/plantexpert_dataset/images \
  --output-dir /kaggle/working/results/qwen3_vl_8b_plantexpert
```

The evaluator uses every visible CUDA GPU, with Accelerate balancing weights
and a default 12 GiB weight budget per GPU. To select two GPUs on a larger
machine, set `CUDA_VISIBLE_DEVICES=0,1`. One sufficiently large CUDA GPU also
works. Auto dtype selects FP16 on T4 and BF16 on Ampere or newer; override with
`--dtype`. Attention defaults to PyTorch SDPA; `--attention-implementation eager`
is available. The official Ministral checkpoint stores FP8 weights; the runner
uses Transformers' `FineGrainedFP8Config(dequantize=True)` to load them in the
selected floating-point dtype, including on T4. This increases weight memory.
Weights must fit the selected GPUs without CPU/disk offload. Adjust
`--gpu-memory-gib` for your hardware; images and generation need additional
memory. The default prompt-plus-output limit is 8192 tokens (`--context-size`).
Run a smoke evaluation on your GPUs before a full evaluation.

Each example uses only the image, question, and fixed dataset instruction.
Reference answers, dataset annotations, demonstrations, and chat history never
enter the model prompt. Generation is greedy and requests thinking disabled
through the model's chat template. Images use native processor preprocessing.
GQA defaults to 64 output tokens; PlantExpert defaults to 256 (`--max-tokens`).
Questions accept CSV (`image,question,answer`, with optional `question_id` or
`anno_id`) or official GQA question JSON. Unlabeled examples produce predictions
without scores. Use `--image-split` for images in a split subfolder.

Only three result files are written per run:

- `predictions.jsonl`: append-only per-question predictions, raw responses,
  references, errors, latency, and lexical scores. On retries, the latest row for
  a question ID takes precedence.
- `report.json`: aggregate scores and completion status, including partial runs.
- `run_config.json`: model identity, pinned revision (or local file hashes),
  processor/model loading settings, prompt, dataset identity, and scoring options
  checked by `--resume`.

Default result folders are `results/<model>_<dataset>_zero_shot`. There are no
CSV/JSON duplicate prediction exports, server logs, or ZIPs.
`--resume` skips successful examples and retries failures; use identical flags
and model files. For reproducible comparisons, use `--hf-revision <commit>`.
A changed model revision, dataset, prompt, or loading/scoring configuration is
rejected on resume. `--continue-on-error` records errors and continues; failed
labeled examples count as wrong, while unattempted questions are excluded.

GQA reports exact-match `accuracy` and case-insensitive `normalized_accuracy`.
PlantExpert reports `em`, `token_f1`, `bleu_1`, `bleu_2`, `rouge_l`, and
`bertscore_f1`, using the shared repository metrics. Formatting cleanup removes
Markdown and repeated/escaped whitespace from predictions and references while
preserving case, punctuation, and answer content. It does not extract answers
from explanatory prose. Scores are fractions; rescaled BERTScore can be negative.
BERTScore defaults to `bert-base-uncased` on CPU, batch size 16, with English
baseline rescaling. Use `--no-bertscore` for lexical-only PlantExpert scoring,
or configure `--bertscore-model`, `--bertscore-device`, `--bertscore-batch-size`,
and `--no-bertscore-rescale`. The report records the scorer's model hash.

### Evaluate a trained repository model

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
`train.py`, `test.py`, and `python -m scripts.compare_fusions`.

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
its lowercase alphanumeric tokenizer and stemming disabled. SciVQA defaults
to `roberta-large` BERTScore without baseline rescaling; change these choices
with `--bertscore_model` and `--bertscore_rescale`. The paper does not specify
its encoder or tokenizer, so the implementation records these choices and the
BERTScore model hash rather than claiming identical published scores.
ROUGE follows the library's zero score for empty token sequences. BERTScore
assigns 1 when both answers are empty and 0 when only one is empty, as an
explicit project convention. Answers retain their original multiple-choice
labels, numbers, and abstention text; scoring does not infer abstention from
gold annotations or rewrite model responses to reproduce the paper's
post-processing. Prediction exports retain `qa_pair_type` and `figure_type`.
Training logs use `train_<metric>` and `val_<metric>`; fusion comparisons and
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

## Tests

```bash
python -m unittest discover -s tests
```
