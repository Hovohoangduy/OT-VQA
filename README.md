# OT-VQA

Visual question answering with a frozen MobileViTV2 image encoder, MiniLM question tokens,
selectable multimodal fusion, and autoregressive answer generation.
For a code-level walkthrough with diagrams and an interactive transport example,
open [the architecture guide](docs/ot_vqa_architecture.html).

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
For training, pass `--train_csv_path data/plantexpert_dataset/train.csv`,
`--dev_csv_path data/plantexpert_dataset/val.csv`, and
`--img_path data/plantexpert_dataset/images` to `train.py`.

To train the OT model on this PlantExpertVQA subset:

```bash
python train.py --device auto --batch_size 2 \
  --train_csv_path data/plantexpert_dataset/train.csv \
  --dev_csv_path data/plantexpert_dataset/val.csv \
  --img_path data/plantexpert_dataset/images \
  --model_path data/plantexpert_model
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
python test.py --checkpoint data/plantexpert_model/best.pt --split dev \
  --dev_csv_path data/plantexpert_dataset/val.csv \
  --img_path data/plantexpert_dataset/images \
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

Training writes `last.pt`, the lowest-validation-loss checkpoint as `best.pt`, a JSONL
metric history, an evaluation plot, and `run_config.json` with dataset hashes.
Add `--save_every_epoch` to retain `epoch_0001.pt`, `epoch_0002.pt`, and so on.
Each epoch file is a full resumable checkpoint, so 100 files can use substantial disk space.
After each epoch, GQA runs report generated-answer binary exact-match `accuracy` for both
the training and validation splits. Other datasets report EM, token F1,
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

### Zero-shot Qwen3-VL with LM Studio

Load the **vision** model `qwen3-vl-2b-instruct` in LM Studio and start its local
server on port 1234. This evaluator uses Python 3.9+ and only the standard
library, so it does not require PyTorch or the OpenAI Python package.
LM Studio supports the
[OpenAI-compatible chat endpoint](https://lmstudio.ai/docs/developer/openai-compat/chat-completions).
Run these commands from the repository root:

```bash
# Check the exact API model identifier.
python3 -m scripts.evaluate_gqa_lmstudio --list-models

# First check five examples in a separate output directory.
python3 -m scripts.evaluate_gqa_lmstudio --limit 5 --output-dir results/qwen_smoke

# Evaluate every row in data/gqa_dataset/test.csv (2,000 in the downloaded subset).
python3 -m scripts.evaluate_gqa_lmstudio

# Resume the full run after an interruption, using its original flags.
python3 -m scripts.evaluate_gqa_lmstudio --resume
```

Each question gets a fresh request containing the original image and question,
with a short-answer instruction, temperature 0, and no examples or reference
answers. The server handles image preprocessing. The default URL is
`http://127.0.0.1:1234/v1`; override it with `--base-url`. If the model identifier
differs, pass the exact ID printed by `--list-models` with `--model`.
If server authentication is enabled, set `LM_STUDIO_API_KEY` in your environment.

Results go to `results/qwen3_vl_2b_gqa_zero_shot/`: `predictions.jsonl` retains
the raw responses and token usage as each request finishes, `predictions.csv`
contains predictions and per-question scores, and `report.json` contains
aggregate metrics and the run settings. `run_config.json` guards resume against
changed question files or inference settings. Existing outputs require
`--resume` or a fresh directory. The script stops on errors by default;
`--continue-on-error` records failures and continues. Failed labeled requests
count as incorrect, and resume retries them. `--max-tokens` (default 64) and
`--timeout` (default 180 seconds) are configurable; the report counts responses
that reached the token limit. Temperature 0 and a seed reduce sampling variation,
but results can still vary between model quantizations and server versions.

The report's `accuracy` is binary exact-match accuracy (0–1), using the
[GQA accuracy definition](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html).
It trims response whitespace without extracting answers from explanations.
`normalized_accuracy` additionally ignores case and repeated whitespace.
Use `accuracy` for comparisons with this repository's GQA training and testing.
Partial reports score attempted questions and explicitly indicate `complete: false`.
This computes answer accuracy, not GQA's other scene-graph-based metrics.

The repository's `test.csv` is a held-out subset of the **labeled validation
split**, not the official GQA hidden test split. To evaluate official question
JSON and its image directory instead:

```bash
python3 -m scripts.evaluate_gqa_lmstudio \
  --questions /path/to/val_balanced_questions.json \
  --images /path/to/gqa/images \
  --output-dir results/qwen_official_val
```

Official JSON maps question IDs to objects containing `imageId`, `question`,
and optionally `answer`. Unlabeled files produce predictions without accuracy.
`gqa_predictions.json` uses GQA's `questionId`/`prediction` format; use official
question JSON for official evaluation, because this repository's CSV `anno_id`
values identify images rather than official questions. `--limit N` always selects
the first N rows, so a smoke run is not a randomly sampled benchmark score.

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
component (such as `data/gqa_dataset/val.csv`) selects GQA scoring. Use
`--dataset gqa` when your GQA files have other names, or `--dataset plantexpert`
to select the six paper metrics explicitly. The same option works for
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

For other datasets, evaluation reports the six PlantExpertVQA paper metrics: case-insensitive exact
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
