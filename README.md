# OT-VQA

Visual question answering with a frozen ViT image encoder, BERT question tokens,
partial optimal transport fusion, and autoregressive answer generation. The
original Stacked Attention Network (SAN) remains available as a baseline. See
[the implementation and evaluation plan](docs/optimal_transport_vqa_plan.md).
For a code-level walkthrough with diagrams and an interactive transport example,
open [the architecture guide](docs/ot_vqa_architecture.html).

## Setup

```bash
pip install -r requirements.txt
```

The default paths expect GQA CSV files under `data/gqa_dataset` and images under
`data/gqa_dataset/images`. Run `python train.py --help` to see path overrides.

To build a larger local GQA subset with the original one-question-per-image format:

```bash
python -m utils.download_gqa --train-images 10000 --val-images 500 \
  --test-images 500
```

Keep the held-out test split for the final comparison. Dataset downloads require
network access and can be large; choose counts that fit your compute budget.
The downloader spaces dataset viewer requests, honors HTTP 429 retry delays, and
saves metadata pages in `<output>/.row_cache`. Rerun with the same `--output` to
reuse completed pages. If rate limiting persists, set `HF_TOKEN` in your shell
and try `--metadata-workers 1 --request-interval 2`. `--workers` controls image
downloads separately.

To download 5,000 PlantExpertVQA training question–answer pairs and 1,000
validation pairs, with their images:

```bash
python -m utils.download_plantexpert
```

The output is `data/plantexpert_dataset/train.csv`, `val.csv`, and `images/`.
The script streams the source train and validation CSVs once and samples
reproducible groups of question–answer rows (`--seed 42` by default). It does
not use the rate-limited dataset viewer API. The image downloader reads only
the required files from the source ZIP archives. You can change the counts
with `--train-pairs` and `--val-pairs`. Selected rows are cached in
`<output>/.selection_cache`, and rerunning with the same output reuses them
and any downloaded images. Set `HF_TOKEN` in your shell if you have a Hugging
Face token.
For training, pass `--train_csv_path data/plantexpert_dataset/train.csv`,
`--dev_csv_path data/plantexpert_dataset/val.csv`, and
`--img_path data/plantexpert_dataset/images` to `train.py`.

To train the OT model on this PlantExpertVQA subset:

```bash
python train.py --fusion ot --device auto --batch_size 2 \
  --max_answer_tokens 128 \
  --train_csv_path data/plantexpert_dataset/train.csv \
  --dev_csv_path data/plantexpert_dataset/val.csv \
  --img_path data/plantexpert_dataset/images \
  --model_path data/plantexpert_model
```

The run uses the training default of 10 epochs. The 128-token answer limit
covers all answers in the downloaded subset; the model's 38-token default
would truncate many of them. It saves
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
python train.py --fusion ot --device cuda --train_csv_path data/gqa_dataset/train.csv --dev_csv_path data/gqa_dataset/val.csv --img_path data/gqa_dataset/images --model_path data/gqa_model
```

Training prints `Training on device: cuda`. If CUDA is unavailable, `--device cuda`
stops with an error instead of training on CPU. Reduce `--batch_size` from its
default of 4 if the GPU runs out of memory. `--device auto` also uses CUDA when
available, but can fall back to another device.

Training writes `last.pt`, the lowest-validation-loss checkpoint as `best.pt`, a JSONL
metric history, an evaluation plot, and `run_config.json` with dataset hashes.
Resume with `--resume path/to/last.pt`.
New runs default to OT. For a matched SAN baseline, run the same command with
`--fusion san` and a different `--model_path`. Set `--ot_dustbin_mass 0` for a
full-OT ablation. Partial OT defaults to mass `0.2`, Sinkhorn epsilon `0.05`,
20 iterations, and dustbin-to-dustbin cost `1.0`. These values require
validation on your dataset.

## Evaluate

```bash
python test.py --checkpoint data/gqa_model/best.pt --split dev \
  --predictions_csv data/gqa_model/dev_predictions.csv \
  --report_json data/gqa_model/dev_report.json
```

Evaluation reports the six PlantExpertVQA paper metrics: case-insensitive exact
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
includes it. After evaluating SAN and OT on the *same* held-out examples, compare
their prediction files with paired bootstrap intervals:

```bash
python -m utils.compare_predictions \
  --san data/san_model/dev_predictions.csv \
  --ot data/gqa_model/dev_predictions.csv \
  --output-json data/ot_comparison.json
```

Pass multiple files after `--san` and `--ot` in the same seed order when you
have repeated runs. The paper's results are for image-text retrieval; VQA gains
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
