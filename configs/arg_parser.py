import argparse

from configs.config import Config
from model.fusion import FUSION_METHODS
from utils.metrics import ALL_METRICS, bertscore_defaults, resolve_dataset


def apply_dataset_defaults(parser, argv):
    """Select built-in Python defaults before parsing explicit CLI overrides."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", type=str.lower)
    bootstrap.add_argument("--resume")
    selected = bootstrap.parse_known_args(argv)[0]
    defaults = dict(Config.DATASET_DEFAULTS.get(selected.dataset, {}))
    if selected.resume:
        # Let training recover optimizer groups and selection from the checkpoint.
        # Explicit flags are still parsed below and checked for compatibility.
        defaults.pop("encoder_lr", None)
        defaults.pop("selection_metric", None)
    parser.set_defaults(**defaults)


def get_args(argv=None, *, evaluation=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str.lower,
                        choices=["auto", "gqa", "plantexpert", "scivqa"], default="auto",
                        help="scivqa/plantexpert load their dataset presets; "
                             "auto detects answer metrics from CSV paths")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--img_path", type=str, default="data/gqa_dataset/images", help="Path to image folder")
    parser.add_argument("--train_img_path", default=None, help="Optional training image folder override")
    parser.add_argument("--dev_img_path", default=None, help="Optional validation image folder override")
    parser.add_argument("--test_img_path", default=None, help="Optional test image folder override")
    parser.add_argument("--train_csv_path", type=str, default="data/gqa_dataset/train.csv", help="Path to training CSV file")
    parser.add_argument("--test_csv_path", type=str, default="data/gqa_dataset/test.csv", help="Path to testing CSV file")
    parser.add_argument("--dev_csv_path", type=str, default="data/gqa_dataset/val.csv", help="Path to development CSV file")
    parser.add_argument("--model_path", type=str, default="data/gqa_model", help="Path to save trained model")
    parser.add_argument("--json_folder_path", type=str, default="data/json", help="Path to folder containing JSON files")
    parser.add_argument("--csv_folder_path", type=str, default="data/csv", help="Path to folder where CSV files will be saved")
    
    parser.add_argument("--text_model", default=Config.text_model,
                        help="English Hugging Face tokenizer and text encoder")
    parser.add_argument("--image_model", default=Config.image_model)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--resume", default=None, help="Version-3/4 training checkpoint to resume")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="Keep epoch_0001.pt, epoch_0002.pt, ... in addition to last.pt and best.pt")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="Separate LR for pretrained text/answer embeddings; --lr applies to fusion/decoder")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Fraction of total optimization steps used for linear LR warmup")
    parser.add_argument("--selection_metric", choices=("loss", *ALL_METRICS), default=None,
                        help="Validation metric for best.pt and early stopping; default loss, or resume setting")
    parser.add_argument("--train_eval_samples", type=int, default=-1,
                        help="Training examples to score each epoch: -1 all, 0 skip, positive fixed random subset")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes per GPU")
    parser.add_argument(
        "--weight_decay", type=float, default=0.05,
        help="AdamW weight decay; the small GQA subset benefits from stronger regularization",
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0,
        help="Maximum gradient norm; set to 0 to disable clipping",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing used only by the training loss",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=8,
        help="Stop after this many epochs without improvement in the selection metric; 0 disables",
    )
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--ffn_hidden", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.2)
    parser.add_argument("--fusion", choices=FUSION_METHODS, default=None,
                        help="Fusion method (default: ot; resume uses checkpoint setting)")
    parser.add_argument("--fusion_glimpses", type=int, default=None,
                        help="SAN/BAN attention glimpses (default: 2)")
    parser.add_argument("--fusion_queries", type=int, default=None,
                        help="Q-Former learned queries (default: 8)")
    parser.add_argument("--fusion_layers", type=int, default=None,
                        help="Q-Former blocks (default: 2)")
    parser.add_argument("--max_answer_tokens", type=int, default=None,
                        help="Answer length including start/end tokens; default 112 covers the local PlantExpertVQA splits")
    parser.add_argument("--max_question_tokens", type=int, default=None,
                        help="Question length including special tokens; default 32 covers the local PlantExpertVQA splits")
    parser.add_argument("--ot_epsilon", type=float, default=None,
                        help="Sinkhorn entropy coefficient (new OT runs default to 0.05)")
    parser.add_argument("--ot_iterations", type=int, default=None,
                        help="Sinkhorn iterations (new OT runs default to 20)")
    parser.add_argument("--ot_dustbin_mass", type=float, default=None,
                        help="Marginal mass assigned to each dustbin (new OT runs default to 0.2)")
    parser.add_argument("--ot_dustbin_cost", type=float, default=None,
                        help="Dustbin-to-dustbin transport cost (new OT runs default to 1.0)")
    parser.add_argument(
        "--freeze_answer_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze pretrained answer-token embeddings (default: enabled)",
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit evaluation checkpoint")
    parser.add_argument("--predictions_csv", default=None,
                        help="Evaluation output with each generated answer")
    parser.add_argument("--report_json", default=None,
                        help="Evaluation output with accuracy and runtime summary")
    parser.add_argument("--bertscore_model", default=None,
                        help="BERTScore encoder; SciVQA defaults to roberta-large, others to bert-base-uncased")
    parser.add_argument("--bertscore_device", choices=["auto", "cpu", "cuda", "mps"],
                        default="cpu", help="Device for BERTScore, separate from VQA model")
    parser.add_argument("--bertscore_batch_size", type=int, default=16)
    parser.add_argument("--bertscore_rescale", action=argparse.BooleanOptionalAction,
                        default=None, help="Rescale BERTScore with its English baseline (default off for SciVQA)")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Compute device; auto prefers CUDA, then Apple MPS, then CPU",
    )
    apply_dataset_defaults(parser, argv)
    args = parser.parse_args(argv)
    dataset_path = args.train_csv_path
    if evaluation:
        dataset_path = args.dev_csv_path if args.split == "dev" else args.test_csv_path
    dataset_name = resolve_dataset(args.dataset, dataset_path)
    model, rescale = bertscore_defaults(dataset_name)
    if args.bertscore_model is None:
        args.bertscore_model = model
    if args.bertscore_rescale is None:
        args.bertscore_rescale = rescale
    return args
