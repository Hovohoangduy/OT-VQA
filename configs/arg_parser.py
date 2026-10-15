import argparse
import sys

from configs.config import Config
from utils.metrics import ALL_METRICS, bertscore_defaults, resolve_dataset


def apply_dataset_defaults(parser, argv):
    """Select built-in Python defaults before parsing explicit CLI overrides."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", type=str.lower)
    bootstrap.add_argument("--model_family")
    selected = bootstrap.parse_known_args(argv)[0]
    parser.set_defaults(**Config.DATASET_DEFAULTS.get(selected.dataset, {}))
    if selected.model_family == "qwen3_vl":
        parser.set_defaults(batch_size=1, gradient_accumulation_steps=8, lr=1e-4,
                            encoder_lr=None, label_smoothing=0., ot_loss_weight=0.,
                            ot_num_heads=1, max_question_tokens=256, max_answer_tokens=64,
                            loss_normalization="example", train_eval_samples=0, amp="fp16")


def get_args(argv=None, *, evaluation=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str.lower,
                        choices=["auto", "gqa", "plantexpert", "scivqa"], default="auto",
                        help="scivqa/plantexpert load their dataset presets; "
                             "auto detects answer metrics from CSV paths")

    parser.add_argument("--model_family", choices=("legacy", "qwen3_vl"), default="legacy")
    parser.add_argument("--prompt_version", choices=("scivqa-v1", "scivqa-v2"), default=None)
    parser.add_argument("--decode_version", choices=("task-v1", "task-v2"), default=None)
    parser.add_argument("--output_drop_prob", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off")
    parser.add_argument("--amp_init_scale", type=float, default=65536.0)
    parser.add_argument("--compute_bertscore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every_updates", type=int, default=25)
    parser.add_argument("--max_updates", type=int, default=None, help="Pilot optimizer-update cap; saves at epoch boundary")
    parser.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vlm_revision", default=None, help="Required immutable HF commit SHA")
    parser.add_argument("--query_revision", default=None, help="Required immutable MiniLM commit SHA")
    parser.add_argument("--visual_tokens", type=int, default=256)
    parser.add_argument("--total_sequence_tokens", type=int, default=2048)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--cost_dim", type=int, default=128)
    parser.add_argument("--quantize_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_rank", type=int, default=0, help="0: adapter-only; 16: OT+LoRA")
    parser.add_argument("--lora_lr", type=float, default=2e-5)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter_control", choices=("ot", "native", "pooled"), default="ot")
    parser.add_argument("--plan_control", choices=("learned", "independent"), default="learned")
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
    parser.add_argument("--image_size", type=int, default=None,
                        help="Square image side; default 256. MobileViTV2 supports multiples of 32")
    parser.add_argument("--image_preprocessing", choices=("crop", "pad"), default=None,
                        help="crop: legacy center crop; pad: preserve the complete figure and aspect ratio")
    parser.add_argument("--ot_visual_memory", action=argparse.BooleanOptionalAction, default=None,
                        help="Append spatial image tokens conditioned by the same OT plan; default off")
    parser.add_argument("--figure_context", action=argparse.BooleanOptionalAction, default=None,
                        help="Append the SciVQA figure type and caption to each question; default off")
    parser.add_argument("--tie_weights", action=argparse.BooleanOptionalAction, default=None,
                        help="Tie decoder output projection to answer token embeddings")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--resume", default=None, help="Version-3/4/5 training checkpoint to resume")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="Keep epoch_0001.pt, epoch_0002.pt, ... in addition to last.pt and best.pt")
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="Separate LR for pretrained text/answer embeddings; --lr applies to fusion/decoder")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Fraction of total optimization steps used for linear LR warmup")
    parser.add_argument("--loss_normalization", choices=("token", "example"), default=None,
                        help="Training CE averaging: token (legacy), or equal weight per answer; "
                             "resume inherits the saved setting")
    parser.add_argument("--selection_metric", choices=("loss", *ALL_METRICS), default=None,
                        help="Validation metric for best.pt and early stopping; new SciVQA runs use "
                             "rouge_l_f1, others loss; resume inherits the saved metric")
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
        "--early_stopping_patience", type=int, default=10,
        help="Stop after this many epochs without improving the selection metric; 0 disables",
    )
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--ffn_hidden", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.2)
    parser.add_argument("--fusion", choices=("ot",), default="ot",
                        help="OT is the only supported fusion method")
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
    parser.add_argument("--ot_num_heads", type=int, default=1,
                        help="Number of heads for Multi-Head Optimal Transport (MH-OT); default 1 (4 for SciVQA)")
    parser.add_argument("--ot_loss_weight", type=float, default=0.0,
                        help="Weight for auxiliary Kantorovich OT alignment loss; default 0.0 (0.05 for SciVQA)")
    parser.add_argument("--fgw_alpha", type=float, default=0.0,
                        help="Fused Gromov-Wasserstein trade-off weight (0.0 = pure feature OT, >0 incorporates 2D spatial chart geometry)")
    parser.add_argument("--unbalanced_tau", type=float, default=None,
                        help="Unbalanced OT marginal relaxation temperature (None = standard balanced OT with dustbin, >0 = soft marginal KL relaxation)")
    parser.add_argument(
        "--freeze_answer_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze pretrained answer-token embeddings (default: enabled)",
    )
    parser.add_argument(
        "--freeze_question_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the pretrained question encoder; enabled by default for fresh SciVQA runs",
    )
    parser.add_argument("--pretrained_eval", action="store_true", help="Evaluate pinned native base without an adapter checkpoint")
    parser.add_argument("--screen_manifest", default=None, help="Fixed dev-only figure screening manifest")
    parser.add_argument("--profile_stages", action="store_true", help="Synchronize CUDA for pilot stage timings")
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
    # Preset choices that affect model inputs or the training objective apply
    # only to fresh runs. Resume inherits the checkpoint unless overridden.
    if evaluation or args.resume:
        supplied = {item.split("=", 1)[0] for item in (sys.argv[1:] if argv is None else argv)
                    if item.startswith("--")}
        for name in ("image_size", "image_preprocessing", "freeze_question_encoder", "tie_weights", "ot_num_heads",
                     "ot_visual_memory", "figure_context", "fgw_alpha", "unbalanced_tau",
                     "ot_loss_weight", "max_question_tokens", "max_answer_tokens"):
            if f"--{name}" not in supplied and f"--no-{name}" not in supplied:
                setattr(args, name, None)
        if args.resume and "--loss_normalization" not in supplied:
            args.loss_normalization = None
    args.explicit_args = sorted({item.split("=", 1)[0] for item in (sys.argv[1:] if argv is None else argv)
                                 if item.startswith("--")})
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
