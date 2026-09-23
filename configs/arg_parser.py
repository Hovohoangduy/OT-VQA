import argparse

from configs.config import Config

def get_args(argv=None):
    parser = argparse.ArgumentParser()

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
    
    parser.add_argument("--text_model", default="bert-base-uncased",
                        help="English Hugging Face tokenizer and text encoder")
    parser.add_argument("--image_model", default=Config.image_model)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--fusion",
        choices=["cross_attention"],
        default="cross_attention",
        help="Retained for explicit experiment records; Cross-Attention is the only model",
    )
    parser.add_argument("--fusion_dropout", type=float, default=0.2)
    parser.add_argument("--cross_fusion_layers", type=int, default=1)
    parser.add_argument(
        "--alignment_mode",
        choices=["none", "ot_contrastive_distill"],
        default="none",
        help=(
            "Train a contrastive UOT alignment teacher and distill it into native "
            "Cross-Attention; OT is omitted from the exported inference model"
        ),
    )
    parser.add_argument("--alignment_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--ot_alignment_lr", type=float, default=1e-4,
        help="Learning rate for the randomly initialized OT alignment adapters",
    )
    parser.add_argument("--ot_alignment_dim", type=int, default=128)
    parser.add_argument("--ot_alignment_epsilon", type=float, default=0.1)
    parser.add_argument("--ot_alignment_tau_visual", type=float, default=0.5)
    parser.add_argument("--ot_alignment_tau_question", type=float, default=0.5)
    parser.add_argument("--ot_alignment_iterations", type=int, default=20)
    parser.add_argument("--ot_alignment_tolerance", type=float, default=0.001)
    parser.add_argument("--ot_negative_count", type=int, default=3)
    parser.add_argument("--ot_negative_queue_size", type=int, default=32)
    parser.add_argument("--ot_contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--ot_distill_weight", type=float, default=0.02)
    parser.add_argument("--ot_distill_warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--ot_gate_failure_policy",
        choices=["fallback", "error"],
        default="fallback",
        help=(
            "Continue with native Cross-Attention when the OT teacher fails validation, "
            "or stop with an error"
        ),
    )
    parser.add_argument(
        "--student_init_checkpoint",
        default=None,
        help="Optional student-only checkpoint used as identical initialization across paired runs",
    )
    parser.add_argument(
        "--save_student_initialization",
        default=None,
        help="Write the freshly initialized student checkpoint to this path and exit",
    )
    parser.add_argument("--feature_cache", default=None, help="Optional precomputed feature-cache folder")
    parser.add_argument(
        "--resume", default=None,
        help="Version-3 standard checkpoint or version-4 OT-alignment checkpoint to resume",
    )
    parser.add_argument("--seed", type=int, default=1105)
    parser.add_argument("--lr", type=float, default=None)
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
        help="Stop after this many epochs without better generated validation F1; 0 disables",
    )
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--ffn_hidden", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.2)
    parser.add_argument(
        "--freeze_answer_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze pretrained answer-token embeddings (default: enabled)",
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit evaluation checkpoint")
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Report Cross-Attention, latency, and output-diversity diagnostics",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Compute device; auto prefers CUDA, then Apple MPS, then CPU",
    )
    return parser.parse_args(argv)
