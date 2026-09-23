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
    parser.add_argument(
        "--train_csv_path", "--train_csv", dest="train_csv_path", type=str,
        default="data/gqa_dataset/train.csv", help="Path to training CSV file",
    )
    parser.add_argument(
        "--test_csv_path", "--test_csv", dest="test_csv_path", type=str,
        default="data/gqa_dataset/test.csv", help="Path to testing CSV file",
    )
    parser.add_argument(
        "--dev_csv_path", "--dev_csv", dest="dev_csv_path", type=str,
        default="data/gqa_dataset/val.csv", help="Path to development CSV file",
    )
    parser.add_argument(
        "--model_path", "--save_dir", dest="model_path", type=str,
        default="data/gqa_model", help="Directory where training outputs are saved",
    )
    parser.add_argument("--json_folder_path", type=str, default="data/json", help="Path to folder containing JSON files")
    parser.add_argument("--csv_folder_path", type=str, default="data/csv", help="Path to folder where CSV files will be saved")
    
    parser.add_argument("--text_model", default="bert-base-uncased",
                        help="English Hugging Face tokenizer and text encoder")
    parser.add_argument("--image_model", default=Config.image_model)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--fusion", "--fusion_method",
        dest="fusion",
        choices=[
            "cross_attention", "ot_evidence_routing", "softmax_evidence_routing",
            "ot_evidence_routing_v2", "softmax_evidence_routing_v2",
        ],
        default="cross_attention",
        help="Visual evidence integration architecture",
    )
    parser.add_argument("--fusion_dropout", type=float, default=0.2)
    parser.add_argument("--cross_fusion_layers", type=int, default=1)
    parser.add_argument("--routing_slots", type=int, default=4)
    parser.add_argument("--routing_steps", type=int, default=2)
    parser.add_argument("--routing_dim", type=int, default=256)
    parser.add_argument("--routing_epsilon", type=float, default=0.1)
    parser.add_argument("--routing_tau", type=float, default=None)
    parser.add_argument("--routing_iterations", type=int, default=40)
    parser.add_argument("--routing_tolerance", type=float, default=0.001)
    parser.add_argument("--routing_preference_smoothing", type=float, default=None)
    parser.add_argument("--routing_null_min", type=float, default=0.02)
    parser.add_argument("--routing_null_max", type=float, default=0.25)
    parser.add_argument(
        "--routing_visual_preference",
        choices=["question_conditioned", "uniform"],
        default="question_conditioned",
    )
    parser.add_argument(
        "--routing_preference_transform",
        choices=["softmax", "sparsemax", "topk"],
        default=None,
        help="V1 defaults to softmax; V2 defaults to sparsemax",
    )
    parser.add_argument("--routing_preference_topk", type=int, default=32)
    parser.add_argument(
        "--routing_cost_scale_mode", choices=["fixed", "learned"], default=None,
        help="V1 defaults to fixed; V2 defaults to a bounded learned scale",
    )
    parser.add_argument("--routing_cost_scale", type=float, default=None)
    parser.add_argument("--routing_cost_scale_min", type=float, default=1.0)
    parser.add_argument("--routing_cost_scale_max", type=float, default=20.0)
    parser.add_argument(
        "--routing_question_conditioned_keys",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--routing_gate_max", type=float, default=4.0)
    parser.add_argument(
        "--routing_tau_warmup_epochs", type=int, default=0,
        help="V2 epochs using independent routing before OT coupling",
    )
    parser.add_argument(
        "--routing_tau_ramp_epochs", type=int, default=0,
        help="V2 epochs over which tau increases linearly to routing_tau",
    )
    parser.add_argument(
        "--routing_query_diversity_weight",
        type=float,
        default=None,
        help=(
            "Weight for the routing-query orthogonality loss. This prevents "
            "all evidence slots from learning the same transport row"
        ),
    )
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
        "--fusion_lr", type=float, default=None,
        help="Optional learning rate for the randomly initialized fusion/router",
    )
    parser.add_argument(
        "--decoder_lr", type=float, default=None,
        help="Optional learning rate for decoder, answer projection, and output head",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument(
        "--lr_schedule", choices=["linear", "cosine"], default="linear",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--weight_decay", type=float, default=0.05,
        help="AdamW weight decay; the small GQA subset benefits from stronger regularization",
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0,
        help="Maximum gradient norm; set to 0 to disable clipping",
    )
    parser.add_argument(
        "--counterfactual_weight", type=float, default=0.0,
        help="Margin-loss weight comparing the true image with an in-batch wrong image",
    )
    parser.add_argument("--counterfactual_margin", type=float, default=0.2)
    parser.add_argument("--counterfactual_fraction", type=float, default=0.25)
    parser.add_argument("--counterfactual_warmup_epochs", type=int, default=5)
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
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA float16 outside the float32 OT solver",
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit evaluation checkpoint")
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Report fusion/routing, latency, and output-diversity diagnostics",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Compute device; auto prefers CUDA, then Apple MPS, then CPU",
    )
    parser.add_argument(
        "--distributed", action="store_true", help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.routing_tau is None:
        args.routing_tau = 0.1 if args.fusion.endswith("_v2") else 0.5
    if args.routing_query_diversity_weight is None:
        args.routing_query_diversity_weight = (
            0.0 if args.fusion.endswith("_v2") else 0.05
        )
    return args
