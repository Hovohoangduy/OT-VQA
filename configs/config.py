import os
import torchvision.transforms as transforms

class Config:
    lr = 0.00001
    text_model = "sentence-transformers/all-MiniLM-L12-v2"
    image_model = "google/siglip-base-patch16-384"
    SEED = 1105
    MAX_LEN_QUES = 32
    MAX_LEN_ANS = 112
    # Dataset-specific paths and training defaults; explicit CLI flags override them.
    DATASET_DEFAULTS = {
        "scivqa": {
            "train_csv_path": "data/scivqa_dataset/train.csv",
            "dev_csv_path": "data/scivqa_dataset/val.csv",
            "test_csv_path": "data/scivqa_dataset/test.csv",
            "img_path": "data/scivqa_dataset/images",
            "model_path": "data/scivqa_model",
            "image_model": "google/siglip-base-patch16-384",
            "epochs": 30,
            "batch_size": 8,
            "lr": 1e-5,
            "encoder_lr": 2e-6,
            "warmup_ratio": 0.05,
            "weight_decay": 0.05,
            "label_smoothing": 0.1,
            "loss_normalization": "example",
            "drop_prob": 0.25,
            "freeze_question_encoder": True,
            "early_stopping_patience": 8,
            "train_eval_samples": 512,
            "num_workers": 2,
            "fusion": "ot",
            "max_question_tokens": 512,
            "max_answer_tokens": 64,
            "image_preprocessing": "pad",
            "image_size": 384,
            "figure_context": True,
            "tie_weights": True,
            "ot_num_heads": 4,
            "ot_loss_weight": 0.05,
        },
        "plantexpert": {
            "train_csv_path": "data/plantexpert_dataset/train.csv",
            "dev_csv_path": "data/plantexpert_dataset/val.csv",
            "test_csv_path": "data/plantexpert_dataset/test.csv",
            "img_path": "data/plantexpert_dataset/images",
            "model_path": "data/plantexpert_model",
            "max_question_tokens": MAX_LEN_QUES,
            "max_answer_tokens": MAX_LEN_ANS,
        },
    }
    NUM_WORKERS = os.cpu_count()
    transforms = transforms.Compose([transforms.Resize(288),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    ])
