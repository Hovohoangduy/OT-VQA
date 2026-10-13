import os
import torchvision.transforms as transforms

class Config:
    lr = 0.00001
    text_model = "sentence-transformers/all-MiniLM-L12-v2"
    image_model = "apple/mobilevitv2-2.0-imagenet1k-256"
    SEED = 1105
    MAX_LEN_QUES = 32
    MAX_LEN_ANS = 112
    SCIVQA_DATA_ROOT = "/kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset"
    # Dataset-specific CLI defaults; remaining training settings are shared.
    DATASET_DEFAULTS = {
        "scivqa": {
            "train_csv_path": f"{SCIVQA_DATA_ROOT}/train.csv",
            "dev_csv_path": f"{SCIVQA_DATA_ROOT}/val.csv",
            "test_csv_path": f"{SCIVQA_DATA_ROOT}/test.csv",
            "img_path": f"{SCIVQA_DATA_ROOT}/images",
            "model_path": "/kaggle/working/scivqa_model_tuned",
            "device": "cuda",
            # Starting tuning recipe; validate scores before final test evaluation.
            "epochs": 60,
            "batch_size": 16,  # Per GPU: global batch is 32 with two GPUs.
            "lr": 1e-4,  # New fusion, projections, decoder, and vocabulary head.
            "encoder_lr": 1e-5,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "label_smoothing": 0.05,
            "drop_prob": 0.1,
            "gradient_clip": 1.0,
            "early_stopping_patience": 10,
            "selection_metric": "rouge_l_f1",
            "train_eval_samples": 512,
            "num_workers": 2,
            "fusion": "ot",
            "max_question_tokens": 256,
            "max_answer_tokens": 64,
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
