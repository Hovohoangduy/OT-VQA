import os
import torchvision.transforms as transforms

class Config:
    lr = 0.00001
    text_model = "sentence-transformers/all-MiniLM-L12-v2"
    image_model = "apple/mobilevitv2-2.0-imagenet1k-256"
    SEED = 1105
    MAX_LEN_QUES = 32
    MAX_LEN_ANS = 112
    # Dataset-specific CLI defaults; remaining training settings are shared.
    DATASET_DEFAULTS = {
        "scivqa": {
            "train_csv_path": "data/scivqa_dataset/train.csv",
            "dev_csv_path": "data/scivqa_dataset/val.csv",
            "test_csv_path": "data/scivqa_dataset/test.csv",
            "img_path": "data/scivqa_dataset/images",
            "model_path": "data/scivqa_model",
            "epochs": 30,
            "lr": lr,
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
