import os
import torchvision.transforms as transforms

class Config:
    lr = 0.00001
    text_model = "sentence-transformers/all-MiniLM-L12-v2"
    image_model = "apple/mobilevitv2-2.0-imagenet1k-256"
    SEED = 1105
    MAX_LEN_QUES = 32
    MAX_LEN_ANS = 112
    NUM_WORKERS = os.cpu_count()
    transforms = transforms.Compose([transforms.Resize(288),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    ])
