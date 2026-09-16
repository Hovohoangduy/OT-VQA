import os
import torchvision.transforms as transforms

class Config:
    lr = 0.00001
    text_model = "bert-base-uncased"
    image_model = 'facebook/deit-base-distilled-patch16-224'
    SEED = 1105
    MAX_LEN = 64
    MAX_LEN_QUES = 28
    MAX_LEN_ANS = 38
    NUM_WORKERS = os.cpu_count()
    transforms = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    ])
