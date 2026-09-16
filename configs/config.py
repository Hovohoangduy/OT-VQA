import os
import torchvision.transforms as transforms

class Config:
    lr = 0.00001
    text_model = '/kaggle/input/huggingface-bert-variants/bert-base-uncased/bert-base-uncased'
#     image_model = 'google/vit-base-patch16-224-in21k'
    image_model = 'facebook/deit-base-distilled-patch16-224'
#     image_model = 'microsoft/beit-base-patch16-224-pt22k'
    vncore_path = '/kaggle/input/vncorenlp/VnCoreNLP-1.1.1.jar'
    vncore_path2 = '/kaggle/input/vncorenlpv2/VnCoreNLP/VnCoreNLP-1.2.jar'
    imgmodel_dir = 'hf-hub:timm/convnext_small.fb_in22k_ft_in1k_384'
    textmodel_dir = "vinai/phobert-base-v2"
    SEED = 1105
    MAX_LEN = 64
    MAX_LEN_QUES = 28
    MAX_LEN_ANS = 38
    NUM_WORKERS = os.cpu_count()
    transforms = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    ])