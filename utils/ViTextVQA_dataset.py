import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from configs.config import Config
from configs.arg_parser import get_args
from utils.data_processing import preprocess_data

args = get_args()
tokenizer = AutoTokenizer.from_pretrained(Config.textmodel_dir)
vocab = tokenizer.get_vocab()

class ViTextVQA_Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            anno_id = row['anno_id']
            image_id = row['image']
            question = row['question']
            answer = row['answer']
        except KeyError as e:
            raise ValueError(f"Missing expected column: {e}")


        image_path = args.img_path + "/" + image_id

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise ValueError(f"Image file not found: {image_path}")
        if self.transform:
            image = self.transform(image)

        return anno_id, image, question, answer

if __name__=="__main__":
    ### load and processing data
    df_train, _, _ = preprocess_data(args)
    train_vlsp_dataset = ViTextVQA_Dataset(df_train, transform=Config.transforms)

    ### Show example datasets
    random_indices = np.random.choice(len(train_vlsp_dataset), 3)

    for idx in random_indices:
        anno_id, image, question, answer = train_vlsp_dataset[idx]
        
        image = image.permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("Question: " + question +"\n"+ "Answer:" + answer, fontsize=16)
        plt.axis('off')
        plt.show()

    train_loader = DataLoader(train_vlsp_dataset, batch_size=args.batch_size, shuffle=True)
    print(len(train_loader))