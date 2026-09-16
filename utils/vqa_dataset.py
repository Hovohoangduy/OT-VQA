"""Dataset that pairs English questions and answers with image files."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, dataframe, transform=None, img_path="data/gqa_dataset/images"):
        self.data = dataframe
        self.transform = transform
        self.img_path = Path(img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self.img_path / str(row["image"])
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except FileNotFoundError as exc:
            raise ValueError(f"Image file not found: {image_path}") from exc
        if self.transform:
            image = self.transform(image)
        return row.get("anno_id", idx), image, str(row["question"]), str(row.get("answer", ""))
