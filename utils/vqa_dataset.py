"""Dataset that pairs English questions and answers with image files."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def resolve_image_root(dataframe, img_path, split, override=None):
    """Resolve flat, split-folder, and split-relative image layouts."""
    if override:
        return Path(override)
    root = Path(img_path)
    if "image" not in dataframe or dataframe.empty:
        return root
    image_value = str(dataframe.iloc[0]["image"])
    if (root / image_value).is_file():
        return root
    split_folder = "val" if split == "dev" else split
    candidate = root / split_folder
    if (candidate / image_value).is_file():
        return candidate
    return root


def append_figure_context(question, caption, figure_type):
    """Keep the question first so long captions cannot truncate it away."""
    text = str(question)
    if isinstance(figure_type, str) and figure_type.strip() and "Figure type:" not in text:
        text += f"\nFigure type: {figure_type.strip()}"
    if isinstance(caption, str) and caption.strip() and "Figure caption:" not in text:
        text += f"\nFigure caption: {caption.strip()}"
    return text


class VQADataset(Dataset):
    def __init__(self, dataframe, transform=None, img_path="data/gqa_dataset/images",
                 dataset_name="plantexpert", figure_context=False):
        if figure_context and not {"caption", "figure_type"}.issubset(dataframe.columns):
            raise ValueError("Figure context requires caption and figure_type CSV columns")
        self.data = dataframe
        self.transform = transform
        self.img_path = Path(img_path)
        self.dataset_name = dataset_name
        self.figure_context = figure_context

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self.img_path / str(row["image"])
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except FileNotFoundError as exc:
            raise ValueError(
                f"Image file not found: {image_path}. Set the matching split image "
                "folder with --train_img_path, --dev_img_path, or --test_img_path."
            ) from exc
        if self.transform:
            image = self.transform(image)
        question = str(row["question"])
        if self.figure_context:
            question = append_figure_context(question, row["caption"], row["figure_type"])
        return row.get("anno_id", idx), image, question, str(row.get("answer", ""))
