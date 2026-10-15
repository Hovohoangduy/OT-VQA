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


FIGURE_TYPE_FOCUS = {
    "line chart": "Focus on colors of different lines and meanings, X and Y axis labels and units, scale and range of values, trends.",
    "bar chart": "Focus on colors of different bars, heights and positions, X and Y axis labels and units.",
    "scatter plot": "Focus on distribution of points, clusters, outliers, X and Y axis coordinates.",
    "confusion matrix": "Focus on row and column labels, numerical values in each cell, color intensity.",
    "pie chart": "Focus on segments and their labels, percentage or proportion values, segment colors.",
    "box plot": "Focus on median line position, box boundaries Q1 and Q3, whisker extent, outliers.",
    "histogram": "Focus on bin ranges, frequency heights, X and Y axis labels.",
    "tree": "Focus on hierarchical levels, root and leaf nodes, connecting edges, labels.",
    "graph": "Focus on nodes, vertices, edges, connectivity, labels.",
    "architecture diagram": "Focus on component blocks, flow directions, connections, layer names.",
    "neural networks": "Focus on layers, connections, input/output dimensions, module names.",
}


def append_figure_context(question, caption, figure_type, compound=None, figs_numb=None):
    """Keep the question first, enrich with compound subfigure guidance, figure focus, and caption."""
    text = str(question)
    # 1. Compound figure subfigure navigation
    if compound and figs_numb is not None:
        nav = f"This is a compound figure containing multiple subfigures. Navigate to subfigure {figs_numb} in the compound figure to answer the question."
        if nav not in text:
            text += f"\nSubfigure navigation: {nav}"
    # 2. Figure type & reasoning focus
    if isinstance(figure_type, str) and figure_type.strip() and "Figure type:" not in text:
        ft = figure_type.strip().lower()
        focus = FIGURE_TYPE_FOCUS.get(ft, "Focus on colors, labels, axes, and visual elements.")
        text += f"\nFigure type: {figure_type.strip()} ({focus})"
    # 3. Figure caption
    if isinstance(caption, str) and caption.strip() and "Figure caption:" not in text:
        text += f"\nFigure caption: {caption.strip()}"
    return text


def format_qa_prompt(question, qa_pair_type=None):
    """Add concise format guidance to question based on SciVQA question type."""
    text = str(question)
    if not isinstance(qa_pair_type, str) or not qa_pair_type.strip():
        return text
    kind = qa_pair_type.lower()
    if "binary" in kind and "non-binary" not in kind:
        if "answer with" not in text.lower() and "yes or no" not in text.lower():
            text += "\nAnswer with 'Yes' or 'No'."
    elif "non-binary" in kind:
        if "return only" not in text.lower() and "letter" not in text.lower():
            text += "\nReturn only the letter(s) of the correct answer."
    elif "infinite" in kind and "visual" in kind:
        lower_q = text.lower()
        numeric_cues = (
            "how many", "what is the value", "what is the number", "percentage",
            "rate", "coordinate", "maximum", "minimum", "average", "total",
            "ratio", "score", "amount", "quantity", "what was the value",
            "what is the percentage", "level", "threshold", "what number",
            "what is the highest", "what is the lowest", "approximate",
        )
        if any(cue in lower_q for cue in numeric_cues) and "numerical value" not in lower_q and "number" not in lower_q:
            text += "\nAnswer with only the raw numerical value."
    return text


class VQADataset(Dataset):
    def __init__(self, dataframe, transform=None, img_path="data/gqa_dataset/images",
                 dataset_name="plantexpert", figure_context=False, qa_prompt=True):
        if figure_context and not {"caption", "figure_type"}.issubset(dataframe.columns):
            raise ValueError("Figure context requires caption and figure_type CSV columns")
        self.data = dataframe
        self.transform = transform
        self.img_path = Path(img_path)
        self.dataset_name = dataset_name
        self.figure_context = figure_context
        self.qa_prompt = qa_prompt

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
        if self.qa_prompt and "qa_pair_type" in row and isinstance(row["qa_pair_type"], str):
            question = format_qa_prompt(question, row["qa_pair_type"])
        if self.figure_context:
            compound = bool(row.get("compound", False))
            figs_numb = row.get("figs_numb", None)
            question = append_figure_context(
                question, row.get("caption"), row.get("figure_type"),
                compound=compound, figs_numb=figs_numb,
            )
        return row.get("anno_id", idx), image, question, str(row.get("answer", ""))
