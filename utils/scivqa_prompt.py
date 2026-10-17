"""Build SciVQA inference context without reading reference answers.

Figure focus and compound navigation follow sci-vqa's scivqa-v2 prompts.
Only task formatting uses qa_pair_type; answerability annotations never do.
"""

import json


PROMPT_VERSION = "scivqa-llm-v1"
CONTEXT_FIELDS = ("original_question", "answer_options", "caption", "figure_type",
                  "compound", "figs_numb", "qa_pair_type")
PROVENANCE_FIELDS = ("figure_id", "venue", "categories", "source_dataset", "paper_id", "pdf_url")
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
SYSTEM_PROMPT = (
    "Answer the scientific question using the figure and the provided context. "
    "Return only a concise answer, without reasoning or an 'Answer:' prefix. "
    "For binary questions return Yes or No. For multiple-choice questions return "
    "only the correct option letter(s), separated by commas when several are correct. "
    "If the answer cannot be inferred from the figure and provided context, "
    "return 'It is not possible to answer this question based only on the provided data.'. "
    "Do not guess or consult external sources."
)


def build_prompt(question: str, context: dict[str, str], mode: str = "full") -> str:
    """Reconstruct exports from original_question to avoid duplicate captions/options."""
    if mode not in {"full", "question-only"}:
        raise ValueError(f"Unknown SciVQA context mode: {mode}")
    text = context.get("original_question", "").strip() or question.strip()
    raw_options = context.get("answer_options", "").strip()
    if raw_options:
        options = json.loads(raw_options)
        if not isinstance(options, list):
            raise ValueError("SciVQA answer_options must be a JSON list of label/text objects")
        choices = {}
        for option in options:
            if not isinstance(option, dict) or len(option) != 1:
                raise ValueError("Each SciVQA answer option must have exactly one label")
            label, value = next(iter(option.items()))
            if label in choices or label not in "ABCDEF" or len(label) != 1 or value is None:
                raise ValueError(f"Duplicate or invalid SciVQA answer option: {label!r}")
            choices[label] = str(value)
        if choices and "answer options:" not in text.lower():
            text += "\nAnswer options: " + "; ".join(f"{key}: {value}" for key, value in choices.items())
    kind = context.get("qa_pair_type", "").lower()
    if "non-binary" in kind:
        text += "\nReturn only the letter(s) of the correct answer."
    elif "binary" in kind:
        text += "\nAnswer with 'Yes' or 'No'."
    # Do not branch on 'unanswerable': that annotation reveals the target.
    if mode == "full":
        compound = context.get("compound", "").strip().lower() in {"true", "1", "yes"}
        if compound and "subfigure navigation:" not in text.lower():
            count = context.get("figs_numb", "").strip()
            description = f"contains {count} subfigures" if count else "contains multiple subfigures"
            text += (f"\nSubfigure navigation: This compound figure {description}. "
                     "Use the panel referenced in the question.")
        figure_type = context.get("figure_type", "").strip()
        if figure_type and "figure type:" not in text.lower():
            focus = FIGURE_TYPE_FOCUS.get(figure_type.lower(), "Focus on colors, labels, axes, and visual elements.")
            text += f"\nFigure type: {figure_type} ({focus})"
        caption = context.get("caption", "").strip()
        if caption and "figure caption:" not in text.lower():
            text += f"\nFigure caption: {caption}"
    return text
