import pandas as pd


def preprocess_text(text):
    """Normalize English text without adding model-specific tokenization."""
    return " ".join(str(text).strip().split())


def process_dataframe(df, require_answers=True):
    df = df.copy()
    required = {"image", "question"} | ({"answer"} if require_answers else set())
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("CSV contains missing images, questions or answers")
    if "anno_id" not in df.columns:
        df["anno_id"] = range(len(df))
    df["question"] = [preprocess_text(x) for x in df["question"]]
    if "answer" in df.columns:
        df["answer"] = [preprocess_text(x) for x in df["answer"].fillna("")]
    else:
        df["answer"] = ""
    return df


def load_dataframe(path, require_answers=True):
    return process_dataframe(pd.read_csv(path, keep_default_na=False), require_answers)


def preprocess_data(args):
    return (load_dataframe(args.train_csv_path),
            load_dataframe(args.dev_csv_path),
            load_dataframe(args.test_csv_path, require_answers=False))


if __name__ == "__main__":
    from configs.arg_parser import get_args
    print(preprocess_data(get_args())[0].head())
