import pandas as pd


def preprocess_text(text, language="vi"):
    text = str(text).strip()
    if language == "vi":
        from underthesea import word_tokenize, text_normalize
        return word_tokenize(text_normalize(text), format="text")
    if language != "en":
        raise ValueError(f"Unsupported language: {language}")
    return " ".join(text.split())


def process_dataframe(df, language="vi", require_answers=True):
    df = df.copy()
    required = {"image", "question"} | ({"answer"} if require_answers else set())
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("CSV contains missing images, questions or answers")
    if "anno_id" not in df.columns:
        df["anno_id"] = range(len(df))
    df["question"] = [preprocess_text(x, language) for x in df["question"]]
    if "answer" in df.columns:
        df["answer"] = [preprocess_text(x, language) for x in df["answer"].fillna("")]
    else:
        df["answer"] = ""
    return df


def load_dataframe(path, language="vi", require_answers=True):
    return process_dataframe(pd.read_csv(path, keep_default_na=False), language, require_answers)


def preprocess_data(args):
    language = getattr(args, "language", "vi")
    return (load_dataframe(args.train_csv_path, language),
            load_dataframe(args.dev_csv_path, language),
            load_dataframe(args.test_csv_path, language, require_answers=False))


if __name__ == "__main__":
    from configs.arg_parser import get_args
    print(preprocess_data(get_args())[0].head())
