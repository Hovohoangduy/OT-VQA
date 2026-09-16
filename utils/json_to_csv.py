import json
from pathlib import Path

import pandas as pd

from configs.arg_parser import get_args


def convert_json_folder(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(input_dir).glob("*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        rows = []
        for annotation in data["annotations"]:
            answers = annotation.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            # Multiple annotations are alternative correct answers, not one long answer.
            answer = answers[0] if answers else ""
            rows.append({"anno_id": annotation["id"], "image": f"{annotation['image_id']}.jpg",
                         "question": annotation["question"], "answer": answer})
        target = output_dir / f"{source.stem}.csv"
        pd.DataFrame(rows, columns=["anno_id", "image", "question", "answer"]).to_csv(target, index=False)
        print(f"Created {target}")


if __name__ == "__main__":
    args = get_args()
    convert_json_folder(args.json_folder_path, args.csv_folder_path)
