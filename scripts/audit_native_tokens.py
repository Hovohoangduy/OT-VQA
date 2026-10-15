"""Train/dev-only query/answer token coverage for pinned native preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from utils.data_processing import load_dataframe
from utils.vqa_dataset import build_question


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data_root',default='data/scivqa_dataset')
    args=parser.parse_args()
    native=AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203')
    query=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2',revision='a50ef00143b4d5391434df20ae11632588ac25be')
    report={}
    for split,csv in [('train','train.csv'),('dev','val.csv')]:
        path=Path(args.data_root)/csv
        frame=load_dataframe(path)
        questions=[build_question(r.question,qa_prompt=False,answer_options=r.answer_options) for r in frame.itertuples()]
        qlens=[len(x) for x in query(questions,add_special_tokens=True,truncation=False).input_ids]
        answers=[len(x)+1 for x in native(frame.answer.tolist(),add_special_tokens=False,truncation=False).input_ids]
        report[split]={'examples':len(frame),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'query_max_tokens':max(qlens),'queries_over_256':sum(x>256 for x in qlens),
                       'native_answer_with_eos_max_tokens':max(answers),'native_answers_over_64':sum(x>64 for x in answers)}
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
