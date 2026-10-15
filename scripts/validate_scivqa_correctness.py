"""Same-snapshot full-dev controls for prompt and decode fixes; no held-out test.

Frozen image features are cached only within this process. These timings are not
an inference benchmark. Existing outputs are never overwritten.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch

from utils.checkpoint import load_model
from utils.data_processing import load_dataframe
from utils.device import resolve_device,seed_everything
from utils.vqa_dataset import VQADataset,resolve_image_root
from utils.metrics import scivqa_score_pairs
from utils.quality_evidence import guardrail_metrics,paired_cluster_interval
from utils.training_runtime import source_manifest
from train import _csv_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='data/weights/scivqa.pt')
    parser.add_argument('--dev_csv',default='data/scivqa_dataset/val.csv')
    parser.add_argument('--img_path',default='data/scivqa_dataset/images')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch_size',type=int,default=8)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True,exist_ok=True)
    sources = source_manifest(destination)
    seed_everything(1105);torch.set_num_threads(4)
    device=resolve_device(args.device)
    model=load_model(args.checkpoint,device)
    frame=load_dataframe(args.dev_csv)
    root=resolve_image_root(frame,args.img_path,'dev')
    datasets={v:VQADataset(frame,model.image_transform,root,figure_context=model.figure_context,
                           prompt_version=v,dataset_name='scivqa') for v in ('scivqa-v1','scivqa-v2')}
    unique=frame.drop_duplicates('image').index.tolist()
    cache={}; started=perf_counter()
    with torch.inference_mode():
        for start in range(0,len(unique),args.batch_size):
            indices=unique[start:start+args.batch_size]
            images=torch.stack([datasets['scivqa-v1'][i][1] for i in indices]).to(device)
            features,_=model.image_model(images)
            values=model.image_projection(features)
            if model.image_model.has_cls_token:
                values=values[:,1:]
            cache.update({str(frame.iloc[i].image):v.clone() for i,v in zip(indices,values)})
        print(f'Cached {len(cache)} figures',flush=True)
        conditions=[('baseline','scivqa-v1','task-v1'),('decode_fix','scivqa-v1','task-v2'),
                    ('prompt_and_decode_fix','scivqa-v2','task-v2')]
        predictions={name:[] for name,_,_ in conditions}
        for name,prompt,decode in conditions:
            model.decode_version=decode
            for start in range(0,len(frame),args.batch_size):
                indices=list(range(start,min(start+args.batch_size,len(frame))))
                questions=[datasets[prompt][i][2] for i in indices]
                v=torch.stack([cache[str(frame.iloc[i].image)] for i in indices])
                q,blocked,_,glob=model.question_encoder.encode_tokens(questions)
                memory,memory_mask=model.ot_fusion(v,q,~blocked,question_global=glob)
                ids=model._generate_from_memory(memory,memory_mask,model.max_answer_tokens,questions,True)
                predictions[name].extend(model.answers_from_ids(ids))
            print(f'{name}: {len(predictions[name])} answers, elapsed {perf_counter()-started:.1f}s',flush=True)
    report={'checkpoint_sha256':_csv_digest(args.checkpoint),'dev_csv_sha256':_csv_digest(args.dev_csv),
            'examples':len(frame),'figures':len(cache),'environment':sources,
            'conditions':{},'elapsed_seconds':perf_counter()-started,
            'historical_baseline_status':'Stored 0.47053 lacks original manifest; not used as matched baseline.'}
    baseline_scores=None
    for name,prompt,decode in conditions:
        preds=predictions[name]
        rows=scivqa_score_pairs(frame.answer.tolist(),preds,compute_bertscore=False)
        scores=[r['rouge_l_f1'] for r in rows]
        if baseline_scores is None:
            baseline_scores=scores
        result={'prompt_version':prompt,'decode_version':decode,'rouge_l_f1':sum(scores)/len(scores),
                'guardrails':guardrail_metrics(frame.answer.tolist(),preds),
                'per_group':{kind:sum(scores[i] for i in idx)/len(idx)
                             for kind,idx in frame.groupby('qa_pair_type').indices.items()},
                'delta_vs_baseline':paired_cluster_interval([b-a for a,b in zip(baseline_scores,scores)],frame.image)}
        report['conditions'][name]=result
        out=frame[['anno_id','image','paper_id','qa_pair_type','answer']].rename(columns={'answer':'reference'}).copy()
        out['prediction']=preds;out['rouge_l_f1']=scores
        out.to_csv(destination/f'{name}.csv',index=False)
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['conditions'],indent=2),flush=True)


if __name__=='__main__':
    main()
