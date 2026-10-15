"""Exercise actual CLI resolution, loaders, selection and epoch-boundary resume."""
from pathlib import Path
import json

import pandas as pd
import pytest
import torch
from PIL import Image

import model.vqa_model as vqa
from configs.arg_parser import get_args
from train import _run_training
from tests.test_legacy_roundtrip import ImageEmbedding,QuestionEmbedding,AnswerEmbedding
from utils.checkpoint import read_checkpoint


def fixture_args(monkeypatch,tmp_path):
    for name,cls in [('ImageEmbedding',ImageEmbedding),('QuestionEmbedding',QuestionEmbedding),('AnswerEmbedding',AnswerEmbedding)]:
        monkeypatch.setattr(vqa,name,cls)
    # Preserve the real VQAModel/decoder while keeping the vision fixture tiny.
    import train
    monkeypatch.setattr(train,'build_image_transform',lambda *args: lambda image: torch.ones(3,2,2))
    images=tmp_path/'images';images.mkdir()
    Image.new('RGB',(4,4)).save(images/'x.png')
    frame=pd.DataFrame({'anno_id':range(5),'image':['x.png']*5,'question':['q']*5,'answer':['ab','a','b','aa','bb']})
    train_csv=tmp_path/'train.csv';dev_csv=tmp_path/'dev.csv'
    frame.to_csv(train_csv,index=False);frame.iloc[:2].to_csv(dev_csv,index=False)
    argv=['--dataset','gqa','--train_csv_path',str(train_csv),'--dev_csv_path',str(dev_csv),
          '--img_path',str(images),'--model_path',str(tmp_path/'run'),'--device','cpu',
          '--batch_size','2','--gradient_accumulation_steps','2','--epochs','2',
          '--d_model','8','--ffn_hidden','16','--num_layers','1','--num_heads','2',
          '--drop_prob','0','--output_drop_prob','0','--max_answer_tokens','8',
          '--max_question_tokens','8','--train_eval_samples','0','--save_every_epoch']
    return get_args(argv)


def test_full_entrypoint_resumes_same_next_epoch(monkeypatch,tmp_path):
    args=fixture_args(monkeypatch,tmp_path)
    _run_training(args,torch.device('cpu'),0,1)
    directory=Path(args.model_path)
    expected=read_checkpoint(directory/'last.pt','cpu')
    assert expected['global_step']==4
    assert expected['bookkeeping'].keys()=={'best_val_loss','best_val_rouge'}
    assert (directory/'source_snapshot.zip').is_file()
    manifest=json.loads((directory/'run_config.json').read_text())
    assert manifest['global_batch_size']==4
    # Re-run epoch 2 from the saved epoch 1, preserving original schedule and path.
    resume_args=get_args(['--resume',str(directory/'epoch_0001.pt')])
    _run_training(resume_args,torch.device('cpu'),0,1)
    actual=read_checkpoint(directory/'last.pt','cpu')
    for key,value in expected['model_state_dict'].items():
        torch.testing.assert_close(value,actual['model_state_dict'][key],atol=0,rtol=0)
    assert expected['scheduler_state_dict']==actual['scheduler_state_dict']
    assert actual['global_step']==expected['global_step']
    # Fresh runs must not truncate the existing metrics/checkpoints.
    metrics=(directory/'metrics.jsonl').read_text()
    with pytest.raises(FileExistsError):
        _run_training(args,torch.device('cpu'),0,1)
    assert (directory/'metrics.jsonl').read_text()==metrics
    drift=get_args(['--resume',str(directory/'epoch_0001.pt'),'--epochs','3'])
    with pytest.raises(ValueError,match='epochs'):
        _run_training(drift,torch.device('cpu'),0,1)



def test_native_training_entrypoint_and_resume(monkeypatch,tmp_path):
    args=fixture_args(monkeypatch,tmp_path)
    args.model_family='qwen3_vl';args.amp='off';args.compute_bertscore=False
    args.batch_size=1;args.gradient_accumulation_steps=3
    args.figure_context=False;args.query_revision='b'*40;args.vlm_revision='a'*40
    args.visual_tokens=10;args.cost_dim=4;args.total_sequence_tokens=128
    args.max_answer_tokens=10;args.ot_num_heads=1
    from tests.test_native_preparation import tiny_wrapper
    from model.factory import native_config
    import model.factory as factory
    def build(args,device,resume=None):
        model=tiny_wrapper()
        model.figure_context=False
        model.model_config.update(resume['model_config'] if resume else native_config(args))
        model.model_config['figure_context']=False
        return model
    monkeypatch.setattr(factory,'build_native',build)
    _run_training(args,torch.device('cpu'),0,1)
    directory=Path(args.model_path)
    expected=read_checkpoint(directory/'last.pt','cpu')
    assert expected['state_kind']=='trainable_only'
    resumed=get_args(['--resume',str(directory/'epoch_0001.pt')])
    _run_training(resumed,torch.device('cpu'),0,1)
    actual=read_checkpoint(directory/'last.pt','cpu')
    assert expected['global_step']==actual['global_step']==4
    for key,value in expected['model_state_dict'].items():
        torch.testing.assert_close(value,actual['model_state_dict'][key],atol=0,rtol=0)
