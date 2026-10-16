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
    for option,value in [('--epochs','1'),('--lr','0.2')]:
        drift=get_args(['--resume',str(directory/'epoch_0001.pt'),option,value])
        with pytest.raises(ValueError,match=option):
            _run_training(drift,torch.device('cpu'),0,1)



def native_fixture_args(monkeypatch,tmp_path,lora=False):
    args=fixture_args(monkeypatch,tmp_path)
    args.model_family='qwen3_vl';args.amp='off';args.compute_bertscore=False
    args.batch_size=1;args.gradient_accumulation_steps=3
    args.figure_context=False;args.query_revision='b'*40;args.vlm_revision='a'*40
    args.visual_tokens=10;args.cost_dim=4;args.total_sequence_tokens=128
    args.max_answer_tokens=10;args.ot_num_heads=1
    args.lora_rank=2 if lora else 0
    from tests.test_native_preparation import tiny_wrapper
    from model.factory import native_config
    import model.factory as factory
    def build(args,device,resume=None):
        model=tiny_wrapper(lora=lora)
        model.figure_context=False
        model.model_config.update(resume['model_config'] if resume else native_config(args))
        model.model_config['figure_context']=False
        return model
    monkeypatch.setattr(factory,'build_native',build)
    return args


def test_native_training_entrypoint_and_resume(monkeypatch,tmp_path):
    args=native_fixture_args(monkeypatch,tmp_path)
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


@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('source_epoch',[1,3])
def test_resume_extends_three_to_twenty_epochs(monkeypatch,tmp_path,native,source_epoch):
    import train as training
    args=(native_fixture_args(monkeypatch,tmp_path,lora=True) if native
          else fixture_args(monkeypatch,tmp_path))
    args.epochs=3;args.warmup_ratio=.05;args.early_stopping_patience=0
    _run_training(args,torch.device('cpu'),0,1)
    directory=Path(args.model_path)
    completed=read_checkpoint(directory/'last.pt','cpu')
    assert all(group['lr']==0 for group in completed['optimizer_state_dict']['param_groups'])
    source=directory/f'epoch_{source_epoch:04d}.pt'
    before=read_checkpoint(source,'cpu')
    real_train=training.train
    offsets=[]

    def recording_train(model,loader,count,optimizer,scheduler,criterion,**kwargs):
        offsets.append(kwargs['epoch_offset'])
        if len(offsets)==1:
            # Extension retains weights and Adam moments before the first update.
            for key,value in before['model_state_dict'].items():
                torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)
            torch.testing.assert_close(optimizer.state_dict()['state'],
                                       before['optimizer_state_dict']['state'],atol=0,rtol=0)
            assert scheduler.last_epoch==before['global_step']
            # Two updates/epoch: 40 total, two warmup updates under the new target.
            for group in optimizer.param_groups:
                assert group['lr']==pytest.approx(group['initial_lr']*(40-before['global_step'])/38)
                assert group['lr']>0
        return real_train(model,loader,count,optimizer,scheduler,criterion,**kwargs)

    monkeypatch.setattr(training,'train',recording_train)
    extended=get_args(['--resume',str(source),'--epochs','20','--save_every_epoch'])
    _run_training(extended,torch.device('cpu'),0,1)
    expected=read_checkpoint(directory/'last.pt','cpu')
    assert offsets==list(range(source_epoch,20))
    assert expected['epoch']==20 and expected['global_step']==40
    assert expected['training_config']['arguments']['epochs']==20
    assert expected['scheduler_state_dict']['last_epoch']==40
    assert all(group['lr']==0 for group in expected['optimizer_state_dict']['param_groups'])
    assert any(not torch.equal(value,expected['model_state_dict'][key])
               for key,value in before['model_state_dict'].items())

    # After an extension, ordinary resume must inherit 20 and reproduce the run.
    monkeypatch.setattr(training,'train',real_train)
    resumed=get_args(['--resume',str(directory/f'epoch_{source_epoch+1:04d}.pt')])
    _run_training(resumed,torch.device('cpu'),0,1)
    actual=read_checkpoint(directory/'last.pt','cpu')
    assert actual['training_config']['arguments']['epochs']==20
    torch.testing.assert_close(expected['model_state_dict'],actual['model_state_dict'],atol=0,rtol=0)
    torch.testing.assert_close(expected['optimizer_state_dict']['state'],
                               actual['optimizer_state_dict']['state'],atol=0,rtol=0)
    assert expected['scheduler_state_dict']==actual['scheduler_state_dict']
