"""Checks that reject an expensive run before avoidable failures."""
from argparse import Namespace
import copy
import json

import pandas as pd
import pytest
import torch
from PIL import Image

from configs.arg_parser import get_args
from scripts.check_t4_pilot import check_pilot
from scripts.kaggle_train import training_arguments, require_matching_pilot
from train import _make_loader
from utils.training_preflight import validate_training_data, validate_trainable_parameters


def launcher_args(**kwargs):
    values = dict(stage='pilot', data_root='/kaggle/input/scivqa/scivqa_dataset',
                  output_root='/kaggle/working', visual_tokens=256, seed=1105,
                  num_workers=2, log_every_updates=25, epochs=3, updates=100)
    values.update(kwargs)
    return Namespace(**values)


def valid_pilot():
    arguments = vars(get_args(training_arguments(launcher_args())))
    memory = {'name':'Tesla T4','usable_headroom_fraction':.15}
    rank = {'memory':memory,'overflow_skips':0,
            'gradient_samples':[{'norms':{k:.01 for k in ('gates','visual_cost','query_cost','query_value','outputs','lora_')}}]}
    manifest = {'arguments':arguments,'world_size':2,'global_batch_size':16,'dev_examples':1680,
                'model_config':{'lora_rank':16,'quantize_4bit':True,'adapter_control':'ot','plan_control':'learned',
                                'quantization_config':{'llm_int8_skip_modules':['model.visual','lm_head']}}}
    rows = [{'global_step':100,'runtime_by_rank':[copy.deepcopy(rank),copy.deepcopy(rank)],
             'validation_performance':{'examples':1680},'validation_memory':memory,
             'val_loss':1.,'val_rouge_l_f1':.4}]
    return manifest,rows


def test_pilot_matches_full_recipe():
    pilot = get_args(training_arguments(launcher_args()))
    full = get_args(training_arguments(launcher_args(stage='train')))
    for name,value in vars(pilot).items():
        if name not in {'max_updates','profile_stages','model_path','explicit_args'}:
            assert value == getattr(full,name),name
    assert pilot.lora_rank == full.lora_rank == 16
    assert pilot.epochs == full.epochs == 3
    assert full.max_updates is None and not full.profile_stages
    assert full.batch_size * full.gradient_accumulation_steps * 2 == 16


def test_pilot_rejects_unmeasured_lora_gradients_and_external_memory():
    manifest,rows = valid_pilot()
    assert check_pilot(manifest,rows)['measured_resource_checks_passed']
    rows[0]['runtime_by_rank'][1]['gradient_samples'][0]['norms']['query_cost']=0.
    assert not check_pilot(manifest,rows)['checks']['gradient_flow_each_rank']
    manifest,rows=valid_pilot();manifest['model_config']['lora_rank']=0
    assert not check_pilot(manifest,rows)['checks']['pilot_has_same_lora_rank']
    manifest,rows=valid_pilot();rows[0]['runtime_by_rank'][1]['memory']['usable_headroom_fraction']=.03
    assert not check_pilot(manifest,rows)['checks']['train_memory_headroom']
    assert not check_pilot({},[])['measured_resource_checks_passed']


def test_trainable_device_and_scaler_dtype_checks():
    model=torch.nn.Linear(2,3)
    assert validate_trainable_parameters(model,torch.device('cpu'),'fp16')['legacy']==9
    with pytest.raises(ValueError,match='FP32'):
        validate_trainable_parameters(model.half(),torch.device('cpu'),'fp16')
    with pytest.raises(ValueError,match='wrong device'):
        validate_trainable_parameters(model,torch.device('cuda',0),'off')


def test_data_validation_and_spawn_worker(tmp_path):
    root=tmp_path/'images';root.mkdir()
    Image.new('RGB',(8,12)).save(root/'figure.png')
    frame=pd.DataFrame({'anno_id':[1,2],'image':['figure.png']*2,'question':['q']*2,'answer':['a']*2})
    for name in ('train','val'):
        frame.to_csv(tmp_path/f'{name}.csv',index=False)
    args=get_args(['--dataset','gqa','--train_csv_path',str(tmp_path/'train.csv'),
                  '--dev_csv_path',str(tmp_path/'val.csv'),'--img_path',str(root),'--num_workers','1'])
    report=validate_training_data(args,2,verify_images=True)
    assert report['train']['examples']==2
    loader=_make_loader(args,'train',False,'x','y',native=True)
    assert loader.multiprocessing_context.get_start_method()=='spawn'
    assert len(next(iter(loader))[1])==2
    args.img_path=str(tmp_path/'missing')
    with pytest.raises(FileNotFoundError,match='missing images'):
        validate_training_data(args,2)



def test_nf4_excludes_native_visual_subtree_with_pinned_transformers():
    from transformers.integrations.bitsandbytes import should_convert_module
    from model.pretrained_vlm import QUANTIZATION_SKIP_MODULES
    from tests.test_native_qwen import tiny_qwen
    native=tiny_qwen()
    visual=[name for name,module in native.named_modules() if name.startswith('model.visual.') and type(module) is torch.nn.Linear]
    language=[name for name,module in native.named_modules() if '.self_attn.' in name and type(module) is torch.nn.Linear and name.startswith('model.language_model.')]
    assert visual and language
    assert all(not should_convert_module(name,list(QUANTIZATION_SKIP_MODULES)) for name in visual)
    assert all(should_convert_module(name,list(QUANTIZATION_SKIP_MODULES)) for name in language)
    assert not should_convert_module('lm_head',list(QUANTIZATION_SKIP_MODULES))
    # The old leaf name silently included the visual subtree in quantization.
    assert should_convert_module(visual[0],['visual','lm_head'])



def test_full_run_rejects_changed_pilot_code_or_budget(tmp_path):
    import hashlib
    from scripts.kaggle_train import ROOT
    from train import _csv_digest
    manifest,rows=valid_pilot()
    source=[ROOT/'train.py', ROOT/'test.py', *sorted((ROOT/'model').glob('*.py')),
            *sorted((ROOT/'utils').glob('*.py')), *sorted((ROOT/'configs').glob('*.py'))]
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    resolved=get_args(training_arguments(launcher_args(stage='train')))
    # Tiny files are sufficient for data-hash matching in this gate fixture.
    for split in ['train','dev']:
        csv=tmp_path/f'{split}.csv';csv.write_text('fixture')
        setattr(resolved,f'{split}_csv_path',str(csv))
        manifest[f'{split}_csv_sha256']=_csv_digest(csv)
    (tmp_path/'metrics.jsonl').write_text(json.dumps(rows[0])+'\n')
    def write_manifest():
        (tmp_path/'run_config.json').write_text(json.dumps(manifest))
    write_manifest()
    assert require_matching_pilot(tmp_path,resolved)['measured_resource_checks_passed']
    resolved.visual_tokens=512
    with pytest.raises(ValueError,match='visual_tokens'):
        require_matching_pilot(tmp_path,resolved)
    resolved.visual_tokens=256;manifest['source_sha256']['model/pretrained_vlm.py']='changed'
    write_manifest()
    with pytest.raises(ValueError,match='source differs'):
        require_matching_pilot(tmp_path,resolved)



def test_memory_headroom_accounts_for_other_cuda_allocations(monkeypatch):
    from utils.training_runtime import cuda_memory_report
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:None)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(200,1600))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:800)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda device:1000)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device:900)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda device:'Tesla T4')
    report=cuda_memory_report(torch.device('cuda',0))
    assert report['reserved_headroom_fraction']==.375
    assert report['usable_headroom_fraction']==0
