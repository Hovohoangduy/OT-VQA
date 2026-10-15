from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from train import train
from utils.losses import AnswerCrossEntropyLoss
from utils.training_runtime import optimizer_update, capture_rng, restore_rng
from utils.checkpoint import checkpoint_payload, restore_training_state, read_checkpoint, load_model_state


class ToyEmbedding:
    def tokenize(self, answers, max_len):
        return torch.tensor([[1]+[int(v) for v in a.split()] for a in answers])


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,7)
        self.answer_embedding = ToyEmbedding()
        self.max_answer_tokens = 4
        self.pad_token_id = 0

    def forward(self, images, questions, answers, anno_ids, return_ot_loss=False):
        logits = self.linear(images)
        targets = self.answer_embedding.tokenize(answers,4)[:,1:].to(logits.device)
        aux = logits.square().mean((1,2)).mean()
        return (logits,targets,aux) if return_ot_loss else (logits,targets)


def rows(count=7):
    rng = torch.Generator().manual_seed(15)
    return [(i,torch.randn(3,4,generator=rng),'q',a) for i,a in enumerate(
        ['2 0 0','3 4 5','6 2 0','2 3 0','4 5 6','5 0 0','3 6 0'][:count])]


@pytest.mark.parametrize('normalization',['example','token'])
@pytest.mark.parametrize('aux',[0.,.3])
@pytest.mark.parametrize('device_type',['cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else []))
def test_accumulation_matches_large_batch_and_short_group(normalization,aux,device_type):
    torch.manual_seed(3)
    device=torch.device(device_type)
    a,b = ToyModel().to(device),ToyModel().to(device); b.load_state_dict(a.state_dict())
    criterion = AnswerCrossEntropyLoss(0,label_smoothing=.1,normalization=normalization)
    outputs = []
    for model,batch,accum in ((a,6,1),(b,2,3)):
        opt = torch.optim.SGD(model.parameters(),lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        out = train(model,DataLoader(rows(),batch_size=batch),1,opt,scheduler,criterion,
                    gradient_accumulation_steps=accum,ot_loss_weight=aux,device=device)
        outputs.append(out)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)
    assert outputs[0][3]['optimizer_updates'] == outputs[1][3]['optimizer_updates'] == 2
    assert outputs[0][1] == pytest.approx(outputs[1][1],abs=1e-6)


def test_overflow_skips_optimizer_and_scheduler_then_recovers():
    model = nn.Linear(2,1)
    opt = torch.optim.SGD(model.parameters(),lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
    scaler = torch.amp.GradScaler('cpu')
    before = copy.deepcopy(model.state_dict())
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    model.weight.grad.fill_(float('inf'))
    assert not optimizer_update(model,opt,scheduler,scaler,1.,torch.device('cpu'))
    assert scheduler.last_epoch == 0
    for name,p in model.state_dict().items():
        assert torch.equal(p,before[name])
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    assert optimizer_update(model,opt,scheduler,scaler,1.,torch.device('cpu'))
    assert scheduler.last_epoch == 1


def test_rng_roundtrip():
    import random
    import numpy as np
    state = capture_rng()
    a = (random.random(),np.random.rand(),torch.rand(2))
    restore_rng(state)
    b = (random.random(),np.random.rand(),torch.rand(2))
    assert a[:2] == b[:2]
    assert torch.equal(a[2],b[2])


def ddp_worker(rank,init_path,result_path,normalization,device_type):
    torch.set_num_threads(1)
    device=torch.device('cuda',rank) if device_type=='cuda' else torch.device('cpu')
    if device_type=='cuda':
        torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if device_type=='cuda' else 'gloo',init_method='file://'+init_path,rank=rank,world_size=2)
    try:
        torch.manual_seed(3)
        base = ToyModel().to(device); model = DistributedDataParallel(base,device_ids=[rank] if device_type=="cuda" else None)
        opt = torch.optim.SGD(model.parameters(),lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        subset = rows(6)[rank::2]
        train(model,DataLoader(subset,batch_size=2),1,opt,scheduler,
              AnswerCrossEntropyLoss(0,normalization=normalization),gradient_accumulation_steps=2,ot_loss_weight=.3,device=device)
        if rank == 0:
            torch.save(base.state_dict(),result_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('normalization',['example','token'])
@pytest.mark.parametrize('device_type',['cpu']+(['cuda'] if torch.cuda.device_count()>=2 else []))
def test_two_rank_ddp_matches_single_global_batch(tmp_path,normalization,device_type):
    torch.manual_seed(3)
    device=torch.device(device_type)
    model = ToyModel().to(device); opt = torch.optim.SGD(model.parameters(),lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
    train(model,DataLoader(rows(6),batch_size=6),1,opt,scheduler,
          AnswerCrossEntropyLoss(0,normalization=normalization),ot_loss_weight=.3,device=device)
    path = tmp_path/'ddp.pt'
    mp.spawn(ddp_worker,args=(str(tmp_path/'init'),str(path),normalization,device_type),nprocs=2,join=True)
    actual = torch.load(path,weights_only=True,map_location=device)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,actual[key],atol=1e-7,rtol=1e-6)
