"""Legacy decoder/mask/checkpoint regression without network model downloads."""
from types import SimpleNamespace
import copy

import torch
from torch import nn

import model.vqa_model as vqa
from utils.checkpoint import save_checkpoint,load_model,read_checkpoint,restore_training_state
from utils.training_runtime import capture_rng
from utils.losses import AnswerCrossEntropyLoss


class Tokenizer:
    pad_token_id=0;bos_token_id=1;eos_token_id=2;cls_token_id=1;sep_token_id=2
    def encode(self,text,**kwargs):
        return [3+ord(c)%20 for c in text]
    def batch_decode(self,rows,**kwargs):
        return [' '.join(map(str,r)) for r in rows]


class ImageEmbedding(nn.Module):
    output_dim=8;has_cls_token=False
    def __init__(self,*args,**kwargs):
        super().__init__();self.model=nn.Linear(3,8)
        self.model.config=SimpleNamespace(_commit_hash='image')
    def forward(self,images,image_ids=None):
        return self.model(images.flatten(2).transpose(1,2)), image_ids


class QuestionEmbedding(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.text_encoder=nn.Embedding(32,8)
        self.text_encoder.config=SimpleNamespace(hidden_size=8,_commit_hash='text')
    def freeze(self):
        self.requires_grad_(False)
    def encode_tokens(self,questions):
        ids=torch.tensor([[3,4,0] for q in questions])
        values=self.text_encoder(ids)
        return values,ids.eq(0),None,values[:,:2].mean(1)


class AnswerEmbedding(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.token_embeddings=nn.Module()
        self.token_embeddings.word_embeddings=nn.Embedding(32,8)
        self.tokenizer=Tokenizer()
    def freeze(self):
        self.requires_grad_(False)
    def embed_ids(self,ids):
        return self.token_embeddings.word_embeddings(ids)
    def tokenize(self,answers,max_len):
        rows=[[1]+self.tokenizer.encode(a)+[2] for a in answers]
        size=max(map(len,rows))
        return torch.tensor([r+[0]*(size-len(r)) for r in rows])


def test_legacy_causal_padding_shift_and_checkpoint_roundtrip(monkeypatch,tmp_path):
    for name,cls in [('ImageEmbedding',ImageEmbedding),('QuestionEmbedding',QuestionEmbedding),('AnswerEmbedding',AnswerEmbedding)]:
        monkeypatch.setattr(vqa,name,cls)
    model=vqa.VQAModel(d_model=8,ffn_hidden=16,num_heads=2,num_layers=1,drop_prob=0,
                       output_drop_prob=0,tie_weights=True,text_model='fake',image_model='fake',max_answer_tokens=8)
    assert not any(isinstance(m,nn.Dropout) and m.p for m in model.modules())
    images=torch.rand(2,3,2,2);questions=['q','q2'];answers=['ab','c']
    model.eval();memory,blocked=model.encode(images,questions)
    a=model.decode(torch.tensor([[1,3,4],[1,5,6]]),memory,memory_blocked=blocked)
    b=model.decode(torch.tensor([[1,3,9],[1,5,8]]),memory,memory_blocked=blocked)
    torch.testing.assert_close(a[:,:2],b[:,:2],atol=0,rtol=0)
    modified=memory.clone();modified[blocked]=1000
    c=model.decode(torch.tensor([[1,3,4],[1,5,6]]),modified,memory_blocked=blocked)
    torch.testing.assert_close(a,c,atol=0,rtol=0)
    logits,targets=model(images,questions,answers)
    assert torch.equal(targets,model.answer_embedding.tokenize(answers,8)[:,1:])
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
    def update(m,o,s):
        o.zero_grad();l,t=m(images,questions,answers)
        AnswerCrossEntropyLoss(0,normalization='example')(l.transpose(1,2),t).backward();o.step();s.step()
    update(model,opt,scheduler)
    path=tmp_path/'legacy.pt'
    save_checkpoint(path,model=model,text_model='fake',image_model='fake',optimizer=opt,scheduler=scheduler,
                    epoch=1,global_step=1,rng_by_rank=[capture_rng()])
    restored=load_model(path,torch.device('cpu'))
    torch.testing.assert_close(model(images,questions,answers)[0],restored(images,questions,answers)[0],atol=0,rtol=0)
    assert torch.equal(model.generate(images,questions),restored.generate(images,questions))
    opt2=torch.optim.AdamW(restored.parameters(),lr=.001)
    sched2=torch.optim.lr_scheduler.LambdaLR(opt2,lambda _:1)
    restore_training_state(read_checkpoint(path,'cpu'),restored,opt2,sched2)
    update(model,opt,scheduler);update(restored,opt2,sched2)
    for p,q in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(p,q,atol=0,rtol=0)
