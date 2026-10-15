from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn

from tests.test_native_qwen import tiny_qwen
from model.pretrained_vlm import PretrainedVLM
from utils.checkpoint import save_checkpoint, read_checkpoint, restore_training_state, load_model_state
from utils.training_runtime import capture_rng
from utils.losses import AnswerCrossEntropyLoss


class Encoding(dict):
    def __getattr__(self,name):
        return self[name]


class Tokenizer:
    pad_token_id = eos_token_id = 2
    padding_side = 'left'
    def __call__(self,texts,add_special_tokens=True,**kwargs):
        rows = [[3+ord(c)%20 for c in text] for text in texts]
        if kwargs.get('return_tensors'):
            rows = [[1]+r+[2] for r in rows]
            n = max(map(len,rows))
            ids = torch.tensor([r+[2]*(n-len(r)) for r in rows])
            mask = torch.tensor([[1]*len(r)+[0]*(n-len(r)) for r in rows])
            special = torch.ones_like(ids)
            for i,r in enumerate(rows):
                special[i,1:len(r)-1]=0
            return Encoding(input_ids=ids,attention_mask=mask,special_tokens_mask=special)
        return Encoding(input_ids=rows)
    def batch_decode(self,ids,**kwargs):
        return ['x' for _ in ids]


class Processor:
    def __init__(self):
        self.tokenizer=Tokenizer()
    def apply_chat_template(self,conversation,**kwargs):
        return conversation[0]['content'][1]['text']
    def __call__(self,text,images,**kwargs):
        lengths = [4,6][:len(images)]
        prefixes = [[1,62]+[60]*n+[63,4] for n in lengths]
        n = max(map(len,prefixes))
        return Encoding(input_ids=torch.tensor([[2]*(n-len(r))+r for r in prefixes]),
                        attention_mask=torch.tensor([[0]*(n-len(r))+[1]*len(r) for r in prefixes]),
                        pixel_values=torch.cat([image if torch.is_tensor(image) else torch.full((n*4,24),.5)
                                                for image,n in zip(images,lengths)]),
                        image_grid_thw=torch.tensor([[1,4,4],[1,4,6]][:len(images)]))


class QueryEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(64,8)
        self.config=SimpleNamespace(hidden_size=8)
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def tiny_wrapper(lora=False):
    base=tiny_qwen();base.requires_grad_(False)
    if lora:
        from peft import LoraConfig,get_peft_model
        base=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
             target_modules=r'model.language_model.layers.\d+.self_attn.(q_proj|k_proj|v_proj|o_proj)',
             task_type='CAUSAL_LM'))
    return PretrainedVLM(base,Processor(),QueryEncoder(),Tokenizer(),
        {'vlm_revision':'a'*40,'query_revision':'b'*40,'query_tokens':256,
         'total_sequence_tokens':128,'visual_tokens':10,'max_answer_tokens':10,'cost_dim':4})


def test_prepared_labels_query_independence_and_request_cleanup():
    model=tiny_wrapper()
    images=[torch.randn(16,24),torch.randn(24,24)]
    questions=[{'prompt':'long caption here','query':'read a'}, {'prompt':'other caption','query':'read b'}]
    prepared=model.prepare_batch(images,questions,['-12','AB'])
    other=model.prepare_batch(images,questions,['99','CD'])
    assert torch.equal(prepared['query_inputs']['input_ids'],other['query_inputs']['input_ids'])
    assert prepared['inputs']['mm_token_type_ids'].shape == prepared['labels'].shape
    assert torch.equal(prepared['labels'][:,-1],torch.tensor([2,2]))
    logits,targets=model(prepared=prepared)
    assert targets.ne(-100).sum() == 7
    loss=AnswerCrossEntropyLoss(-100,normalization='example')(logits.transpose(1,2),targets)
    loss.backward()
    assert model.adapter.gates.grad.abs().sum()>0
    assert model.core.visual.request is None and model.core.rope_deltas is None
    assert model.adapter.ot_loss is None
    before=model.core.visual.calls
    model.generate(images,questions,max_len=3)
    assert model.core.visual.calls == before+1
    # Failed preparation is rejected before any request state is created.
    with pytest.raises(ValueError):
        model.prepare_batch(images,questions,['x'*100,'y'])
    assert model.core.visual.request is None


@pytest.mark.parametrize("lora", [False, True])
def test_adapter_checkpoint_prediction_and_next_update_roundtrip(tmp_path,lora):
    torch.manual_seed(22)
    model=tiny_wrapper(lora=lora); pristine=copy.deepcopy(model)
    images=[torch.randn(16,24),torch.randn(24,24)]
    questions=[{'prompt':'p','query':'q'},{'prompt':'p2','query':'q2'}]
    prepared=model.prepare_batch(images,questions,['a','bc'])
    criterion=AnswerCrossEntropyLoss(-100,normalization='example')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    def update(m,o,s):
        o.zero_grad();logits,targets=m(prepared=prepared)
        loss=criterion(logits.transpose(1,2),targets);loss.backward();o.step();s.step()
    update(model,opt,sched)
    path=tmp_path/'adapter.pt'
    save_checkpoint(path,model=model,text_model='q',image_model='v',optimizer=opt,scheduler=sched,
                    scaler=scaler,rng_by_rank=[capture_rng()],epoch=1,global_step=1)
    checkpoint=read_checkpoint(path,'cpu')
    assert checkpoint['state_kind']=='trainable_only'
    assert all('adapter.' in k or 'lora_' in k for k in checkpoint['model_state_dict'])
    assert any('lora_' in k for k in checkpoint['model_state_dict']) == lora
    assert path.stat().st_size < 200000
    new_opt=torch.optim.AdamW([p for p in pristine.parameters() if p.requires_grad],lr=.01)
    new_sched=torch.optim.lr_scheduler.LambdaLR(new_opt,lambda _:1.)
    restore_training_state(checkpoint,pristine,new_opt,new_sched,scaler=scaler)
    model.eval();pristine.eval()
    torch.testing.assert_close(model(prepared=prepared)[0],pristine(prepared=prepared)[0],atol=0,rtol=0)
    assert torch.equal(model.generate(images,questions,max_len=2),pristine.generate(images,questions,max_len=2))
    update(model,opt,sched);update(pristine,new_opt,new_sched)
    for p,q in zip(model.parameters(),pristine.parameters()):
        torch.testing.assert_close(p,q,atol=0,rtol=0)
    corrupted=copy.deepcopy(checkpoint);corrupted['model_state_dict'].pop(next(iter(corrupted['model_state_dict'])))
    with pytest.raises(ValueError):
        load_model_state(corrupted,pristine)
