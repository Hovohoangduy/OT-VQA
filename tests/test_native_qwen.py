"""Tiny randomly initialized *real* native Qwen, no pretrained weight download."""
from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig

from model.pretrained_vlm import ConditionedVision, VisualOTContextualizer, PretrainedVLM
from utils.losses import AnswerCrossEntropyLoss


def tiny_qwen():
    text = Qwen3VLTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
                num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                pad_token_id=0,rope_parameters={'rope_type':'default','rope_theta':10000.,'mrope_section':[2,1,1]})
    vision = Qwen3VLVisionConfig(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,
                 patch_size=2,temporal_patch_size=2,spatial_merge_size=2,out_hidden_size=32,
                 num_position_embeddings=16,deepstack_visual_indexes=[0,1,2])
    config = Qwen3VLConfig(text_config=text,vision_config=vision,image_token_id=60,
                           video_token_id=61,vision_start_token_id=62,vision_end_token_id=63,
                           eos_token_id=2,pad_token_id=0)
    return Qwen3VLForConditionalGeneration(config).eval()


def inputs():
    return {'input_ids':torch.tensor([[0,0,1,62,60,60,60,60,63,4],
                                      [1,62,60,60,60,60,60,60,63,5]]),
            'attention_mask':torch.tensor([[0,0,1,1,1,1,1,1,1,1],[1]*10]),
            'mm_token_type_ids':torch.tensor([[0,0,0,0,1,1,1,1,0,0],[0,0,1,1,1,1,1,1,0,0]]),
            'pixel_values':torch.randn(40,24),
            'image_grid_thw':torch.tensor([[1,4,4],[1,4,6]])}


@pytest.mark.parametrize("device_type", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_native_noop_logits_generation_cache_and_query_reset(device_type):
    torch.manual_seed(4)
    base = tiny_qwen().to(device_type)
    batch = {k:v.to(device_type) for k,v in inputs().items()}
    with torch.no_grad():
        expected = base(**batch,use_cache=False).logits
        ids_expected = base.generate(**batch,max_new_tokens=3,do_sample=False,use_cache=True)
    adapter = VisualOTContextualizer(32,8,4,cost_dim=4).to(device_type)
    base.requires_grad_(False)
    base.model.visual = ConditionedVision(base.model.visual,adapter)
    query, mask = torch.randn(2,5,8,device=device_type), torch.ones(2,5,dtype=torch.bool,device=device_type)
    with base.model.visual.conditioning(query,mask),torch.no_grad():
        actual = base(**batch,use_cache=False).logits
        ids_actual = base.generate(**batch,max_new_tokens=3,do_sample=False,use_cache=True)
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    assert torch.equal(ids_expected,ids_actual)
    # one vision forward per request, not one per generated token
    assert base.model.visual.calls == 2
    assert base.model.visual.request is None
    with pytest.raises(RuntimeError):
        with base.model.visual.conditioning(query,mask):
            raise RuntimeError('failed request')
    assert base.model.visual.request is None


@pytest.mark.parametrize("device_type", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_native_frozen_llm_backprop_checkpointing_lora_and_sparse_logits(device_type):
    torch.manual_seed(12)
    base = tiny_qwen().to(device_type); base.requires_grad_(False)
    from peft import LoraConfig,get_peft_model
    base = get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
        target_modules=r'model.language_model.layers.\d+.self_attn.(q_proj|k_proj|v_proj|o_proj)',
        task_type='CAUSAL_LM'))
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    adapter = VisualOTContextualizer(32,8,4,cost_dim=4).to(device_type)
    adapter.gates.data.fill_(.1)
    core = base.get_base_model().model
    core.visual = ConditionedVision(core.visual,adapter)
    base.train()
    batch = {k:v.to(device_type) for k,v in inputs().items()}; query = torch.randn(2,5,8,device=device_type); mask = torch.ones(2,5,dtype=torch.bool,device=device_type)
    labels = torch.full_like(batch['input_ids'],-100); labels[:,-2:] = torch.tensor([6,2],device=device_type)
    with core.visual.conditioning(query,mask):
        full = base(**batch,use_cache=False,labels=labels)
        positions = labels[:,1:].ne(-100).any(0).nonzero().flatten()
        sparse = base(**batch,use_cache=False,logits_to_keep=positions)
        targets = labels[:,1:][:,positions]
        loss = AnswerCrossEntropyLoss(-100,normalization='token')(sparse.logits.transpose(1,2),targets)
        torch.testing.assert_close(full.loss,loss)
        grads = torch.autograd.grad(full.loss,list(adapter.parameters()),retain_graph=True,allow_unused=True)
        loss.backward()
    for expected,parameter in zip(grads,adapter.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(expected,parameter.grad,atol=1e-6,rtol=1e-4)
    assert adapter.visual_cost.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in base.named_parameters() if 'lora_' in n)
    assert all(p.grad is None for n,p in core.visual.base.named_parameters())



if torch.cuda.is_available():
    def test_cuda_fp16_native_lora_adapter_optimizer_boundary():
        device_type="cuda"
        from peft import LoraConfig,get_peft_model
        from utils.training_runtime import optimizer_update
        base=tiny_qwen().to(device_type,dtype=torch.float16)
        base.requires_grad_(False)
        base=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
            target_modules=r'model.language_model.layers.\d+.self_attn.(q_proj|k_proj|v_proj|o_proj)',
            task_type='CAUSAL_LM'))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        adapter=VisualOTContextualizer(32,8,4,cost_dim=4).to(device_type)
        adapter.gates.data.fill_(.1)
        core=base.get_base_model().model
        core.visual=ConditionedVision(core.visual,adapter)
        base.train()
        batch={k:v.to(device_type) for k,v in inputs().items()}
        query=torch.randn(2,5,8,device=device_type)
        mask=torch.ones(2,5,dtype=torch.bool,device=device_type)
        opt=torch.optim.AdamW([p for p in base.parameters() if p.requires_grad],lr=1e-3)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        with core.visual.conditioning(query,mask),torch.autocast('cuda',dtype=torch.float16):
            logits=base(**batch,use_cache=False,logits_to_keep=1).logits
            loss=torch.nn.functional.cross_entropy(logits[:,0],torch.tensor([6,2],device=device_type))
        scaler.scale(loss).backward()
        stats={}
        assert optimizer_update(base,opt,scheduler,scaler,1.,torch.device(device_type),stats)
        assert all(stats[k]>0 for k in ('visual_cost','query_cost','gates','lora_'))
        assert scheduler.last_epoch==1
