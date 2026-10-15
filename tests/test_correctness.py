import pandas as pd
import pytest
import torch
from torch import nn

from configs.arg_parser import get_args
from model.optimal_transport import PartialTransportFusion, compute_fgw_cost, independent_coupling, log_sinkhorn
from model.vqa_model import VQAModel
from model.pretrained_vlm import answer_labels, VisualOTContextualizer
from utils.vqa_dataset import build_question, parse_bool
from utils.metrics import scivqa_score_pairs
from utils.quality_evidence import guardrail_metrics, paired_cluster_interval


def test_prompt_count_and_boolean():
    for false in (False, 'False', 'false', '', 0, float('nan')):
        assert not parse_bool(false)
    prompt = build_question('Read panel (a), lower left.', compound='True', figs_numb=3,
                            caption='Caption', figure_type='line chart', figure_context=True)
    assert 'contains 3 subfigures' in prompt
    assert '(a), lower left' in prompt
    assert 'Navigate to subfigure 3' not in prompt
    assert 'Subfigure navigation' not in build_question('q', compound='False', figs_numb=3,
                                                       caption='c', figure_type='x', figure_context=True)
    assert 'Navigate to subfigure 3' in build_question('q', compound=True, figs_numb=3,
                caption='c', figure_type='x', figure_context=True, prompt_version='scivqa-v1')


@pytest.mark.parametrize('kind', ['infinite answer set non-visual', 'infinite answer set visual', 'unanswerable'])
def test_prompt_no_numeric_or_answerability_oracle(kind):
    q = 'Which method achieves the maximum score?'
    assert build_question(q, qa_pair_type=kind) == q


def test_parser_inheritance_and_native_recipe():
    for args in (get_args(['--dataset', 'scivqa'], evaluation=True),
                 get_args(['--dataset', 'scivqa', '--resume', 'x.pt'])):
        for key in ('figure_context', 'fgw_alpha', 'ot_loss_weight', 'max_question_tokens', 'ot_num_heads'):
            assert getattr(args, key) is None
    args = get_args(['--dataset', 'scivqa', '--model_family', 'qwen3_vl'])
    assert (args.batch_size, args.gradient_accumulation_steps, args.ot_loss_weight, args.label_smoothing) == (1, 8, 0, 0)


def constraint_model(version):
    model = VQAModel.__new__(VQAModel)
    nn.Module.__init__(model)
    model.decode_version = version
    model.eos_token_id = 2
    model.numeric_token_ids = {3, 4}
    model.yes_token_id, model.no_token_id = 5, 6
    model.option_token_ids = {'A': 7, 'B': 8, 'C': 9, 'D': 10}
    model.comma_token_id = 11
    return model


def test_numeric_tokens_unrestricted_binary_and_multilabel_preserved():
    model = constraint_model('task-v2')
    questions = ['Answer with only the raw numerical value.']
    ids = torch.tensor([[1]])
    logits = torch.zeros(1, 40)
    assert torch.isfinite(model._apply_step_constraints(logits.clone(), ids, 0, questions)).all()
    assert not torch.isfinite(constraint_model('task-v1')._apply_step_constraints(logits.clone(), ids, 0, questions)[0, 20])
    binary = model._apply_step_constraints(logits.clone(), ids, 0, ["Answer with 'Yes' or 'No'."])
    assert binary.isfinite().nonzero()[:, 1].tolist() == [5, 6]
    mc = ['Return only the letter(s). Answer options: A: x; B: y; C: z; D: w']
    for step, tokens, allowed in [(0, [1], [7,8,9,10]), (1, [1,7], [2,11]),
                                   (2, [1,7,11], [8,9,10])]:
        out = model._apply_step_constraints(logits.clone(), torch.tensor([tokens]), step, mc)
        assert out.isfinite().nonzero()[:,1].tolist() == allowed


def test_native_labels_preserve_eos_equal_pad_and_no_double_shift():
    ids, labels = answer_labels([1, 8, 8, 4], [5, 6], 2, 4, 20)
    assert ids == [1, 8, 8, 4, 5, 6, 2]
    assert labels == [-100, -100, -100, -100, 5, 6, 2]
    with pytest.raises(ValueError):
        answer_labels([1]*20, [5], 2, 4, 20)
    with pytest.raises(ValueError):
        answer_labels([1], [5]*5, 2, 4, 20)


@pytest.mark.parametrize('heads,visual', [(1,False), (2,True)])
def test_ot_padding_fp32_and_gradients(heads, visual):
    torch.manual_seed(7)
    fusion = PartialTransportFusion(8, 8, num_heads=heads, visual_memory=visual, iterations=100, epsilon=.2)
    v, q = torch.randn(2,4,8,requires_grad=True), torch.randn(2,5,8,requires_grad=True)
    mask = torch.tensor([[1,1,0,0,0],[1,1,1,1,0]], dtype=torch.bool)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        memory, blocked, diag = fusion(v,q,mask,return_transport=True)
    assert diag['plan'].dtype == torch.float32
    changed = q.detach().clone(); changed[~mask] = 100
    memory2, blocked2 = fusion(v,changed,mask)
    torch.testing.assert_close(memory, memory2)
    assert torch.equal(blocked, blocked2)
    assert diag['row_residual'].max() < 2e-4
    assert (diag['matched_mass'] >= .6-2e-4).all()
    memory.square().mean().backward()
    assert torch.isfinite(v.grad).all() and torch.isfinite(q.grad).all()
    assert q.grad[~mask].abs().max() == 0


def test_fgw_padding_invariance():
    torch.manual_seed(1)
    cost = torch.rand(1,4,3); plan = torch.rand(1,4,3); mask = torch.tensor([[1,1,0]],dtype=torch.bool)
    plan[:,:,2] = 0
    a = compute_fgw_cost(cost,4,3,plan,alpha=.5,question_mask=mask)
    b = compute_fgw_cost(torch.nn.functional.pad(cost,(0,4)),4,7,torch.nn.functional.pad(plan,(0,4)),
                         alpha=.5,question_mask=torch.nn.functional.pad(mask,(0,4)))
    torch.testing.assert_close(a[:,:,:2], b[:,:,:2])


def test_feasible_independent_control():
    cost = torch.rand(2,5,6)
    r = torch.ones(2,5)/5; c = torch.tensor([[.2,.2,.2,.2,0,.2]]).expand(2,-1)
    plan = log_sinkhorn(cost,r,c,iterations=100)
    control = independent_coupling(plan)
    torch.testing.assert_close(plan.sum(-1),control.sum(-1))
    torch.testing.assert_close(plan.sum(-2),control.sum(-2))
    torch.testing.assert_close(plan[:,-1],control[:,-1])
    torch.testing.assert_close(plan[:,:,-1],control[:,:,-1])
    assert not control[:,:,4].any()


def test_adapter_zero_gate_then_cost_gradients_and_deepstack():
    torch.manual_seed(3)
    adapter = VisualOTContextualizer(16,8,4,cost_dim=4,iterations=50)
    main = torch.randn(10,16); deep = [torch.randn_like(main) for _ in range(3)]
    query = torch.randn(2,5,8); mask = torch.ones(2,5,dtype=torch.bool)
    result, stack = adapter(main,deep,query,mask,[4,6])
    assert torch.equal(result,main)
    assert all(torch.equal(a,b) for a,b in zip(stack,deep))
    sum(x.square().mean() for x in [result,*stack]).backward()
    assert (adapter.gates.grad.abs() > 0).all()
    adapter.zero_grad(); adapter.gates.data.fill_(.1)
    result, stack = adapter(main,deep,query,mask,[4,6])
    sum(x.square().mean() for x in [result,*stack]).backward()
    for module in (adapter.visual_cost,adapter.query_cost,adapter.query_value,*adapter.outputs):
        assert module.weight.grad.isfinite().all() and module.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        adapter(main,deep[:-1],query,mask,[4,6])


def test_official_metric_sign_guardrail_and_cluster_ci():
    rows = scivqa_score_pairs(['-12', 'running'],['12','runs'],compute_bertscore=False)
    assert rows[0]['rouge_l_f1'] == 1
    assert guardrail_metrics(['-12'],['12'])['numeric_exact'] == 0
    interval = paired_cluster_interval([.1,.1,.2,.2],['a','a','b','b'],resamples=100)
    assert interval['clusters'] == 2 and interval['delta'] == pytest.approx(.15)


@pytest.mark.parametrize('device_type', ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []) + (['cuda'] if torch.cuda.is_available() else []))
def test_ot_available_device_backward(device_type):
    device=torch.device(device_type)
    module=PartialTransportFusion(8,8,epsilon=.2,iterations=100).to(device)
    v=torch.randn(2,4,8,device=device,requires_grad=True)
    q=torch.randn(2,3,8,device=device,requires_grad=True)
    mask=torch.tensor([[1,1,0],[1,1,1]],device=device,dtype=torch.bool)
    memory,blocked,diag=module(v,q,mask,return_transport=True)
    (memory.square().mean()+diag['ot_loss']).backward()
    assert torch.isfinite(memory).all() and torch.isfinite(v.grad).all() and torch.isfinite(q.grad).all()


def test_choices_builder_shared_and_preserves_question():
    options='[{"A":"red"},{"B":"blue"}]'
    q=build_question('Which panel (b)?',answer_options=options,qa_prompt=False)
    assert q=='Which panel (b)?\nAnswer options: A: red; B: blue'
    assert build_question(q,answer_options=options,qa_prompt=False)==q
