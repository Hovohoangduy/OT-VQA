"""Native Qwen3-VL with query-only partial OT on main and DeepStack features.

Integration contract is tested against transformers==5.5.0. Only this model's
visual module is wrapped; no global hooks or upstream methods are patched.
"""
from contextlib import contextmanager
import re

import torch
from torch import nn
from torch.nn import functional as F

from model.optimal_transport import independent_coupling, log_sinkhorn
from utils.answer_text import postprocess_answer
from utils.profiling import StageProfiler

TRANSFORMERS_VERSION = "5.5.0"
QUANTIZATION_SKIP_MODULES = ("model.visual", "lm_head")


class VisualOTContextualizer(nn.Module):
    def __init__(self, visual_dim, query_dim, levels, cost_dim=128, epsilon=0.05,
                 iterations=20, dustbin_mass=0.2, dustbin_cost=1.0,
                 control="ot", plan_control="learned"):
        super().__init__()
        if cost_dim < 1 or levels < 1 or not 0 <= dustbin_mass < 1:
            raise ValueError("Invalid OT dimensions or dustbin mass")
        if not 0 <= dustbin_cost <= 2:
            raise ValueError("dustbin_cost must be in [0, 2]")
        if control not in {"ot", "native", "pooled"} or plan_control not in {"learned", "independent"}:
            raise ValueError("Unknown experimental control")
        self.visual_cost = nn.Linear(visual_dim, cost_dim)
        self.query_cost = nn.Linear(query_dim, cost_dim)
        self.query_value = nn.Linear(query_dim, cost_dim)
        self.outputs = nn.ModuleList(nn.Linear(cost_dim, visual_dim) for _ in range(levels))
        self.gates = nn.Parameter(torch.zeros(levels))
        self.epsilon, self.iterations = epsilon, iterations
        self.dustbin_mass, self.dustbin_cost = dustbin_mass, dustbin_cost
        self.control, self.plan_control = control, plan_control
        self.diagnostics = {}

    def forward(self, main, deepstack, query, mask, split_sizes):
        values = [main, *deepstack]
        if len(values) != len(self.outputs) or any(x.shape != main.shape for x in values):
            raise ValueError("Main/DeepStack merged token layouts must agree")
        if sum(split_sizes) != main.size(0) or len(split_sizes) != query.size(0):
            raise ValueError("Exactly one image per question is required")
        if self.control == "native":
            self.diagnostics = {}
            return main, deepstack
        conditioned = [[] for _ in values]
        diagnostics = []
        with torch.autocast(device_type=main.device.type, enabled=False):
            for i, image in enumerate(main.split(split_sizes)):
                valid = mask[i].bool()
                if not valid.any():
                    raise ValueError("Query needs content tokens")
                q = query[i, valid].float()
                v = F.normalize(self.visual_cost(image.float()), dim=-1)
                qc = F.normalize(self.query_cost(q), dim=-1)
                qv = self.query_value(q)
                local = 1 - v @ qc.T
                d = self.dustbin_mass
                row = local.new_full((1, image.size(0) + 1), (1-d) / image.size(0))
                col = local.new_full((1, q.size(0) + 1), (1-d) / q.size(0))
                row[:, -1] = col[:, -1] = d
                cost = torch.cat((torch.cat((local, 1-v @ F.normalize(qc.mean(0), dim=0)[:, None]), -1),
                                  torch.cat((1-F.normalize(v.mean(0), dim=0)[None] @ qc.T,
                                             local.new_full((1, 1), self.dustbin_cost)), -1)), 0)[None]
                if self.control == "pooled":
                    # Capacity control: same trainable projections, pooled cross-modal
                    # gate and values, no pairwise transport solve.
                    relevance = torch.sigmoid((v.mean(0) * qc.mean(0)).sum())
                    aligned = qv.mean(0, keepdim=True).expand(image.size(0), -1) * relevance
                    diag = {"ot_loss": local.new_zeros(())}
                else:
                    plan = log_sinkhorn(cost, row, col, self.epsilon, self.iterations)
                    if self.plan_control == "independent":
                        plan = independent_coupling(plan)
                    local_plan = plan[0, :-1, :-1]
                    aligned = local_plan @ qv / local_plan.sum(-1, keepdim=True).clamp_min(1e-8)
                    diag = {"matched_mass": local_plan.sum(),
                            "row_residual": (plan.sum(-1)-row).abs().max(),
                            "column_residual": (plan.sum(-2)-col).abs().max(),
                            "ot_loss": (local_plan * local).sum()}
                diagnostics.append(diag)
                start = sum(split_sizes[:i])
                for level, value in enumerate(values):
                    delta = self.gates[level].tanh() * self.outputs[level](aligned)
                    conditioned[level].append(value[start:start+split_sizes[i]] + delta.to(value.dtype))
        # Keep only detached scalar telemetry, never an autograd graph across requests.
        self.diagnostics = {key: torch.stack([x[key] for x in diagnostics]).detach().mean()
                            for key in diagnostics[0] if key != "ot_loss"}
        self.ot_loss = torch.stack([x["ot_loss"] for x in diagnostics]).mean()
        outputs = [torch.cat(x, 0) for x in conditioned]
        return outputs[0], outputs[1:]


class ConditionedVision(nn.Module):
    """Preserve native merged order shared by the main and DeepStack mergers."""
    def __init__(self, base, adapter, profiler=None):
        super().__init__()
        self.base, self.adapter = base, adapter
        self.profiler = profiler or StageProfiler()
        self.request = None
        self.calls = 0

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def spatial_merge_size(self):
        return self.base.spatial_merge_size

    def forward(self, hidden_states, grid_thw, **kwargs):
        # Frozen feature extraction is the only no_grad region. Adapter + LLM
        # activations remain differentiable even with frozen LLM parameters.
        with self.profiler.measure("vision", hidden_states.device), torch.no_grad():
            out = self.base(hidden_states, grid_thw=grid_thw, **kwargs)
        if self.request is None:
            raise RuntimeError("Visual conditioning requires a request-local query")
        sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        with self.profiler.measure("ot", hidden_states.device):
            out.pooler_output, out.deepstack_features = self.adapter(
                out.pooler_output, out.deepstack_features, *self.request, sizes)
        self.calls += 1
        return out

    @contextmanager
    def conditioning(self, query, mask):
        if self.request is not None:
            raise RuntimeError("Concurrent requests on one model are unsupported")
        self.request = (query, mask)
        try:
            yield
        finally:
            self.request = None


def answer_labels(prefix_ids, answer_ids, eos_token_id, max_answer_tokens, total_cap):
    """Append unshifted answer+EOS; PAD may equal EOS, labels use positions."""
    if len(answer_ids) + 1 > max_answer_tokens:
        raise ValueError("Answer exceeds max_answer_tokens; refusing silent truncation")
    ids = list(prefix_ids) + list(answer_ids) + [eos_token_id]
    if len(ids) > total_cap:
        raise ValueError("Prompt/image/answer exceeds total sequence cap")
    labels = [-100] * len(prefix_ids) + list(answer_ids) + [eos_token_id]
    return ids, labels


class PretrainedVLM(nn.Module):
    model_family = "qwen3_vl"
    fusion = "ot"
    image_transform = None
    target_ignore_index = -100

    def __init__(self, base, processor, query_encoder, query_tokenizer, config):
        super().__init__()
        self.base, self.processor = base, processor
        self.query_encoder, self.query_tokenizer = query_encoder, query_tokenizer
        self.model_config = dict(config)
        self.tokenizer = processor.tokenizer
        self.figure_context = config.get("figure_context", True)
        self.prompt_version = config.get("prompt_version", "scivqa-v2")
        self.decode_version = "native-greedy-v1"
        self.max_answer_tokens = config.get("max_answer_tokens", 64)
        self.max_question_tokens = config.get("query_tokens", 256)
        self.image_size, self.image_preprocessing = None, "native"
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.query_encoder.requires_grad_(False).eval()
        core = self.core
        adapter = VisualOTContextualizer(
            base.config.text_config.hidden_size, query_encoder.config.hidden_size,
            1 + len(base.config.vision_config.deepstack_visual_indexes),
            cost_dim=config.get("cost_dim", 128), epsilon=config.get("ot_epsilon", .05),
            iterations=config.get("ot_iterations", 20), dustbin_mass=config.get("ot_dustbin_mass", .2),
            dustbin_cost=config.get("ot_dustbin_cost", 1.), control=config.get("adapter_control", "ot"),
            plan_control=config.get("plan_control", "learned"))
        adapter.to(next(query_encoder.parameters()).device)
        if adapter.control == "native":
            adapter.requires_grad_(False)
        self.profiler = StageProfiler(config.get("profile_stages", False))
        core.visual = ConditionedVision(core.visual, adapter, self.profiler)
        self.tokenizer.padding_side = "left"
        self.last_batch_stats = {}

    @property
    def core(self):
        return self.base.get_base_model().model if hasattr(self.base, "peft_config") else self.base.model

    @property
    def adapter(self):
        return self.core.visual.adapter

    def train(self, mode=True):
        super().train(mode)
        self.query_encoder.eval()
        self.core.visual.base.eval()
        return self

    def prepare_batch(self, images, questions, answers=None, device=None):
        device = device or next(self.adapter.parameters()).device
        texts = [q["prompt"] if isinstance(q, dict) else str(q) for q in questions]
        queries = [q["query"] if isinstance(q, dict) else str(q) for q in questions]
        conversations = [[{"role": "user", "content": [{"type": "image"},
                           {"type": "text", "text": text + "\nGive only the concise answer."}]}] for text in texts]
        prompts = [self.processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                   for c in conversations]
        encoded = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        encoded.pop("token_type_ids", None)
        q = self.query_tokenizer(queries, padding=True, truncation=False, return_tensors="pt",
                                 return_special_tokens_mask=True)
        if q.input_ids.shape[1] > self.max_question_tokens:
            raise ValueError("Query+choices exceeds query token cap")
        content_mask = q.attention_mask.bool() & ~q.pop("special_tokens_mask").bool()
        valid_ids = [ids[mask.bool()].tolist() for ids, mask in zip(encoded.input_ids, encoded.attention_mask)]
        labels = None
        if answers is not None:
            tokenized = self.tokenizer(list(answers), add_special_tokens=False).input_ids
            rows = [answer_labels(p, a, self.eos_token_id, self.max_answer_tokens,
                                  self.model_config.get("total_sequence_tokens", 2048))
                    for p, a in zip(valid_ids, tokenized)]
            length = max(len(x[0]) for x in rows)
            encoded["input_ids"] = torch.tensor([[self.pad_token_id]*(length-len(x)) + x for x, _ in rows])
            encoded["attention_mask"] = torch.tensor([[0]*(length-len(x)) + [1]*len(x) for x, _ in rows])
            labels = torch.tensor([[-100]*(length-len(y)) + y for _, y in rows])
        elif encoded.input_ids.shape[1] + self.max_answer_tokens > self.model_config.get("total_sequence_tokens", 2048):
            raise ValueError("Generation prefix+answer budget exceeds total sequence cap")
        # Transformers 5.5 requires these for M-RoPE. Rebuild after appending
        # answers so their shape/left-padding agree with the full input IDs.
        encoded["mm_token_type_ids"] = encoded["input_ids"].eq(self.base.config.image_token_id).long()
        grid = encoded["image_grid_thw"]
        counts = grid.prod(-1) // self.core.visual.spatial_merge_size**2
        if len(counts) != len(images) or (counts > self.model_config.get("visual_tokens", 256)).any():
            raise ValueError("Native processor exceeded visual token budget")
        self.last_batch_stats = {"sequence_tokens": encoded.input_ids.shape[1],
                                 "visual_tokens": counts.tolist()}
        return {"inputs": {k: v.to(device) for k, v in encoded.items()},
                "query_inputs": {k: v.to(device) for k, v in q.items()},
                "query_mask": content_mask.to(device),
                "labels": None if labels is None else labels.to(device)}

    @contextmanager
    def request(self, prepared):
        with self.profiler.measure("query", prepared["query_mask"].device), torch.no_grad():
            query = self.query_encoder(**prepared["query_inputs"]).last_hidden_state
        self.core.rope_deltas = None
        try:
            with self.core.visual.conditioning(query, prepared["query_mask"]):
                yield
        finally:
            self.core.rope_deltas = None
            if hasattr(self.adapter, "ot_loss"):
                self.adapter.ot_loss = None

    def forward(self, images=None, questions=None, answers=None, anno_ids=None,
                return_ot_loss=False, prepared=None):
        prepared = prepared or self.prepare_batch(images, questions, answers)
        labels = prepared["labels"]
        if labels is None:
            raise ValueError("Training requires answer labels")
        # Project only positions predicting answer tokens. Native causal shift
        # is represented exactly once by selecting t-1 for label at t.
        positions = labels[:, 1:].ne(-100).any(0).nonzero().flatten()
        with self.request(prepared):
            with self.profiler.measure("native_forward_including_vision_ot", labels.device):
                outputs = self.base(**prepared["inputs"], use_cache=False,
                                    logits_to_keep=positions, return_dict=True)
            logits = outputs.logits
            targets = labels[:, 1:][:, positions]
            aux = getattr(self.adapter, "ot_loss", None)
        if return_ot_loss:
            return logits, targets, (logits.new_zeros(()) if aux is None else aux)
        return logits, targets

    @torch.no_grad()
    def generate(self, images, questions, anno_ids=None, max_len=None, **kwargs):
        prepared = self.prepare_batch(images, questions)
        was_training = self.training
        self.eval()
        try:
            with self.request(prepared):
                ids = self.base.generate(**prepared["inputs"], do_sample=False, num_beams=1,
                                         max_new_tokens=max_len or self.max_answer_tokens,
                                         use_cache=True, pad_token_id=self.pad_token_id,
                                         eos_token_id=self.eos_token_id)
            return ids[:, prepared["inputs"]["input_ids"].size(1):]
        finally:
            self.train(was_training)

    @torch.no_grad()
    def evaluate_batch(self, images, questions, answers, anno_ids=None):
        logits, targets = self(images, questions, answers, anno_ids)
        ids = self.generate(images, questions, anno_ids)
        return logits, targets, ids

    def answers_from_ids(self, ids):
        return [postprocess_answer(s) for s in self.tokenizer.batch_decode(ids, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)]


def build_pretrained_vlm(config, device):
    import transformers
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Native OT integration requires transformers=={TRANSFORMERS_VERSION}")
    config = dict(config)
    for name in ("vlm_revision", "query_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", config.get(name) or ""):
            raise ValueError(f"{name} must be an immutable 40-character HF commit SHA")
    for name in ("visual_tokens", "total_sequence_tokens", "query_tokens", "cost_dim", "max_answer_tokens"):
        if config.get(name, 1) < 1:
            raise ValueError(f"{name} must be positive")
    if config.get("lora_rank", 0) < 0:
        raise ValueError("lora_rank cannot be negative")
    quantized = config.get("quantize_4bit", False)
    stored_quantization = config.get("quantization_config")
    if quantized and stored_quantization and set(stored_quantization.get("llm_int8_skip_modules") or []) != set(QUANTIZATION_SKIP_MODULES):
        raise ValueError("Checkpoint has a different NF4 visual scope; start a fresh run with corrected exclusions")
    if quantized and device.type != "cuda":
        raise ValueError("NF4 training is supported only on CUDA")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = dict(revision=config["vlm_revision"], dtype=dtype, attn_implementation="sdpa")
    if quantized:
        kwargs.update(device_map={"": device.index or 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=list(QUANTIZATION_SKIP_MODULES)))
    base = Qwen3VLForConditionalGeneration.from_pretrained(config["vlm_model"], **kwargs)
    if not quantized:
        base.to(device)
    if quantized and any(type(parameter).__name__ == "Params4bit" for parameter in base.model.visual.parameters()):
        raise RuntimeError("Vision encoder was unexpectedly quantized; check full module exclusion paths")
    base.requires_grad_(False)
    if quantized:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        base.model.visual.to(dtype=dtype)
    rank = config.get("lora_rank", 0)
    if rank:
        from peft import LoraConfig, get_peft_model
        base = get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank*2, lora_dropout=.05,
            target_modules=r"model.language_model.layers.\d+.self_attn.(q_proj|k_proj|v_proj|o_proj)",
            bias="none", task_type="CAUSAL_LM"))
    if config.get("gradient_checkpointing", True):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    processor = AutoProcessor.from_pretrained(config["vlm_model"], revision=config["vlm_revision"])
    # Native processor uses patch size * spatial merge size as its resize factor.
    factor = base.config.vision_config.patch_size * base.config.vision_config.spatial_merge_size
    processor.image_processor.size = {"shortest_edge": factor**2 * 4,
                                      "longest_edge": factor**2 * config.get("visual_tokens", 256)}
    processor.image_processor.min_pixels = factor**2 * 4
    processor.image_processor.max_pixels = factor**2 * config.get("visual_tokens", 256)
    query_id = config.get("text_model", "sentence-transformers/all-MiniLM-L12-v2")
    query_encoder = AutoModel.from_pretrained(query_id, revision=config["query_revision"]).to(device)
    query_tokenizer = AutoTokenizer.from_pretrained(query_id, revision=config["query_revision"])
    config["transformers_version"] = TRANSFORMERS_VERSION
    import hashlib
    import json
    quant_config = getattr(base.config, "quantization_config", None)
    config["quantization_config"] = quant_config.to_dict() if hasattr(quant_config, "to_dict") else quant_config
    config["lora_config"] = ({name: json.loads(json.dumps(value.to_dict(), default=str))
                              for name, value in base.peft_config.items()} if hasattr(base, "peft_config") else None)
    config["processor_config"] = processor.to_dict()
    config["tokenizer_special_tokens"] = {k: str(v) for k, v in processor.tokenizer.special_tokens_map.items()}
    config["tokenizer_vocab_sha256"] = hashlib.sha256(json.dumps(
        processor.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
    config["query_tokenizer_vocab_sha256"] = hashlib.sha256(json.dumps(
        query_tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
    return PretrainedVLM(base, processor, query_encoder, query_tokenizer, config)
