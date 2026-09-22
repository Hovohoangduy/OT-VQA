# OT-VQA Architecture, Multimodal Fusion & Benchmark Pipeline Documentation

This document provides a comprehensive technical breakdown of the **OT-VQA** system codebase based on the current repository state (`compare-results` branch).

---

## 1. System Overview

**OT-VQA** unifies vision-language representations for Visual Question Answering (VQA) using **Optimal Transport (Balanced & Unbalanced)** paired with five diverse multimodal fusion architectures:
1. **SAN** (Stacked Attention Network) & **OT-SAN** (Gated Global Summary over OT Tokens)
2. **BAN** (Bilinear Attention Network with Bilinear Glimpses + Additive OT Log-Prior)
3. **MUTAN** (Multimodal Tucker Fusion with Low-Rank Tensor Product + Normalized OT Column Pooling)
4. **Cross-Attention Transformer** (Multi-Head Cross-Attention + Additive OT Logit Bias)
5. **Q-Former** (Learnable Query Tokens Attending Grounded Multimodal Memory)
6. **Barycentric Reference Baseline** (Direct 4-way Interaction MLP on Aligned Patches)

The source also provides an experimental **Gated OT-Aligned Cross-Attention** family.
It interpolates raw and OT-grounded question embeddings with a learned token-wise gate
before native Cross-Attention. It is intentionally outside the original 15-configuration
result matrix until its paired multi-seed benchmark is completed.

### Controlled 15-Configuration Matrix
Each fusion family is evaluated under three transport modes:
- **`none`**: Native multimodal fusion without Optimal Transport.
- **`balanced`**: Balanced Optimal Transport (strict marginal matching via Sinkhorn).
- **`uot`**: Question-conditioned Unbalanced Optimal Transport (relaxed marginal matching via KL divergence).

Together with the 3 random seeds (`1105`, `1106`, `1107`), this forms an automated **45-run controlled benchmark** measuring whether explicit OT alignment improves accuracy or visual grounding.

---

## 2. Core Model Architecture

The complete model logic is centralized in `model/vqa_model.py` and its modular subsystems in `model/`:

```
               [ Input Image: 3×224×224 ]          [ Question Text ]
                           │                               │
            ┌──────────────▼──────────────┐ ┌──────────────▼──────────────┐
            │  Frozen ViT                 │ │  Frozen BERT                │
            │  (Removes CLS token)        │ │  (Masks PAD & boundary)     │
            │  V ∈ ℝ^(B × 196 × 768)      │ │  Q ∈ ℝ^(B × M × 768)        │
            └──────────────┬──────────────┘ └──────────────┬──────────────┘
                           │                               │
                           ├───────────────────────────────┤
                           │                               │
            ┌──────────────▼───────────────────────────────▼──────────────┐
            │  Optimal Transport Alignment Subsystem (model/optimal_transport.py) │
            │  1. Question-Conditioned Marginals: a ∈ ℝ^(B×196), b ∈ ℝ^(B×M)│
            │  2. Hybrid Ground Cost: C = λ·C_cos + (1-λ)·C_learned       │
            │  3. Log-Domain Sinkhorn Solver (Balanced or UOT relaxation) │
            │  ==> Transport Plan: P ∈ ℝ^(B × 196 × M)                    │
            └──────────────────────────────┬──────────────────────────────┘
                                           │
            ┌──────────────────────────────▼──────────────────────────────┐
            │  Multimodal Fusion Module (model/fusion_methods.py & ot_san.py)     │
            │  • SAN: Legacy 1-token summary                              │
            │  • OT-SAN: Gated global summary S + OT tokens [g·S, H]      │
            │  • BAN: Glimpse scores + λ_ot·log(P + ε)                    │
            │  • MUTAN: Rank-5 Tucker tensor with OT visual pooling Ṽ_j   │
            │  • Cross-Attention: Attention logits + λ_ot·log(P^T + ε)    │
            │  • Q-Former: 8 query tokens cross-attending [V, H_ot]       │
            │  ==> Fused Context Memory: H ∈ ℝ^(B × L × d_model)          │
            └──────────────────────────────┬──────────────────────────────┘
                                           │
            ┌──────────────────────────────▼──────────────────────────────┐
            │  Autoregressive Transformer Decoder (model/decoder_model.py)│
            │  • Causal Masking on Shifted Answer Prefix                  │
            │  • Cross-Attention to Memory H with Boolean Padding Mask    │
            │  • Output Linear Projection ==> Logits ∈ ℝ^(B × L_ans × |V|) │
            └─────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Technical Details

### 3.1 Feature Extraction (`model/features_extraction.py`)
- **Visual Encoder**: `google/vit-base-patch16-224-in21k` (`ImageEmbedding`). It discards the first `[CLS]` token to yield exactly $N = 196$ spatial patch vectors ($14 \times 14$ grid).
- **Text Encoder**: `bert-base-uncased` (`QuestionEmbedding`). Questions are whitespace-normalized and tokenized. Padding and special tokens are retained for positional integrity but masked out during marginal estimation and transport.
- **Answer Embedding**: `AnswerEmbedding` wraps BERT's embedding table. Can be frozen via `--freeze_answer_embeddings` to prevent overfitting on small datasets.

### 3.2 Optimal Transport Subsystem (`model/optimal_transport.py`)
- **Marginals ($a, b$)**:
  - $a = \text{masked\_softmax}(\text{MLP}([V_i, g_Q, V_i \odot g_Q, |V_i - g_Q|]))$ where $g_Q = \text{masked\_mean}(Q)$.
  - $b = \text{masked\_softmax}(\text{MLP}([Q_j, g_V, Q_j \odot g_V, |Q_j - g_V|]))$ where $g_V = \text{masked\_mean}(V)$.
- **Ground Cost Matrix ($C$)**:
  - $C_{\text{cos}} = 1 - \cos(\bar{V}, \bar{Q})$.
  - $C_{\text{learned}} = \text{softplus}(-\text{MLP}([\bar{V}, \bar{Q}, \bar{V} \odot \bar{Q}, |\bar{V} - \bar{Q}|]))$.
  - $C = \lambda C_{\text{cos}} + (1 - \lambda) C_{\text{learned}}$.
- **Log-Domain Sinkhorn**:
  - Operates strictly in float32 log-space to guarantee numerical stability:
    $$\log K = -C / \varepsilon$$
    $$\log u \leftarrow \rho_v [\log a - \text{logsumexp}(\log K + \log v)]$$
    $$\log v \leftarrow \rho_q [\log b - \text{logsumexp}(\log K^T + \log u)]$$
    $$P = \exp(\log u + \log K + \log v)$$
  - Balanced OT: $\rho_v = \rho_q = 1.0$.
  - Unbalanced OT (UOT): $\rho = \tau / (\tau + \varepsilon)$, with KL relaxation.

### 3.3 Multimodal Fusion Families (`model/fusion_methods.py` & `model/ot_san.py`)
All fusions adhere to `FusionInput` and `FusionOutput` contracts:
1. **SAN / OT-SAN**:
   - Without OT: Legacy SAN compresses question and image into a single global summary vector.
   - With OT (`uot_san`): Masked stacked attention computes global context $S$; gated memory is $[g \cdot S, H]$ where $g = \sigma(\text{gate\_logit})$ with `gate_init = -2.0`.
2. **BAN**:
   - $G=2$ bilinear attention glimpses: $\text{score}_g(i, j) = w_g^T(\tanh(W_v V_i) \odot \tanh(W_q Q_j))$.
   - In `uot_ban`, the transport plan is an additive log-prior: $\text{score}_g + \lambda_{\text{ot}} \log(P + \epsilon)$ with learned scalar $\lambda_{\text{ot}}$.
3. **MUTAN**:
   - Tucker bilinear tensor decomposition with rank $R=5$: $v_r = W_v(v), q_r = W_q(q), z = W_o(\sum_{r=1}^5 v_r \odot q_r)$.
   - In `uot_mutan`, native pooling is substituted directly with normalized OT columns $\tilde{V}_j = \sum_i P_{ij} V_i / \max(\sum P_{ij}, \epsilon)$.
4. **Cross-Attention Transformer**:
   - Multi-head attention with questions querying visual keys/values.
   - In `uot_cross_attention`, normalized log transport is injected directly into attention logits: $\frac{QK^T}{\sqrt{d_k}} + \lambda_{\text{ot}} \log(P^T + \epsilon)$.
5. **Gated OT-Aligned Cross-Attention**:
   - Available as `aligned_cross_attention`, `balanced_ot_aligned_cross_attention`, and `uot_aligned_cross_attention`.
   - Projects raw question tokens $Q_r$ and OT-grounded tokens $H_{ot}$, predicts a scalar gate per valid token, and computes $Q_{aligned}=Q_r+g(H_{ot}-Q_r)$.
   - Gate weights initialize to zero and bias to `-2.0`, so OT initially contributes about 11.9%; native attention logits remain unbiased by the transport plan.
   - Reports gate mean/standard deviation, alignment distance, and attention entropy.
6. **Q-Former**:
   - 8 learned query tokens. In `uot_qformer`, queries cross-attend multimodal memory $[V, H_{\text{ot}}]$ where $H_{\text{ot}}$ is the grounded barycentric tokens.

---

## 4. Pipeline Execution & Flows

### 4.1 Training Flow (`train.py`)
- **Teacher Forcing**: Answer IDs $[y_0, y_1, \dots, y_L]$ are split:
  - Decoder inputs: $\text{ids}[:, :-1]$ (starting with `[BOS]`).
  - Target outputs: $\text{ids}[:, 1:]$ (ending with `[EOS]`).
- **Loss**: `nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)`.
- **Checkpoint Selection**: Evaluated on **generated validation F1** (reference-free), breaking ties with validation loss. Never uses teacher-forced metrics for model selection.
- **Early Stopping**: Default patience of 8 epochs.

### 4.2 Inference Flow (`predict.py`)
- Takes single image and question string.
- Computes contextual memory $H$.
- Initializes answer sequence with `[BOS]`.
- Autoregressively generates next token via greedy argmax until `[EOS]` or `max_len` (12) is reached.
- Decodes token IDs back to human-readable string.

### 4.3 Feature Caching (`precompute_features.py` & `utils/feature_cache.py`)
- Precomputes ViT patch tokens and BERT question tokens as float16 tensors.
- Stores cryptographic manifest (CSV SHA-256, encoder revisions, tokenization policies).
- Eliminates 75%+ of training epoch compute time.

---

## 5. Automated Benchmark Suite

### 5.1 Runner Script (`scripts/run_fusion_benchmark.sh`)
Supports preflight validation and multi-seed grid execution:
```bash
# Smoke test across all 15 configurations
DEVICE=mps EPOCHS=2 SEEDS="1105" \
METHODS="san ban mutan cross_attention qformer" \
TRANSPORTS="none balanced uot" \
RUN_ROOT=results/smoke_test \
scripts/run_fusion_benchmark.sh

# Complete 3-seed benchmark (45 runs)
DEVICE=mps SEEDS="1105 1106 1107" \
RUN_ROOT=results/full_benchmark \
scripts/run_fusion_benchmark.sh
```

### 5.2 Aggregator & Paired Deltas (`scripts/summarize_fusion_benchmark.py`)
Computes seed-matched differences:
- $\Delta_{\text{balanced}} = \text{Metric}_{\text{balanced}} - \text{Metric}_{\text{none}}$
- $\Delta_{\text{uot}} = \text{Metric}_{\text{uot}} - \text{Metric}_{\text{none}}$
- $\Delta_{\text{uot\_vs\_balanced}} = \text{Metric}_{\text{uot}} - \text{Metric}_{\text{balanced}}$
Generates `runs.csv`, `aggregate.csv`, `paired_deltas.csv`, and `paired_aggregate.csv`.

---

## 6. Source File Map

| Path | Purpose |
| --- | --- |
| `model/vqa_model.py` | Unified `VQAModel` integrating encoders, OT subsystem, 5 fusion modules, and decoder |
| `model/fusion_methods.py` | Implementation of BAN, MUTAN, Cross-Attention, and Q-Former with OT augmentation |
| `model/ot_san.py` | OT-SAN gated stacked attention module over barycentric tokens |
| `model/optimal_transport.py` | `OTConfig`, marginal MLPs, hybrid cost, log-domain Sinkhorn solver |
| `model/features_extraction.py` | Pretrained ViT and English BERT token extraction |
| `model/decoder_model.py` | Autoregressive Transformer decoder with causal self-attention |
| `scripts/run_fusion_benchmark.sh` | Automated execution script for the 15-configuration grid |
| `scripts/summarize_fusion_benchmark.py` | Aggregator computing paired differences and summary tables |
| `train.py` | Training loop with teacher forcing, label smoothing, and validation generation |
| `test.py` | Generated EM/F1 evaluation and teacher-forced validation loss scoring |
| `predict.py` | Single-image, single-question autoregressive inference CLI with optional visual diagnostics |
| `diagnose_training.py` | Counterfactual modality shuffle test, bottleneck diagnostic, and output diversity metrics |
| `docs/optimal_transport_vqa_architecture.html` | Interactive, styled visual system guide with live Sinkhorn convergence charts |
