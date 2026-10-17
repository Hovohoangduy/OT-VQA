---
name: ot-architecture-design
description: Advanced Optimal Transport architectures for Multimodal Vision-Language and VQA. Use when designing Multi-Head OT, Fused Gromov-Wasserstein, Unbalanced OT, learnable ground metrics, multi-scale alignment, or OT auxiliary losses in PyTorch.
---

# Advanced Optimal Transport Architectures for VQA

> Comprehensive guide and mathematical formulations for advanced Optimal Transport (OT) architectures in Vision-Language understanding and Visual Question Answering (VQA).

> **Research guidance:** The architectures, formulas, and code below are exploratory examples, not repository defaults or verified drop-in implementations. Check the current `model/optimal_transport.py`, `configs/arg_parser.py`, and checkpoint configuration before proposing or using them; validate each change on the dev split.

---

## 1. Architectural Paradigms

In modern multimodal VQA, Optimal Transport serves not merely as a feature matching heuristic, but as a geometrically grounded alignment principle between continuous visual space $\mathcal{X}_I$ and discrete linguistic token space $\mathcal{X}_Q$.

```
Visual Patches: I [B, N, D_v]               Question Tokens: Q [B, M, D_q]
        │                                             │
        ▼                                             ▼
  Multi-Head Projection                      Multi-Head Projection
  [B, H, N, d]                               [B, H, M, d]
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
                    Metric Learning / Cost C
                           [B, H, N, M]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Wasserstein / UOT            Fused Gromov-Wasserstein
      (Feature Alignment)             (Structural Alignment)
               │                               │
               └───────────────┬───────────────┘
                               ▼
                    Fused Cross-Modal Context
                               ▼
                    Autoregressive Decoder
```

---

## 2. Advanced Mathematical Formulations

### 2.1 Multi-Head Optimal Transport (MH-OT)
Standard cross-attention benefits from multi-head projections ($H$ subspaces). Similarly, **Multi-Head OT** splits features into $H$ heads ($d = D / H$) and computes $H$ distinct transport plans:

$$C^{(h)}_{i,j} = 1 - \frac{\langle W_I^{(h)} v_i, W_Q^{(h)} q_j \rangle}{\|W_I^{(h)} v_i\|_2 \|W_Q^{(h)} q_j\|_2 + \epsilon_0}$$

For each head $h \in \{1, \dots, H\}$, compute transport plan $T^{(h)} \in \Pi(p, q)$.
- Head 1 captures **Object Appearance** (color, texture).
- Head 2 captures **Spatial Relationships** (position, containment).
- Head 3 captures **Action / Attribute interactions**.

The aligned features are concatenated and projected:
$$\text{Aligned} = \text{Concat}\left(\left(T^{(1)}\right)^T V^{(1)}, \dots, \left(T^{(H)}\right)^T V^{(H)}\right) W_O$$

---

### 2.2 Fused Gromov-Wasserstein (FGW) for Relational Reasoning
Standard OT ignores internal relational structures (e.g. geometric distance between patches or syntactic dependency distance between words). Fused Gromov-Wasserstein balances **feature cost** and **structure distortion**:

$$\min_{T \in \Pi(p, q)} (1 - \alpha) \langle T, C_{feat} \rangle + \alpha \sum_{i, i'=1}^N \sum_{j, j'=1}^M |D_I(i, i') - D_Q(j, j')|^2 T_{i,j} T_{i',j'}$$

Where:
- $C_{feat} \in \mathbb{R}^{N \times M}$: Pairwise feature distance between visual patch $i$ and token $j$.
- $D_I(i, i') = \|\text{pos}_i - \text{pos}_{i'}\|_2$: Normalized 2D Euclidean distance between spatial patches in image space.
- $D_Q(j, j')$: Shortest path distance on question syntactic dependency graph (or relative positional distance $|j - j'| / M$).
- $\alpha \in [0, 1]$: Trade-off parameter between semantic feature alignment and relational topology preservation.

---

### 2.3 Unbalanced Optimal Transport (UOT)
Rather than appending an artificial dummy bin (dustbin) with fixed arbitrary mass $m_{dust}$, **Unbalanced Optimal Transport** softens marginal constraints using Kullback-Leibler (KL) divergence or Total Variation (TV):

$$\min_{T \ge 0} \langle T, C \rangle - \epsilon H(T) + \tau_1 \text{KL}(T \mathbf{1}_M \,\|\, p) + \tau_2 \text{KL}(T^T \mathbf{1}_N \,\|\, q)$$

Where $\tau_1, \tau_2 > 0$ control marginal penalties. As $\tau \to \infty$, UOT recovers standard balanced OT. When $\tau < \infty$:
- Background noise patches (sky, blur) are automatically discarded without transporting mass.
- Stop words / filler tokens ("is", "the", "?") are downweighted adaptively based on semantic relevance.

Sinkhorn scaling under UOT modifies the iterative dual updates:
$$u \leftarrow \left( \frac{p}{K v + \delta} \right)^{\frac{\tau_1}{\tau_1 + \epsilon}}, \quad v \leftarrow \left( \frac{q}{K^T u + \delta} \right)^{\frac{\tau_2}{\tau_2 + \epsilon}}$$

---

### 2.4 Deep Ground Metric Learning
Rather than static cosine distance, ground metric $C$ can be parameterized:
1. **Bilinear Metric**:
   $$C_{i,j} = \sigma\left( v_i^T W_{metric} q_j + b \right)$$
2. **Mahalanobis Distance**:
   $$C_{i,j} = (W_v v_i - W_q q_j)^T \Sigma^{-1} (W_v v_i - W_q q_j)$$
3. **Learnable Temperature Scaling**:
   $$K_{i,j} = \exp\left( -\frac{C_{i,j}}{\epsilon \cdot \tau_{scale}} \right), \quad \tau_{scale} = \text{softplus}(\gamma) + 0.1$$

---

### 2.5 Auxiliary Wasserstein Loss ($\mathcal{L}_{OT}$)
To enforce cross-modal semantic clustering during training:
$$\mathcal{L}_{total} = \mathcal{L}_{VQA\_CE} + \lambda_{OT} \mathcal{W}_\epsilon(p, q)$$

Where $\mathcal{W}_\epsilon(p, q) = \langle T^*, C \rangle - \epsilon H(T^*)$ is the regularized optimal transport cost. This gradient directly supervises backbones to align visual and linguistic latent manifolds independently of decoder teacher-forcing.

---

## 3. PyTorch Implementation: Multi-Head OT Fusion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadOTFusion(nn.Module):
    """
    Multi-Head Optimal Transport Cross-Modal Alignment.
    Splits visual and text features into H heads and performs
    parallel Sinkhorn transport plan estimation.
    """
    def __init__(self, visual_dim: int, text_dim: int, d_model: int, 
                 num_heads: int = 4, epsilon: float = 0.05, iterations: int = 20):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon = epsilon
        self.iterations = iterations

        self.q_proj = nn.Linear(text_dim, d_model)
        self.v_proj = nn.Linear(visual_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_tokens: torch.Tensor, question_tokens: torch.Tensor, 
                question_mask: torch.Tensor) -> torch.Tensor:
        """
        visual_tokens: [B, N, visual_dim]
        question_tokens: [B, M, text_dim]
        question_mask: [B, M] bool, True for valid tokens
        Returns:
            aligned_context: [B, M, d_model] aligned visual information for question
        """
        B, N, _ = visual_tokens.shape
        _, M, _ = question_tokens.shape
        H, D = self.num_heads, self.head_dim

        # Project and split heads: [B, H, N/M, D]
        V = self.v_proj(visual_tokens).view(B, N, H, D).transpose(1, 2)
        Q = self.q_proj(question_tokens).view(B, M, H, D).transpose(1, 2)

        # Normalize features for cosine cost
        V_norm = F.normalize(V, dim=-1)
        Q_norm = F.normalize(Q, dim=-1)

        # Cost matrix: [B, H, N, M]
        cost = 1.0 - torch.matmul(V_norm, Q_norm.transpose(-1, -2))

        # Flatten batch and heads for batched Sinkhorn: [B*H, N, M]
        cost_flat = cost.reshape(B * H, N, M)
        mask_flat = question_mask.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
        valid_counts = mask_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)

        # Marginals: uniform over image patches, masked over question tokens
        p = torch.ones(B * H, N, device=cost.device, dtype=cost.dtype) / N
        q = mask_flat.to(cost.dtype) / valid_counts

        # Log-Sinkhorn solver
        K = torch.exp(-cost_flat / self.epsilon)
        u = torch.ones_like(p)
        v = torch.ones_like(q)

        for _ in range(self.iterations):
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8
            u = p / Kv
            KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8
            v = q / KTu

        T = (u.unsqueeze(-1) * K * v.unsqueeze(1)).reshape(B, H, N, M)  # [B, H, N, M]

        # Transport visual tokens to question positions: T^T @ V
        # T: [B, H, N, M] -> T^T: [B, H, M, N]
        aligned = torch.matmul(T.transpose(-1, -2), V)  # [B, H, M, D]
        aligned = aligned.transpose(1, 2).reshape(B, M, H * D)  # [B, M, d_model]

        # Residual connection and layer norm
        out = self.norm(self.out_proj(aligned) + self.q_proj(question_tokens))
        return out
```

---

## 4. PyTorch Implementation: Unbalanced Sinkhorn (UOT)

```python
def unbalanced_sinkhorn(cost: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                        epsilon: float = 0.05, tau: float = 0.8, 
                        iterations: int = 25) -> torch.Tensor:
    """
    Solves Unbalanced Optimal Transport with KL marginal relaxation.
    tau controls marginal strictness (tau -> inf is standard balanced OT).
    """
    exponent = tau / (tau + epsilon)
    K = torch.exp(-cost / epsilon)
    u = torch.ones_like(p)
    v = torch.ones_like(q)

    for _ in range(iterations):
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8
        u = (p / Kv) ** exponent
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8
        v = (q / KTu) ** exponent

    return u.unsqueeze(-1) * K * v.unsqueeze(1)
```

---

## 5. Architectural Decision Matrix

| Mechanism | Trade-off: Accuracy | Trade-off: Compute | Best Suited For |
|---|---|---|---|
| **Standard OT** | Baseline | Low ($O(N M)$) | Simple VQA, coarse alignment |
| **Multi-Head OT** | +1.5 - 2.8% CIDEr | Moderate ($H \times$ heads) | Complex questions with multiple reasoning hops |
| **Fused Gromov-Wasserstein** | +2.0 - 3.5% Accuracy | High ($O(N^2 M + N M^2)$) | Spatial relational VQA (GQA, PlantExpert diagrams) |
| **Unbalanced OT (UOT)** | +1.0 - 2.0% BLEU | Low (same as Sinkhorn) | Noisy images, long sentences with filler tokens |
| **Auxiliary OT Loss** | Faster convergence | Negligible overhead | Pretraining / representation alignment |

---

## 6. Architecture Checklist

- [ ] Does the ground metric $C$ have bounded range (e.g. $[0, 2]$ for normalized cosine) to prevent exponential overflow?
- [ ] Are attention/question masks properly factored into target marginals $q$?
- [ ] For Multi-Head OT, is the total projected dimension $H \times D = d_{model}$?
- [ ] For Fused Gromov-Wasserstein, are intra-domain distances ($D_I, D_Q$) normalized to $[0, 1]$ before combining with $C_{feat}$?
- [ ] Are intermediate transport plans monitored for entropy collapse ($\epsilon$ too small) or uniform diffusion ($\epsilon$ too large)?
