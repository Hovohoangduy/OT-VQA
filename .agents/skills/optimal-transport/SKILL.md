---
name: optimal-transport
description: Optimal Transport and Sinkhorn algorithms for multimodal feature fusion. Use when designing, debugging, or analyzing transport plans, cost matrices, entropic regularization, or partial OT in PyTorch.
---

# Optimal Transport for Multimodal Fusion (OT-VQA)

> Core knowledge for Optimal Transport theory, Sinkhorn-Knopp algorithm, partial OT, and multimodal vision-language fusion.

> **Repository note:** The codebase uses a masked, question-conditioned partial OT fusion path. The production implementation is `model/optimal_transport.py`; generic balanced/uniform examples below explain the math and are not drop-in replacements. Preserve valid-token masks, dustbin semantics, optional unbalanced marginals, and checkpointed settings when changing it.

---

## 1. Mathematical Foundations

### 1.1 Kantorovich Formulation
Given source distribution $p \in \Delta^n$ (visual patch features) and target distribution $q \in \Delta^m$ (question token features), and a ground cost matrix $C \in \mathbb{R}_{+}^{n \times m}$, the optimal transport problem seeks a transport plan $T \in \mathbb{R}_{+}^{n \times m}$:

$$\min_{T \in \Pi(p, q)} \langle T, C \rangle = \sum_{i=1}^n \sum_{j=1}^m T_{i,j} C_{i,j}$$

Subject to marginal constraints:
$$T \mathbf{1}_m = p, \quad T^T \mathbf{1}_n = q$$

### 1.2 Entropic Regularization (Cuturi, 2013)
To enable efficient computation on GPUs via matrix scaling:

$$\min_{T \in \Pi(p, q)} \langle T, C \rangle - \epsilon H(T)$$

Where $H(T) = -\sum_{i,j} T_{i,j} (\log T_{i,j} - 1)$ is the entropy, and $\epsilon > 0$ is the regularization parameter.

The unique optimal solution has the form:
$$T^* = \text{diag}(u) K \text{diag}(v), \quad \text{where } K = \exp(-C / \epsilon)$$

---

## 2. Sinkhorn-Knopp Iterations

### 2.1 Standard Kernel Scaling (Linear Domain)
Given uniform or conditioned marginals $p$ and $q$:
- Initialize $v^{(0)} = \mathbf{1}_m$
- Iteratively update:
  $$u^{(k+1)} = \frac{p}{K v^{(k)} + \delta}$$
  $$v^{(k+1)} = \frac{q}{K^T u^{(k+1)} + \delta}$$
Where $\delta = 10^{-8}$ prevents division by zero.

### 2.2 Log-Domain Stabilization (For Small $\epsilon$)
When $\epsilon < 0.05$, direct exponentiation $\exp(-C/\epsilon)$ can underflow to 0. Log-domain Sinkhorn uses `logsumexp`:
$$\log u \leftarrow \log p - \text{logsumexp}(-C/\epsilon + \log v, \text{dim}=-1)$$
$$\log v \leftarrow \log q - \text{logsumexp}(-C^T/\epsilon + \log u, \text{dim}=-1)$$

---

## 3. Partial Optimal Transport & Dummy Bins

In VQA, not all visual patches correspond to question words (and vice-versa). Partial OT relaxes strict conservation of mass:
1. **Dummy Bin / Margin Extension**:
   - Append an extra row/column with fixed cost $\alpha$ or learned slack penalty.
   - Allows uninformative background patches or filler tokens to be "discarded" with minimal penalty.
2. **Question-Conditioned Marginals**:
   - Weight tokens by attention/importance rather than strictly uniform $\frac{1}{m}$.
   - Focus transport mass onto key semantic question tokens (nouns, verbs, attributes).

---

## 4. PyTorch Implementation Guidelines

```python
import torch
import torch.nn.functional as F

def sinkhorn_knopp(cost_matrix: torch.Tensor, 
                   reg: float = 0.1, 
                   max_iter: int = 50, 
                   thresh: float = 1e-3) -> torch.Tensor:
    """
    Computes optimal transport plan T given cost matrix C [B, N, M].
    Args:
        cost_matrix: [B, N, M] pairwise costs
        reg: entropic regularization epsilon
        max_iter: maximum scaling iterations
        thresh: early stopping threshold on marginal error
    Returns:
        T: [B, N, M] optimal transport plan
    """
    B, N, M = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    # Uniform marginal distributions
    p = torch.ones(B, N, device=device, dtype=dtype) / N
    q = torch.ones(B, M, device=device, dtype=dtype) / M

    # Gibbs kernel
    K = torch.exp(-cost_matrix / reg)  # [B, N, M]
    
    u = torch.ones(B, N, device=device, dtype=dtype)
    v = torch.ones(B, M, device=device, dtype=dtype)

    for _ in range(max_iter):
        u_prev = u
        # u = p / (K @ v)
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8
        u = p / Kv
        
        # v = q / (K^T @ u)
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8
        v = q / KTu

        # Convergence check
        if torch.max(torch.abs(u - u_prev)) < thresh:
            break

    # Transport plan: T = diag(u) @ K @ diag(v)
    T = u.unsqueeze(-1) * K * v.unsqueeze(1)
    return T
```

---

## 5. Diagnostics & Debugging OT

- **Symptom: Uniform Transport Plan (Mode Collapse)**
  - Cause: Regularization $\epsilon$ is too large relative to the scale of $C$.
  - Solution: Reduce $\epsilon$ or normalize $C$ (e.g. cosine distance normalized to $[0, 1]$ or divide by median cost).
- **Symptom: NaN / Inf in Transport Plan**
  - Cause: Zero marginals, division by zero, or extreme costs causing exponent underflow.
  - Solution: Use $\delta = 10^{-8}$ clamp, check input feature normalization (L2 norm), or switch to log-domain.
- **Symptom: Gradient vanishing through OT**
  - Cause: Too many unrolled iterations or extreme $\epsilon$.
  - Solution: Use 20-50 iterations; verify that backpropagation flows through $C$ via $T$.
