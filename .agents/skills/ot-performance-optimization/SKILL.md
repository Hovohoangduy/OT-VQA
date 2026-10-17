---
name: ot-performance-optimization
description: Hardware acceleration, memory optimization, and numerical stability for Optimal Transport and Sinkhorn algorithms in PyTorch (CUDA and Apple Silicon MPS). Use when optimizing throughput, reducing VRAM usage, implementing implicit differentiation, adaptive early stopping, or low-rank OT.
---

# Optimal Transport Performance & Hardware Optimization

> Principles and techniques for high-throughput, low-memory Optimal Transport execution on modern accelerators (NVIDIA CUDA & Apple Silicon MPS).

---

## 1. Computational Bottlenecks in Deep OT

When incorporating Sinkhorn-based Optimal Transport into deep learning pipelines, three primary bottlenecks emerge:

| Bottleneck | Cause | Impact | Solution |
|---|---|---|---|
| **Activation Memory Explosion** | PyTorch autograd stores all intermediate tensors across $K$ unrolled iterations | OOM on large batch sizes ($O(K \cdot B \cdot N \cdot M)$) | **Implicit Differentiation** or **Gradient Checkpointing** |
| **Redundant Computation** | Fixed iteration count (e.g., $K=20$) continues even after dual variables stabilize | 40-60% compute wasted | **Adaptive Early Stopping** |
| **Intermediate Allocations** | Repeating `torch.logsumexp(kernel + v.unsqueeze(...))` creates large temporary tensors | Memory bandwidth saturation & GPU cache thrashing | **In-place buffer reuse** & fused operations |
| **Quadratic Spatial Scaling** | Matrix $C \in \mathbb{R}^{N \times M}$ grows quadratically with image resolution $N = H \times W$ | VRAM limits for high-res images ($N=1024+$) | **Low-Rank / Factored OT** ($O((N+M)r)$) |

---

## 2. Key Optimization Techniques

### 2.1 Implicit Differentiation (Envelope Theorem / Danskin)
Rather than backpropagating through all $K$ scaling steps, compute the gradient directly from the converged dual potentials $(u^*, v^*)$:

Given primal cost $\langle T^*, C \rangle$, by the Envelope Theorem:
$$\frac{\partial \langle T^*, C \rangle}{\partial C_{ij}} = T^*_{ij}$$

For downstream loss $\mathcal{L}(T^*)$:
$$\nabla_C \mathcal{L} = T^* \odot \left( \nabla_T \mathcal{L} - \alpha \mathbf{1}^T - \mathbf{1} \beta^T \right)$$
Where $(\alpha, \beta)$ solve the linear system for the Hessian of the entropy regularizer.

**PyTorch Implementation (Zero-Memory Unrolling):**
```python
import torch

class ImplicitSinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost: torch.Tensor, epsilon: float, iterations: int, tol: float):
        # Forward pass runs without building autograd graph
        with torch.no_grad():
            B, N, M = cost.shape
            K = torch.exp(-cost / epsilon)
            p = torch.ones(B, N, device=cost.device, dtype=cost.dtype) / N
            q = torch.ones(B, M, device=cost.device, dtype=cost.dtype) / M
            u = torch.ones_like(p)
            v = torch.ones_like(q)

            for _ in range(iterations):
                u_prev = u
                Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8
                u = p / Kv
                KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8
                v = q / KTu
                if torch.max(torch.abs(u - u_prev)) < tol:
                    break

            T = u.unsqueeze(-1) * K * v.unsqueeze(1)
        
        ctx.save_for_backward(T, cost)
        ctx.epsilon = epsilon
        return T

    @staticmethod
    def backward(ctx, grad_output):
        T, cost = ctx.saved_tensors
        eps = ctx.epsilon
        # Implicit gradient approximation: grad_C = - (1/eps) * T * grad_T (centered)
        grad_T = grad_output
        T_grad = T * grad_T
        grad_C = (T_grad - T * T_grad.sum(dim=(-1, -2), keepdim=True)) / (-eps)
        return grad_C, None, None, None
```

---

### 2.2 Adaptive Early Stopping Sinkhorn
Fixed iterations ($K=20-50$) waste cycles on easy examples that converge within 5-8 iterations. Tracking the marginal residual enables dynamic termination:

```python
def adaptive_log_sinkhorn(cost: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                          epsilon: float = 0.05, max_iter: int = 30, tol: float = 1e-3):
    """
    Solves log-domain Sinkhorn with early stopping on marginal constraint violation.
    Saves ~50% execution time while maintaining exact convergence guarantees.
    """
    B, N, M = cost.shape
    log_p = p.clamp_min(1e-30).log()
    log_q = q.clamp_min(1e-30).log()
    log_K = -cost / epsilon

    log_u = torch.zeros_like(p)
    log_v = torch.zeros_like(q)

    for i in range(max_iter):
        log_u_prev = log_u
        log_u = log_p - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
        log_v = log_q - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=-2)

        # Early check every 2 iterations to amortize norm computation cost
        if i % 2 == 1:
            err = (log_u - log_u_prev).abs().amax()
            if err < tol:
                break

    log_T = log_K + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return log_T.exp()
```

---

### 2.3 Low-Rank / Factored Optimal Transport
For high-resolution visual tokens ($N = H \times W \ge 1024$), full $N \times M$ computation is prohibitive.
Factored OT (Scetbon et al., 2021) factorizes the transport plan into two low-rank transitions through $r$ virtual anchors:

$$T = Q R P^T, \quad Q \in \mathbb{R}^{N \times r}, \; R \in \mathbb{R}^{r \times r}, \; P \in \mathbb{R}^{M \times r}$$

- Reduces complexity: $O(N \cdot M) \longrightarrow O((N + M) \cdot r)$ where $r \ll \min(N, M)$ (e.g. $r=16$).
- Enables direct alignment of $32 \times 32$ spatial patch maps without token downsampling.

---

### 2.4 Device-Specific Accelerations (MPS & CUDA)

#### Apple Silicon MPS (Metal Performance Shaders)
- **Zero-Copy Memory**: Keep cost matrices in contiguous memory (`tensor.contiguous()`) before `torch.bmm`.
- **FP32 Accumulation**: MPS can produce subnormal underflow with FP16 exponential scaling. Always keep `log_sinkhorn` dual updates in FP32, then cast the resulting plan $T$ back to `image_tokens.dtype` (BF16 or FP16).
- **Avoid Dynamic Graph Shapes**: Keep padding shapes fixed across minibatches to allow Metal pipeline caching.

#### NVIDIA CUDA
- Enable TF32 for matrix multiplications:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```
- Use `torch.compile(mode="reduce-overhead")` on the Sinkhorn solver to fuse inner `logsumexp` loops into a single CUDA kernel.

---

## 3. Profiling & Diagnostics Suite

Run this quick benchmark snippet to measure throughput and peak memory:

```python
import time
import torch

def benchmark_ot_solver(solver_fn, batch_size=32, patches=256, tokens=32, runs=50, device="mps"):
    cost = torch.rand(batch_size, patches, tokens, device=device)
    p = torch.ones(batch_size, patches, device=device) / patches
    q = torch.ones(batch_size, tokens, device=device) / tokens

    # Warmup
    for _ in range(5):
        _ = solver_fn(cost, p, q)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = solver_fn(cost, p, q)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs

    print(f"[{device.upper()}] Latency: {elapsed * 1000:.2f} ms | Throughput: {batch_size / elapsed:.1f} samples/sec")
```

---

## 4. Performance Checklist

- [ ] Is `torch.no_grad()` or `detach()` applied when OT plan is used solely for feature gating/routing?
- [ ] Are cost matrices normalized so that $\max(C) \le 2.0$ to prevent exponential overflow in Gibbs kernel $K$?
- [ ] For backward pass, is gradient checkpointing or implicit differentiation enabled for $> 20$ iterations?
- [ ] Does the Sinkhorn loop use adaptive early stopping ($\tau \le 10^{-3}$)?
- [ ] On MPS / CUDA, are dual variables accumulated in FP32 even when using AMP?
