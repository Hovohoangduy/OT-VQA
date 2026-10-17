---
name: pytorch-patterns
description: PyTorch best practices, tensor shape management, device placement (MPS/CUDA/CPU), memory optimization, and safe training loops.
---

# PyTorch Best Practices for OT-VQA

> Guidelines for writing robust, efficient, and memory-safe PyTorch code in research and training.

---

## 1. Device Agnostic Execution

Always use device helpers (see `utils/device.py`):
```python
import torch

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```
- **MPS (Apple Silicon Mac)**: Supported for most operations; some complex operations (like custom sparse indexing or specific scatter ops) may fallback to CPU. Ensure tensors are on the same device.
- **CUDA (NVIDIA GPUs)**: Always check `torch.cuda.is_available()` before invoking CUDA-specific calls (`torch.cuda.empty_cache()`, `torch.cuda.amp`).

---

## 2. Explicit Tensor Shapes & Assertions

Multimodal fusion involves changing dimensions:
- Image spatial features: `[B, C, H, W]` -> Flatten/Project -> `[B, N, D]` (e.g. $N = H \times W$)
- Question tokens: `[B, L, D]`
- Cost Matrix: `[B, N, L]`
- Transport Plan: `[B, N, L]`
- Fused Representation: `[B, N, D]` or `[B, L, D]` or pooled `[B, D]`

**Rule**: Add shape comments or lightweight asserts in critical transforms:
```python
# image_feat: [B, C, H, W] -> [B, N, D]
B, C, H, W = image_feat.shape
visual_tokens = image_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
assert visual_tokens.shape == (B, H * W, C), f"Expected [B, {H*W}, {C}], got {visual_tokens.shape}"
```

---

## 3. Training & Evaluation Safety

### 3.1 Inference Mode & Detaching
Never compute metrics or logs on active computation graphs:
```python
# During validation/test
model.eval()
with torch.no_grad():
    outputs = model(images, questions)
    # Always detach and move to cpu for metrics
    loss_val = loss.detach().item()
    preds = outputs.argmax(dim=-1).detach().cpu().numpy()
```

### 3.2 Gradient Clipping
Prevent exploding gradients, especially during early warm-up or when computing through Sinkhorn:
```python
scaler.step(optimizer) # if using AMP
# Or standard:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
```

---

## 4. Efficient Data Loading

In PyTorch, data loading often becomes the GPU bottleneck:
1. `pin_memory=True`: Use when training on CUDA.
2. `num_workers`: Typically 2–4 workers per GPU. Avoid high workers on macOS to prevent memory ballooning.
3. `persistent_workers=True`: Keeps workers alive between epochs to avoid re-spawning overhead.

---

## 5. Checkpointing & Reproducibility

1. **Deterministic Seeds**:
   ```python
   import random, os, numpy as np, torch
   def seed_everything(seed=42):
       random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
           torch.backends.cudnn.deterministic = True
   ```
2. **Safe Saving**:
   Save dictionary containing model state, optimizer state, scheduler state, epoch, and validation metric:
   ```python
   checkpoint = {
       "epoch": epoch,
       "model_state_dict": model.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
       "best_val_loss": best_loss,
       "config": config,
   }
   torch.save(checkpoint, save_path)
   ```
