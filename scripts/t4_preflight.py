"""Probe two T4 GPUs and real FP16/NF4 kernels before loading a large model."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import torch


def probe_gpu(index):
    device = torch.device('cuda', index)
    report = {'name': torch.cuda.get_device_name(index),
              'bytes': torch.cuda.get_device_properties(index).total_memory,
              'capability': torch.cuda.get_device_capability(index)}
    try:
        import bitsandbytes as bnb
        with torch.cuda.device(index):
            layer = bnb.nn.Linear4bit(64, 64, bias=False, quant_type='nf4',
                                     compress_statistics=True, compute_dtype=torch.float16).to(device)
            layer.requires_grad_(False)
            values = torch.randn(2, 64, device=device, dtype=torch.float16, requires_grad=True)
            with torch.autocast('cuda', dtype=torch.float16):
                output = layer(values)
                output.float().square().mean().backward()
            torch.cuda.synchronize(device)
            if not torch.isfinite(output).all() or not torch.isfinite(values.grad).all() or not values.grad.abs().any():
                raise FloatingPointError('NF4 forward/input gradients are not finite and nonzero')
            report['fp16_nf4_forward_backward'] = True
            report['free_bytes_after_probe'], report['total_bytes'] = torch.cuda.mem_get_info(device)
    except Exception as exc:
        report['fp16_nf4_forward_backward'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    count = torch.cuda.device_count()
    try:
        command = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                                  '--format=csv,noheader'], text=True, capture_output=True) if count else None
    except FileNotFoundError:
        command = None
    gpus = [probe_gpu(i) for i in range(count)]
    report = {'cuda_device_count': count, 'torch_version': str(torch.__version__),
              'cuda_version': torch.version.cuda, 'disk_free_bytes': shutil.disk_usage('.').free,
              'gpus': gpus, 'nvml_snapshot': command.stdout.strip() if command else None,
              'two_t4_ready': count == 2 and all('T4' in gpu['name'] and gpu['fp16_nf4_forward_backward'] for gpu in gpus)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['two_t4_ready']:
        raise SystemExit('Two T4 GPUs with working FP16/NF4 were not detected; inspect the preflight report.')


if __name__ == '__main__':
    main()
