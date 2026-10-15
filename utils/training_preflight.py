"""Cheap data and trainable-parameter checks before expensive optimization."""
from pathlib import Path
import math

import torch

from utils.data_processing import load_dataframe
from utils.vqa_dataset import resolve_image_root


def validate_training_data(args, world_size=1, figure_context=False, verify_images=False):
    if Path(args.train_csv_path).resolve() == Path(args.dev_csv_path).resolve():
        raise ValueError("Train and dev CSV must be different files")
    report = {}
    for split in ('train', 'dev'):
        frame = load_dataframe(getattr(args, f'{split}_csv_path'))
        if frame.empty or frame.anno_id.astype(str).duplicated().any():
            raise ValueError(f'{split} must be nonempty with unique annotation IDs')
        if figure_context and not {'caption', 'figure_type'}.issubset(frame.columns):
            raise ValueError(f'{split}: figure context requires caption and figure_type')
        if split == 'train' and len(frame) % world_size:
            raise ValueError('Train sample count must divide world_size without sampler duplication')
        root = resolve_image_root(frame, args.img_path, split, getattr(args, f'{split}_img_path'))
        paths = [root / name for name in frame.image.astype(str).unique()]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: {len(missing)} missing images, first: {missing[0]}')
        if verify_images:
            from PIL import Image
            for path in paths:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    raise ValueError(f'Invalid image: {path}') from exc
        report[split] = {'examples': len(frame), 'figures': len(paths), 'image_root': str(root)}
    train_ids = set(load_dataframe(args.train_csv_path).anno_id.astype(str))
    dev_ids = set(load_dataframe(args.dev_csv_path).anno_id.astype(str))
    # IDs can be split-local on other datasets; image overlap is more useful to
    # report than rejecting independently numbered official splits.
    report['annotation_id_overlap'] = len(train_ids & dev_ids)
    report['optimizer_updates_per_epoch'] = math.ceil(
        math.ceil(report['train']['examples'] / world_size / args.batch_size) / args.gradient_accumulation_steps)
    report['effective_batch'] = args.batch_size * world_size * args.gradient_accumulation_steps
    return report


def validate_trainable_parameters(model, device, amp='off'):
    summary = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        expected_index = torch.cuda.current_device() if device.type == 'cuda' and device.index is None else device.index
        if parameter.device.type != device.type or (expected_index is not None and parameter.device.index != expected_index):
            raise ValueError(f'Trainable parameter on wrong device: {name}: {parameter.device}, expected {device}')
        if amp == 'fp16' and parameter.dtype != torch.float32:
            raise ValueError(f'FP16 GradScaler requires FP32 trainable parameters: {name} is {parameter.dtype}')
        group = 'lora' if 'lora_' in name else 'ot' if 'adapter.' in name else 'legacy'
        summary[group] = summary.get(group, 0) + parameter.numel()
    if not summary:
        raise ValueError('No trainable parameters; native training control requires LoRA')
    return summary
