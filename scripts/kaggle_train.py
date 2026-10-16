"""Validated Kaggle two-T4 commands; delegates optimization to train.py."""
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_t4_pilot import check_pilot
from utils.training_preflight import validate_training_data
from configs.arg_parser import get_args

BASE_REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
QUERY_REVISION = 'a50ef00143b4d5391434df20ae11632588ac25be'


def training_arguments(args):
    if args.stage == 'resume':
        argv = ['--resume', str(Path(args.resume).resolve()), '--device', 'cuda',
                '--log_every_updates', str(args.log_every_updates)]
        if args.epochs is not None:
            argv += ['--epochs', str(args.epochs)]
        return argv
    data = Path(args.data_root).resolve()
    run_name = f'scivqa_qwen2b_ot_lora16_v{args.visual_tokens}_s{args.seed}'
    if args.stage == 'pilot':
        run_name += '_pilot'
    output = Path(args.output_root).resolve() / run_name
    argv = ['--dataset', 'scivqa', '--model_family', 'qwen3_vl',
            '--vlm_model', 'Qwen/Qwen3-VL-2B-Instruct', '--vlm_revision', BASE_REVISION,
            '--query_revision', QUERY_REVISION, '--quantize_4bit', '--amp', 'fp16', '--amp_init_scale', '1024', '--device', 'cuda',
            '--batch_size', '1', '--gradient_accumulation_steps', '8',
            '--visual_tokens', str(args.visual_tokens), '--total_sequence_tokens', '2048',
            '--query_tokens', '256', '--max_answer_tokens', '64', '--cost_dim', '128',
            '--figure_context', '--prompt_version', 'scivqa-v2',
            '--adapter_control', 'ot', '--plan_control', 'learned', '--ot_num_heads', '1',
            '--ot_epsilon', '0.05', '--ot_iterations', '20', '--ot_dustbin_mass', '0.2', '--ot_dustbin_cost', '1.0',
            '--loss_normalization', 'example', '--ot_loss_weight', '0', '--label_smoothing', '0',
            '--lora_rank', '16', '--lora_lr', '2e-5', '--lr', '1e-4', '--weight_decay', '0.05',
            '--warmup_ratio', '0.05', '--gradient_clip', '1', '--gradient_checkpointing',
            '--epochs', str(args.epochs if args.epochs is not None else 3),
            '--early_stopping_patience', '2', '--seed', str(args.seed),
            '--no-compute_bertscore', '--selection_metric', 'rouge_l_f1', '--train_eval_samples', '0',
            '--num_workers', str(args.num_workers), '--log_every_updates', str(args.log_every_updates),
            '--train_csv_path', str(data / 'train.csv'), '--dev_csv_path', str(data / 'val.csv'),
            '--img_path', str(data / 'images'), '--model_path', str(output)]
    if args.stage == 'pilot':
        argv += ['--max_updates', str(args.updates), '--profile_stages']
    return argv


def require_matching_pilot(path, resolved):
    directory = Path(path)
    manifest = json.loads((directory / 'run_config.json').read_text())
    rows = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines() if s.strip()]
    report = check_pilot(manifest, rows, lora_rank=16)
    if not report['measured_resource_checks_passed']:
        raise ValueError('Pilot failed: ' + json.dumps(report['checks']))
    critical = [ROOT/'train.py', ROOT/'test.py', *sorted((ROOT/'model').glob('*.py')),
                *sorted((ROOT/'utils').glob('*.py')), *sorted((ROOT/'configs').glob('*.py'))]
    saved_hashes = manifest.get('source_sha256', {})
    changed = [str(p.relative_to(ROOT)) for p in critical
               if saved_hashes.get(str(p.relative_to(ROOT))) != hashlib.sha256(p.read_bytes()).hexdigest()]
    if changed:
        raise ValueError('Training source differs from measured pilot: ' + ', '.join(changed))
    saved = manifest['arguments']
    differences = [name for name in ('vlm_model', 'vlm_revision', 'query_revision', 'visual_tokens',
                   'query_tokens', 'max_answer_tokens', 'total_sequence_tokens', 'cost_dim', 'lora_rank',
                   'batch_size', 'gradient_accumulation_steps', 'amp', 'amp_init_scale', 'quantize_4bit', 'epochs',
                   'lr', 'lora_lr', 'warmup_ratio', 'gradient_checkpointing', 'figure_context',
                   'prompt_version', 'ot_epsilon', 'ot_iterations', 'ot_dustbin_mass', 'ot_dustbin_cost',
                   'label_smoothing', 'loss_normalization', 'ot_loss_weight', 'adapter_control', 'plan_control',
                   'text_model', 'weight_decay', 'gradient_clip', 'seed', 'selection_metric',
                   'early_stopping_patience', 'num_workers', 'compute_bertscore', 'train_eval_samples')
                   if saved.get(name) != getattr(resolved, name)]
    if differences:
        raise ValueError('Full run differs from measured pilot: ' + ', '.join(differences))
    from train import _csv_digest
    for split in ('train', 'dev'):
        if manifest.get(f'{split}_csv_sha256') != _csv_digest(getattr(resolved, f'{split}_csv_path')):
            raise ValueError(f'{split} CSV differs from pilot')
    return report


def validate_environment():
    expected = {'torch': '2.10.0', 'torchvision': '0.25.0', 'transformers': '5.5.0', 'peft': '0.18.1', 'accelerate': '1.13.0', 'bitsandbytes': '0.49.2'}
    versions = {name: importlib.metadata.version(name).split('+')[0] for name in expected}
    if versions != expected:
        raise RuntimeError(f'Install requirements.txt with this interpreter first: {versions}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'train', 'resume'), required=True)
    parser.add_argument('--data_root', help='Folder containing train.csv, val.csv and images/')
    parser.add_argument('--output_root', default='/kaggle/working')
    parser.add_argument('--pilot_run')
    parser.add_argument('--resume')
    parser.add_argument('--visual_tokens', type=int, choices=(256, 512), default=256)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total epochs: fresh runs default to 3; resume inherits unless explicitly increased')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1105)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--log_every_updates', type=int, default=25)
    parser.add_argument('--dry_run', action='store_true', help='Resolve command/data without loading a model or requiring CUDA')
    args = parser.parse_args()
    if args.stage == 'resume' and not args.resume:
        parser.error('--stage resume requires --resume')
    if args.stage != 'resume' and not args.data_root:
        parser.error('--data_root is required for a fresh run')
    if args.stage == 'train' and not args.pilot_run:
        parser.error('--stage train requires --pilot_run from the same configuration')
    if (args.epochs is not None and args.epochs < 1) or args.num_workers < 0 or args.log_every_updates < 1:
        parser.error('epochs/log_every_updates must be positive; num_workers must be nonnegative')
    if args.stage == 'pilot' and not 100 <= args.updates <= 200:
        parser.error('Resource pilot requires 100–200 updates')
    argv = training_arguments(args)
    if args.stage != 'resume':
        resolved = get_args(argv)
        print(json.dumps(validate_training_data(resolved, 2, figure_context=True, verify_images=True), indent=2), flush=True)
        destination = Path(resolved.model_path)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'Fresh run output must be empty: {destination}')
        if args.stage == 'train':
            print(json.dumps(require_matching_pilot(args.pilot_run, resolved), indent=2), flush=True)
    else:
        if not Path(args.resume).is_file():
            raise FileNotFoundError(args.resume)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=2', str(ROOT / 'train.py'), *argv]
    print(json.dumps({'command': command, 'dry_run': args.dry_run}, indent=2), flush=True)
    if args.dry_run:
        return
    validate_environment()
    # Run GPU probing in a separate process so this launcher leaves no CUDA
    # context on GPU 0 while the two training workers are active.
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    subprocess.run([sys.executable, str(ROOT / 'scripts/t4_preflight.py'), '--output',
                    str(output / f't4-preflight-{stamp}.json')], check=True, cwd=ROOT)
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    subprocess.run(command, check=True, cwd=ROOT, env=env)


if __name__ == '__main__':
    main()
