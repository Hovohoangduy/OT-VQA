"""Reject resource/gradient mismatches before a costly two-T4 full run."""
import argparse
import json
import math
from pathlib import Path


def check_pilot(manifest, rows, min_updates=100, headroom=.10, lora_rank=16):
    last = rows[-1] if rows else {}
    ranks = last.get('runtime_by_rank', [])
    args = manifest.get('arguments', {})
    config = manifest.get('model_config', {})
    memory_ok = lambda memory: memory.get('usable_headroom_fraction', -1) >= headroom
    gradient_categories = {'gates', 'visual_cost', 'query_cost', 'query_value', 'outputs'}
    if lora_rank:
        gradient_categories.add('lora_')
    def gradients_ok(rank):
        samples = rank.get('gradient_samples', [])
        # Zero cost gradients at the initial zero gate are expected; every
        # category must receive finite nonzero gradients later in the pilot.
        return all(any(math.isfinite(s.get('norms', {}).get(name, float('nan'))) and
                       s['norms'][name] > 0 for s in samples) for name in gradient_categories)
    checks = {
        'two_ranks': len(ranks) == 2 and manifest.get('world_size') == 2,
        'effective_batch_16': manifest.get('global_batch_size') == 16,
        'enough_updates': last.get('global_step', 0) >= min_updates,
        'pilot_has_same_lora_rank': config.get('lora_rank') == lora_rank,
        'cuda_fp16_nf4': args.get('amp') == 'fp16' and config.get('quantize_4bit') is True,
        'correct_visual_quantization_scope': set((config.get('quantization_config') or {}).get('llm_int8_skip_modules') or []) == {'model.visual', 'lm_head'},
        'ot_control': config.get('adapter_control') == 'ot' and config.get('plan_control') == 'learned',
        'no_overflow': bool(rows) and all(r.get('overflow_skips', -1) == 0 for row in rows for r in row.get('runtime_by_rank', [])),
        'all_t4': len(ranks) == 2 and all('T4' in r.get('memory', {}).get('name', '') for r in ranks),
        'train_memory_headroom': bool(ranks) and all(memory_ok(r.get('memory', {})) for row in rows for r in row.get('runtime_by_rank', [])),
        'gradient_flow_each_rank': len(ranks) == 2 and all(gradients_ok(r) for r in ranks),
        'validation_completed': bool(manifest.get('dev_examples')) and last.get('validation_performance', {}).get('examples') == manifest.get('dev_examples'),
        'finite_validation': math.isfinite(last.get('val_loss', float('nan'))) and math.isfinite(last.get('val_rouge_l_f1', float('nan'))),
        'validation_headroom': memory_ok(last.get('validation_memory', {})),
    }
    residuals = [r.get('ot_diagnostics', {}).get('row_residual', {}).get('max', 0.) for r in ranks]
    notices = ([f"Maximum OT row residual {max(residuals):.3g} exceeds 1e-4; finite transport is not proof of convergence at this iteration cap."]
               if residuals and max(residuals) > 1e-4 else [])
    return {'solver_notices': notices, 'checks': checks, 'measured_resource_checks_passed': all(checks.values()),
            'manual_gate_required': ['Exercise worst allowed image/query/answer shapes; observed maxima alone do not prove coverage.',
                                     'Check native/NF4 parity and CUDA/NCCL equivalence/resume fixtures on target hardware.'],
            'quality_gate': 'Requires matched full-dev comparisons and repeat seeds; a resource pilot does not establish accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--min_updates', type=int, default=100)
    parser.add_argument('--headroom', type=float, default=.10)
    parser.add_argument('--lora_rank', type=int, default=16, help='Must match the intended full run')
    args = parser.parse_args()
    directory = Path(args.run)
    manifest = json.loads((directory / 'run_config.json').read_text())
    rows = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines() if s.strip()]
    report = check_pilot(manifest, rows, args.min_updates, args.headroom, args.lora_rank)
    print(json.dumps(report, indent=2))
    if not report['measured_resource_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
