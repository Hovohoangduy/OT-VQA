# Implementation validation — 15/10/2026

**Trạng thái cập nhật: DONE.** Hoàn tất báo cáo kiểm chứng local; kết quả lượt rà soát cuối là 43 tests pass.

Code cho correctness, training infrastructure và native OT prototype đã được
triển khai. **Chưa nghiệm thu CUDA/2 × T4 hoặc mục tiêu ROUGE-L >0.80.**
Hướng dẫn chạy tiếp: [runbook](scivqa-implementation-runbook.md).

## Kiểm tra thực sự đã chạy

Môi trường tách biệt `/tmp/ot-vqa-implementation-env`, PyTorch 2.10.0,
Transformers 5.5.0, PEFT 0.18.1, Accelerate 1.13.0; máy CPU/MPS, CUDA count 0.
Không sửa môi trường Python global để chạy native tests.

| Lệnh / phép kiểm tra | Kết quả |
|---|---|
| `python -m pytest tests -q` | **36 passed**, không skip; CPU/Gloo 2 process và MPS có trong lượt local |
| `python -m compileall -q model utils configs train.py test.py predict.py scripts tests` | Pass |
| `git diff --check` | Pass |
| `python train.py --help`, `python test.py --help`, `python predict.py --help` | Pass |
| `HF_HUB_OFFLINE=1 python scripts/smoke_native_processor.py` | Actual pinned processor + hai aspect ratios + tiny native Qwen: CE/backward/generation pass |
| `HF_HUB_OFFLINE=1 python scripts/audit_native_tokens.py --output .../native_token_audit_reproduced.json` | Full train/dev token audit pass, không đọc test |
| `scripts/validate_scivqa_correctness.py --checkpoint data/weights/scivqa.pt --device mps --output .../correctness` | Đủ 1.680 dev QA × 3 conditions; 5.040 predictions |
| `test.py --dataset scivqa --checkpoint data/weights/scivqa.pt --screen_manifest .../screen-one-smoke.json --batch_size 7 --device mps --no-compute_bertscore --predictions_csv ... --report_json ...` | Entry point thật, 7 QA của 1 complete figure: pass; không coi sample score là quality estimate |
| `scripts/scivqa_evidence.py --output .../screen-80.json` | Fixed manifest 80 figures / 560 QA |
| `scripts/scivqa_evidence.py --baseline .../baseline.csv --candidate .../decode_fix.csv --output .../decode-paired-comparison.json` | Official ROUGE + paired figure/paper intervals và guardrails pass |
| `scripts/t4_preflight.py --output .../local_preflight.json` | Exit 1 đúng dự kiến: không có 2 × T4; ghi rõ chưa đạt hardware gate |

Các lệnh trên dùng interpreter `/tmp/ot-vqa-implementation-env/bin/python`.
`...` trong bảng là `results/scivqa/implementation_20261015` hoặc file tương ứng
trong thư mục này; scripts có tên trong bảng được giữ nguyên để tái lập.

Tests bao phủ numeric/binary/multilabel, bool/count/choices prompt,
causal/PAD/BOS/EOS semantics, PAD=EOS native labels, OT padding/gradients/FP32,
FGW padding invariance, feasible independent plan, gate mở rồi cost gradients,
main+DeepStack native no-op logits/generation parity, sparse-vs-full native loss
và gradients, frozen LLM + checkpointing + LoRA, request/cache reset, exact
accumulation/short groups/token-vs-example/auxiliary weighting, overflow skip,
CPU DDP equivalence, RNG, OT+LoRA/legacy checkpoint round-trip và next-update,
full training-entrypoint epoch resume và fresh-output safety.

## Kết quả correctness trên dev

| Cùng checkpoint, không retrain | ROUGE-L F1 |
|---|---:|
| Prompt v1 / decode v1 | 0.4489067 |
| Prompt v1 / decode v2 | 0.4516977 |
| Prompt v2 / decode v2 | 0.4379918 |

Decode delta +0.2791 điểm phần trăm, CI95 theo 240 figure
[-0.0936; +0.6735] điểm phần trăm, chưa đủ bằng chứng tăng quality.
Prompt mới trên weights cũ làm giảm 1.0915 điểm phần trăm; checkpoint cũ vì vậy
inherit prompt cũ. Prompt v2 sửa metadata được dùng cho fresh training và explicit
inference intervention, không silently đổi đầu vào checkpoint cũ.

Baseline 44.89% khớp review trước dù environment nay đúng Transformers 5.5.0;
chưa phục hồi stored 47.05%. Historical training snapshot không có manifest vẫn
được ghi là chưa tái lập, không dùng stored metric làm matched baseline.

Token audit: question+choices max 210 train / 140 dev; native answer+EOS max
47 / 30. Không có mẫu vượt cap 256/64 trên revision train/dev local. Parsed choices
không đổi prompt nào hiện có (choices đã nằm trong question). Inference source
hashes cuối khớp report full-dev; source snapshot full-dev được thu ở cuối lượt.

## Cổng còn lại

1. Full pretrained 2B native/NF4 parity và gradients trên CUDA thật.
2. CUDA/NCCL accumulation, overflow agreement, per-rank resume; 100–200 updates
   và validation với worst allowed shapes, ≥10–15% headroom mỗi T4.
3. Matched native LoRA / OT+LoRA / pooled control, same evidence/updates; full-dev
   và 7 nhóm, figure/paper CI; repeat winner seeds 1105/42/2026.
4. Chỉ chọn OCR/tiles/external/distillation/retrieval sau error evidence; không
   tự xem các nhánh có điều kiện này là đã được thực nghiệm.
5. Freeze recipe rồi final held-out test. **Chỉ measured held-out >0.80 mới đạt
   mục tiêu test.** Không dùng 36 tests hay tiny Qwen smoke để suy ra accuracy.

[Artifacts local](../../results/scivqa/implementation_20261015/implementation_summary.json)
và [full-dev report](../../results/scivqa/implementation_20261015/correctness/report.json).
Results bị Git ignore theo cấu hình repo; source tests/scripts được giữ để tái lập.
