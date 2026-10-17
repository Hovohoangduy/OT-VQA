# Triển khai SciVQA: code, bằng chứng và lệnh chạy 2 × T4

**Trạng thái cập nhật: DONE.** Hoàn tất cập nhật runbook/lệnh trực tiếp cho code hiện tại; chưa phải nghiệm thu training trên T4.

Ngày 15/10/2026. Đây là runbook implementation cho hai plan đã review, không phải
báo cáo đạt ROUGE-L >0.80. Giữ test split đến khi recipe được chốt. OCR/tiles,
external data, retrieval và distillation vẫn là các nhánh **có điều kiện theo lỗi
đo được** trong plan, không bật đồng thời ở prototype đầu.

## Hợp đồng đã triển khai

- Legacy là mặc định. Fresh: `scivqa-v2` + `task-v2`; checkpoint thiếu version:
  `scivqa-v1` + `task-v1`. V2 bỏ numeric hard mask/heuristic numeric prompt;
  binary và multi-answer choice vẫn có format guidance. Unanswerability gold
  không thành oracle. Count panel không được dùng làm target panel.
- `VQADataset` và predict dùng chung `build_question`, parse choices và CSV bool.
  Native dataset giữ raw RGB, query = question+choices, caption chỉ ở language
  input. Không có answer/ID/paper trong query OT.
- Native Qwen3-VL: frozen vision/connector và MiniLM; cost dimension 128, 1 plan
  cho main và toàn bộ native DeepStack levels. Các native mergers giữ cùng
  grid/order; wrapper kiểm shapes/split counts. Không monkeypatch thư viện global.
- `g=0`, projection random: no-op native; mở gate thì gradient tới cost/value
  projections. FP32 island cho projection/cost/Sinkhorn/division OT.
- Native causal supervision tính tại đúng vị trí t−1 dự đoán answer ở t, chỉ
  project union các answer positions. Không shift labels hai lần; EOS vẫn được
  học khi PAD=EOS. Test so loss/gradient với native full-logits causal loss.
- Greedy generation dùng native cache, vision/OT một lần ở prefill; query/RoPE
  request state reset trong `finally`. Không tính lại OT từng generated token.
- Train có accumulation theo global example/token counts, auxiliary theo mean
  example riêng, `no_sync`, AMP, unscale/clip, all-rank overflow agreement,
  scheduler chỉ theo successful updates. Batch cuối thiếu được chuẩn hóa thật.
  Train distributed từ chối số sample không chia hết world size để không silently
  duplicate padding của DistributedSampler. Full SciVQA 15.120 mẫu chia hết 2.
- Checkpoint schema 5: adapter+LoRA trainable state cho native, full state legacy;
  base/query immutable revisions, processor/tokenizer metadata, optimizer,
  scheduler, scaler, best loss/ROUGE độc lập, RNG từng rank và resolved recipe.
  Save ở epoch/update boundary. Pilot bị cắt giữa epoch không được exact-resume.
  Full-run resume yêu cầu cùng schedule/world size/data/model/objective.
  Historical v3/4 continuation cần chỉ rõ auxiliary weight, không được gọi exact.
- Fresh `model_path` phải rỗng. Không ghi đè run khác. Source ZIP/hash và package
  versions, CSV hashes được lưu trong `run_config.json`.
- `--no-compute_bertscore` giữ official stemmed ROUGE; BERTScore vắng mặt được ghi
  là vắng mặt. Numeric exact có giữ dấu/unit và false abstention là guardrails.

## Environment

Native integration được kiểm với `transformers==5.5.0`, PyTorch 2.10.0,
PEFT 0.18.1, Accelerate 1.13.0. Local là macOS CPU/MPS, **không có CUDA**.
`requirements.txt` pin core candidate cho T4; chưa phải CUDA lock đã
được thực nghiệm. Sau cài trên Kaggle, lưu `pip freeze` cùng preflight; không dùng
local Mac package freeze để tuyên bố môi trường CUDA đã chạy.

```bash
python -m pip install -r requirements.txt
python -m pip freeze > /kaggle/working/ot-vqa-environment.txt
python -m pytest tests -q
python scripts/t4_preflight.py --output results/t4-preflight.json
```

Preflight yêu cầu thực sự thấy 2 × T4; local exit 1 là bằng chứng không đạt phần
cứng, không phải lỗi pipeline. DDP tests tự chạy thêm CUDA/NCCL khi thấy hai GPU.
CPU/Gloo equivalence không thay thế CUDA/NCCL equivalence.

Base 2B và query revisions đã resolve từ HF:

- Qwen/Qwen3-VL-2B-Instruct: `89644892e4d85e24eaac8bacfd4f463576704203`
- sentence-transformers/all-MiniLM-L12-v2: `a50ef00143b4d5391434df20ae11632588ac25be`

Không dùng branch `main` làm revision huấn luyện. 4B phải resolve revision riêng,
qua pilot trước khi chạy dài; không áp revision 2B sang model khác.

## Correctness và baseline

```bash
python scripts/validate_scivqa_correctness.py \
  --checkpoint data/weights/scivqa.pt --device auto \
  --output results/scivqa/correctness-final-snapshot
python scripts/scivqa_evidence.py \
  --output results/scivqa/screen-80.json
```

Lệnh đầu cache frozen vision trong process và so baseline / decode fix / prompt+
decode fix trên đủ dev. Timing có cache **không là benchmark throughput**.
Stored 47.05% thiếu historical manifest vẫn được ghi là chưa tái lập; baseline
đối chứng là lượt same-snapshot mới, không dùng stored metric làm đối chứng.
Lệnh thứ hai chốt 80 complete figures, phân tầng theo figure type trước predictions.
Dùng `test.py --screen_manifest ...` để screening trên đúng subset; full dev để confirm.

## Native screening và pilot

Zero-shot native control, cùng evidence budget với OT:

```bash
python test.py --dataset scivqa --model_family qwen3_vl \
  --pretrained_eval --adapter_control native \
  --vlm_revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --query_revision a50ef00143b4d5391434df20ae11632588ac25be \
  --quantize_4bit --device cuda --batch_size 1 --no-compute_bertscore \
  --screen_manifest results/scivqa/screen-80.json \
  --predictions_csv results/scivqa/native-screen.csv \
  --report_json results/scivqa/native-screen.json
```

**Cập nhật sau final review:** pilot resource phải dùng cùng LoRA rank với full
run. Pilot adapter-only không chứng minh OT+LoRA vừa VRAM. Dùng
[lệnh torchrun trực tiếp](scivqa-kaggle-final-review.md) hoặc launcher đã kiểm tra:

```bash
python scripts/kaggle_train.py --stage pilot \
  --data_root /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset \
  --output_root /kaggle/working --visual_tokens 256 --epochs 3 --updates 100
python scripts/check_t4_pilot.py \
  --run /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105_pilot --lora_rank 16
```

Giữ lịch epochs 3 và mọi budget cho full run. Xem
[final review/runbook](scivqa-kaggle-final-review.md) cho lệnh full/resume.
Adapter-only vẫn là một thí nghiệm phát triển riêng, không được dùng qua cổng
resource của full LoRA run.

Lặp pilot mới với `--visual_tokens 512`, output dir khác. Kiểm train + validation,
≥10–15% usable VRAM headroom từng GPU, không overflow/OOM, gradient gate/OT đúng
stage. `metrics.jsonl` có memory từng rank, actual sequence/visual maxima,
gradient samples và opt-in synchronized stage timings. `native_forward` gồm cả
vision/OT; không cộng các timings này lần nữa. Stage profiling làm chậm pilot;
tắt khi benchmark throughput thật. Dùng NVML snapshot/preflight và theo dõi
`nvidia-smi` khi chạy để xem cả memory ngoài allocator.

Cổng worst-shape vẫn cần chạy examples ở biên cap (ảnh rectangular/compound,
question/choices/answer dài), không suy từ một pilot ngẫu nhiên. Processor hoặc
answer vượt cap sẽ báo lỗi, **không silent truncate**. Nếu dữ liệu thực vượt 64
answer tokens native thì audit, tăng cap có kiểm chứng và chạy lại memory pilot;
không cắt gold hoặc tự đổi objective cho vừa.

Pilot checkpoint cắt giữa epoch dùng để debug/reload, không resume thành final
comparison. Full training cần thư mục mới, bỏ `--max_updates`, dùng lịch chung.
Với native control để **train**, cần `--lora_rank 16`; native + rank 0 là inference.

## Final architecture comparisons sau khi pilot đạt

Reset cùng pinned pretrained base cho ba run: `--adapter_control native`, `ot`,
`pooled`. Cùng seed, data order, effective batch, evidence, epochs, LR LoRA,
warmup và checkpoint selection; `--lora_rank 16 --lora_lr 2e-5`, OT LR 1e-4.
Không khởi tạo OT từ pilot rồi so với native chưa warmup mà quy gain cho solver.
Parameter-matched pooled control giữ các trainable projections/gates, không giải
transport. Native ít parameters hơn được báo riêng.

```bash
python test.py --dataset scivqa --checkpoint data/RUN/best.pt \
  --device cuda --batch_size 1 --no-compute_bertscore \
  --predictions_csv results/RUN-dev.csv --report_json results/RUN-dev.json
python test.py --dataset scivqa --checkpoint data/RUN/best.pt \
  --device cuda --batch_size 1 --no-compute_bertscore --plan_control independent \
  --predictions_csv results/RUN-independent.csv --report_json results/RUN-independent.json
python scripts/scivqa_evidence.py --baseline results/NATIVE-dev.csv \
  --candidate results/OT-dev.csv --output results/OT-vs-native.json
```

Independent coupling giữ local row/column sums, matched mass và dustbin blocks;
đây là inference reliance intervention, không thay cho retrained controls.
Lặp winner với seeds 1105/42/2026, báo full-dev overall/7 nhóm/CI và guardrails.
Chỉ sau dev >0.80 ổn định mới freeze recipe và chạy held-out test một lần theo
plan. Chưa có measured quality thì chưa đạt R4/R5, dù mọi regression tests pass.

## Hạn chế cần nghiệm thu tiếp

CUDA FP16/NF4 và NCCL chưa chạy ở máy local. Tiny native no-op parity/gradient
không chứng minh full pretrained 2B/4B quality. Chưa có accuracy claim cho native,
chưa chọn backbone 4B, chưa có OCR/external/ensemble winner. Không ghi “done” cho
hai evidence plans chỉ vì code/protocol đã được thêm.


## Bằng chứng local ngày 15/10/2026

Chi tiết các lệnh đã chạy và giới hạn: [validation report](scivqa-implementation-validation.md).
36 regression tests đã pass trên CPU/MPS; CUDA tests sẽ được thêm vào lượt chạy khi có GPU.

- Full-dev 1.680 QA / 240 figures, checkpoint `scivqa.pt`, Transformers 5.5.0:
  baseline v1/v1 **0.4489067**; chỉ decode v2 **0.4516977**; prompt v2 + decode v2
  **0.4379918**. Đây là inference interventions trên cùng weights, không retrain.
- Decode delta +0.0027910; paired figure-bootstrap CI95
  **[-0.0009357, +0.0067345]**, chưa hỗ trợ claim gain chắc chắn.
  Prompt+decode delta −0.0109149; CI95 **[-0.0177875, −0.0042926]**.
  Vì vậy checkpoint cũ tiếp tục inherit prompt cũ; prompt mới dùng fresh training.
- Query-only MiniLM max 210 train / 140 dev; native answer+EOS max 47 / 30.
  Cap query 256 / answer 64 không truncate revision train/dev hiện có. Không đọc
  test split để chọn cap. Native total sequence/evidence vẫn phải pilot thực tế.
- Actual pinned Qwen processor + heterogeneous images đã qua CE/backward và
  generation với tiny random native Qwen. Chưa phải full pretrained 2B parity.
- [Reports/predictions/pytest](../../results/scivqa/implementation_20261015/)
  lưu bằng chứng local; thư mục results hiện bị repo Git ignore. Các tests và
  scripts tái lập được giữ trong source, không bị xóa sau khi chạy.

Nguồn API đối chiếu: [Transformers Qwen3-VL v5.5.0](https://huggingface.co/docs/transformers/v5.5.0/en/model_doc/qwen3_vl),
[bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/main/en/installation).
Có hỗ trợ API/phần cứng trong tài liệu không thay thế pilot memory/quality thực tế.
