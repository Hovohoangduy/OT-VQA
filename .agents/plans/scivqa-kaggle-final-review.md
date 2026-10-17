# Rà soát cuối và lệnh Kaggle 2 × T4

**Trạng thái cập nhật: DONE.** Hoàn tất review code và cập nhật lệnh torchrun; chờ người dùng chạy pilot CUDA trên Kaggle.

Ngày 15/10/2026. Phạm vi: code/config của đường Qwen3-VL-2B + OT + LoRA,
training/runtime, loader, checkpoints và lệnh torchrun trực tiếp. Không chạy lại full-dev
chỉ để lặp số cũ; không huấn luyện dài hoặc đọc held-out test trong lượt này.

## Kết luận trước khi mang đi train

**Có thể chuyển sang kiểm chứng pilot trên Kaggle; chưa thể xác nhận full run
an toàn trên T4 từ máy local không CUDA.** Không dùng command legacy SigLIP hoặc
pilot adapter-only để nghiệm thu QLoRA+OT. Notebook file đã được xóa theo yêu cầu;
dùng lệnh torchrun bên dưới trong repo mới nhất. Full run dùng cùng
source/config/data với pilot; lệnh trực tiếp không tự chặn khi pilot chưa pass.

## Phát hiện và sửa trong lượt cuối

| Mức | Phát hiện | Sửa và kiểm chứng |
|---|---|---|
| P1 | NF4 exclude `visual` không khớp `model.visual.*` trong Transformers 5.5.0; vision có thể bị quantize ngoài ý định | Dùng `model.visual`, `lm_head`; assert vision không chứa `Params4bit`; regression test kiểm toàn bộ visual linear subtree bằng hàm upstream thật. Checkpoint dùng scope cũ bị từ chối thay vì đổi ngầm |
| P1 | Linux default fork DataLoader sau init CUDA/NCCL có nguy cơ deadlock | Dùng multiprocessing `spawn` khi workers >0; test thực sự iterate worker spawn |
| P1 | Adapter-only pilot không đại diện activation/grads/optimizer của LoRA16 | Pilot và full đều LoRA16, cùng caps/schedule; full launcher so source hashes, CSV hashes và cấu hình trước khi chạy |
| P2 | Pilot gate có thể pass dù gradient cost/LoRA chưa được kiểm, reserved-only headroom bỏ qua CUDA allocations bên ngoài | Gate yêu cầu gradient finite/nonzero từng rank, đúng quantization scope, FP16/NF4, full validation; memory headroom xét cả free memory và peak reservation |
| P2 | Load backbone trước khi phát hiện sai CSV/ảnh; log tiến độ quá thưa | Kiểm train/dev/path/IDs/figure context trước model; launcher verify ảnh trước torchrun; log mỗi 25 attempts với successful updates, loss/LR/overflow/time |
| P2 | FP16 GradScaler sẽ không unscale được trainable gradients FP16 nếu freeze/PEFT policy sai | Assert trainables đúng local device và FP32 trước optimizer. Launcher dùng initial scale 1024, scaler vẫn dynamic; không cam kết hết overflow |
| P2 | Lệnh ngắn trong README có thể chạy nhầm legacy preset/30 epochs hoặc path local | Thêm lệnh torchrun/launcher với explicit native recipe 3 epochs; hướng dẫn legacy được gắn nhãn riêng; historical tuning dẫn sang runbook mới |

Không đổi dataset splits, gold answers hoặc default legacy architecture. OT vẫn
là fusion mặc định của native research path; không thêm auxiliary loss để che
lỗi integration. Sửa scope quantization làm pilot cũ không còn đại diện memory;
phải pilot lại với code đã sửa.

## Lệnh torchrun trực tiếp trong Kaggle

Bật **GPU T4 ×2**, Internet; chạy từ repo mới nhất. Giữ đúng các đường dẫn dữ liệu
người dùng đã cung cấp. Thư mục output phải mới/rỗng.

```python
%pip install -r requirements.txt
```

```bash
!python -m pytest tests -q
!python scripts/t4_preflight.py --output /kaggle/working/t4-preflight-before-native.json
```

Pilot 100 updates, **cùng lịch 3 epochs** với full run để giữ warmup/scheduler:

```bash
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset scivqa --device cuda --fusion ot --model_family qwen3_vl \
  --vlm_model Qwen/Qwen3-VL-2B-Instruct \
  --vlm_revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --text_model sentence-transformers/all-MiniLM-L12-v2 \
  --query_revision a50ef00143b4d5391434df20ae11632588ac25be \
  --quantize_4bit --amp fp16 --amp_init_scale 1024 --gradient_checkpointing \
  --epochs 3 --batch_size 1 --gradient_accumulation_steps 8 --num_workers 2 \
  --lora_rank 16 --lora_lr 2e-5 --lr 1e-4 --warmup_ratio 0.05 \
  --weight_decay 0.05 --label_smoothing 0 --loss_normalization example \
  --gradient_clip 1.0 --early_stopping_patience 2 \
  --ot_num_heads 1 --cost_dim 128 --ot_loss_weight 0 \
  --ot_epsilon 0.05 --ot_iterations 20 --ot_dustbin_mass 0.2 --ot_dustbin_cost 1.0 \
  --adapter_control ot --plan_control learned \
  --figure_context --prompt_version scivqa-v2 \
  --query_tokens 256 --max_answer_tokens 64 --total_sequence_tokens 2048 \
  --visual_tokens 256 --selection_metric rouge_l_f1 \
  --train_eval_samples 0 --no-compute_bertscore --log_every_updates 25 --seed 1105 \
  --train_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/train.csv \
  --dev_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/val.csv \
  --img_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/images \
  --max_updates 100 --profile_stages \
  --model_path /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105_pilot
!python scripts/check_t4_pilot.py \
  --run /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105_pilot --lora_rank 16
```

Full run fresh từ pinned base sau khi pilot và kiểm tra worst-shape đạt:

```bash
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset scivqa --device cuda --fusion ot --model_family qwen3_vl \
  --vlm_model Qwen/Qwen3-VL-2B-Instruct \
  --vlm_revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --text_model sentence-transformers/all-MiniLM-L12-v2 \
  --query_revision a50ef00143b4d5391434df20ae11632588ac25be \
  --quantize_4bit --amp fp16 --amp_init_scale 1024 --gradient_checkpointing \
  --epochs 3 --batch_size 1 --gradient_accumulation_steps 8 --num_workers 2 \
  --lora_rank 16 --lora_lr 2e-5 --lr 1e-4 --warmup_ratio 0.05 \
  --weight_decay 0.05 --label_smoothing 0 --loss_normalization example \
  --gradient_clip 1.0 --early_stopping_patience 2 \
  --ot_num_heads 1 --cost_dim 128 --ot_loss_weight 0 \
  --ot_epsilon 0.05 --ot_iterations 20 --ot_dustbin_mass 0.2 --ot_dustbin_cost 1.0 \
  --adapter_control ot --plan_control learned \
  --figure_context --prompt_version scivqa-v2 \
  --query_tokens 256 --max_answer_tokens 64 --total_sequence_tokens 2048 \
  --visual_tokens 256 --selection_metric rouge_l_f1 \
  --train_eval_samples 0 --no-compute_bertscore --log_every_updates 25 --seed 1105 \
  --train_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/train.csv \
  --dev_csv_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/val.csv \
  --img_path /kaggle/input/datasets/duyho2907/sci-vqa-dataset-nganga01/scivqa_dataset/images \
  --model_path /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105
```

Không resume từ pilot. Batch 1/GPU × accumulation8 × 2 GPU = **effective batch16**,
945 updates/epoch trên 15.120 QA. Khởi đầu 3 epochs cho nhánh pretrained; không
mang nguyên 25 epochs của decoder cũ sang QLoRA. `--lr` dành cho OT,
`--lora_lr` dành cho LoRA. Query MiniLM và native vision giữ frozen.

Các flags SigLIP/custom decoder `--image_model`, `--image_size`,
`--image_preprocessing`, `--d_model`, `--ffn_hidden`, `--num_layers`,
`--num_heads`, `--drop_prob`, `--ot_visual_memory`, `--tie_weights`,
`--freeze_answer_embeddings`, `--no-freeze_question_encoder`, `--encoder_lr`
không thuộc recipe native này. FGW/UOT chưa được hỗ trợ trong native prototype:
không truyền `--fgw_alpha 0.15` hoặc `--unbalanced_tau 1.0`; dùng single-head OT,
auxiliary0 và smoothing0 theo plan. Native processor kiểm soát độ phân giải qua
`--visual_tokens`; query+choices cap256, answer cap64, tổng context2048.
BERTScore và train-answer scoring được tắt trong run để tiết kiệm compute;
checkpoint vẫn chọn theo full-dev official ROUGE-L F1.

Nếu cần visual512, chạy pilot mới và full cùng `--visual_tokens 512`, output
khác. Native 256 chưa chắc đủ đọc nhãn nhỏ: chọn budget bằng pilot/dev screening.
`kaggle_train.py` vẫn là launcher tùy chọn có gate tự động; lệnh trực tiếp ở đây
để người dùng kiểm soát rõ argv.

## Pilot đo gì, không đo gì

`kaggle_train.py` kiểm ảnh/CSV; process preflight riêng chạy NF4 forward/backward
thật trên từng GPU để phát hiện ngay wheel/driver/kernel lỗi. Process preflight
thoát trước torchrun, không để lại CUDA context cạnh GPU0 worker.

```bash
!python scripts/check_t4_pilot.py \
  --run /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105_pilot \
  --lora_rank 16 --min_updates 100 --headroom 0.10
```

Gate tự động yêu cầu hai T4, batch16, đủ updates, đúng LoRA/quantization scope,
không overflow, gradient gate/cost/value/output/LoRA hữu hạn và khác0 từng rank,
validation đủ dev/hữu hạn, và ít nhất10% usable memory headroom. Nhắm10–15%.
`metrics.jsonl` cũng ghi min/max matched mass và OT row/column residual theo rank;
residual lớn hơn1e-4 có notice, không tự xem solver đã hội tụ chỉ vì loss hữu hạn.
Initial zero gate có cost gradients bằng0 ở bước đầu là đúng; gate phải mở và
cost/LoRA phải nhận gradient ở các samples tiếp theo.

Vẫn cần kiểm actual **worst allowed shapes** và native/NF4 no-op parity trên
GPU đích. CPU tests/tiny native tests không thay cho full pretrained CUDA check.
Không coi pass memory là quality gain. Nếu gate fail, đọc report trước khi rerun;
không tăng gradient clipping, giảm ảnh hoặc giảm cap một cách ngầm định để pass.

Dùng `seconds/update` và validation timing trong pilot để ước lượng train+validation
và đối chiếu quota. Pilot profiling có synchronization nên timing là ước lượng thận trọng,
không là benchmark throughput cuối. Không chạy full nếu một epoch+validation
không vừa thời gian còn lại trong session.

## Lưu và resume giữa các Kaggle sessions

Hiện checkpoint **chỉ lưu sau epoch+validation**; ngắt giữa epoch phải làm lại
phần epoch đó. Chưa có exact mid-epoch resume. Đây là hạn chế có thật, không được
che bằng một flag save nhưng thiếu sampler/accumulation/RNG state.

Save toàn bộ thư mục full run trong Kaggle Output. Khi mở session mới, copy toàn
bộ folder từ input saved output vào đúng original model path dưới working rồi:

```bash
!python scripts/kaggle_train.py --stage resume \
  --resume /kaggle/working/scivqa_qwen2b_ot_lora16_v256_s1105/last.pt \
  --output_root /kaggle/working
```

Giữ epochs3/worldsize2/effectivebatch16/data mounts/schedule. Đừng chỉ copy
`last.pt` rồi bỏ `run_config.json`, `metrics.jsonl`, source snapshot. Pilot bị cắt
giữa epoch không được resume thành full. Nếu full đã đủ3 epochs, không resume nữa.

## Bằng chứng local và giới hạn

**43 tests pass** trong lượt local (CPU/MPS và CPU/Gloo 2 process); CUDA-only
fixtures sẽ chạy thêm khi pytest thấy GPU thích hợp. Đã chạy worker spawn thật, native tiny logits/
generation/gradient/checkpoint regressions, command dry run trên toàn train/dev
local và compile/diff checks. Không tải lại 2B weights, không chạy full training,
không lặp 5.040 dev predictions vì các sửa chính nằm ở CUDA loading/runtime.
Artifacts: [`results/scivqa/final_pretraining_review_2xt4`](../../results/scivqa/final_pretraining_review_2xt4/).
Chưa có CUDA trên máy local. Lệnh torchrun đã được đối chiếu parser và recipe
đã kiểm tra; chưa chạy trên Kaggle. Chỉ report GPU pilot thật mới nghiệm thu hardware.

Nguồn API đối chiếu: [Transformers 5.5 bitsandbytes](https://huggingface.co/docs/transformers/v5.5.0/quantization/bitsandbytes),
[PyTorch 2.10 DDP](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html),
[AMP examples](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html).
Mục tiêu >80% vẫn cần matched dev experiments/seeds và final held-out evaluation.
