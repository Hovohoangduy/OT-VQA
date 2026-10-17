# Kiểm toán trước triển khai: SciVQA trên 2 × T4

**Trạng thái cập nhật: DONE.** Hoàn tất review và chuyển các sửa kỹ thuật vào code; nghiệm thu CUDA/quality nằm ở các cổng tiếp theo.

> Cập nhật implementation 15/10/2026: đã thêm correctness fixes, native OT wrapper, training infrastructure và persistent tests. Xem [runbook + trạng thái nghiệm thu](scivqa-implementation-runbook.md). CUDA/2 × T4 và mục tiêu quality >0.80 chưa được nghiệm thu; các nhận xét “chưa triển khai” bên dưới mô tả thời điểm review ban đầu.

Ngày 15/10/2026. Trạng thái: **DONE — REVIEW VÀ SỬA CODE ĐÃ HOÀN TẤT; GPU/QUALITY PENDING**.

Kế hoạch nguồn: [scivqa-accuracy-80-evidence-plan.md](scivqa-accuracy-80-evidence-plan.md). Phần cập nhật quan trọng: §8 và §14.

## Quyết định sau review

- Chuẩn phần cứng: **2 × T4, 16 GB/GPU**, cùng runtime, `nproc_per_node=2`.
- Effective batch mục tiêu **16**. Current model: 8/GPU × accumulation 1. VLM: 1/GPU × accumulation 8; capability đã triển khai và kiểm local.
- 15.120 training QA / global 16 = **945 optimizer updates/epoch**. Full dataset không có accumulation group cuối thiếu; subsets vẫn phải được kiểm tra.
- **Qwen3-VL-2B trước, 4B sau memory/quality pilot**; InternVL3-8B là optional reference/teacher, không thuộc critical path T4.
- OT tiếp tục là thành phần fusion nghiên cứu; có pretrained backbone không đồng nghĩa đã chứng minh giá trị OT.
- Mục tiêu final held-out **ROUGE-L F1 >0.80** được giữ nguyên; chưa có bằng chứng bảo đảm model/hardware này sẽ đạt mức đó.

## Phát hiện cần được xử lý, không chỉ đổi GPU count

| Ưu tiên | Phát hiện | Hành động đã đưa vào plan |
|---|---|---|
| P0 | Numeric mask chặn token số đúng; heuristic còn route cả nonvisual | Tắt hard mask mặc định, kiểm tokenizer/type schema và đáp án text/range |
| P0 | `figs_numb` là số panel, bị dùng như panel đích | Chỉ cung cấp count; target panel phải lấy từ question/evidence |
| P1 | Main embedding-only OT để DeepStack đi vòng | Kiểm và condition main + mọi native DeepStack levels, giữ grid/native interface |
| P1 | Zero gate và zero projection làm adapter không học | Chỉ zero một phía, test mở gate và gradient tới cost projections |
| P1 | AMP có thể cast bmm xuống half dù inputs `.float()` | FP32 region tắt autocast cho cost/solver/divisions cần thiết |
| P1 | Loss/token shift và PAD/EOS khác legacy decoder | Hợp đồng native answer span, labels -100 theo vị trí, không double shift |
| P1 | OT uniform control có thể phá marginal/mask | Feasible independent coupling giữ local marginals/dustbin; retrained native và capacity control |
| P1 | Full logits/attention/OT autograd bị bỏ sót khi ước lượng memory | Profile toàn step, worst shapes, optimizer state, cả hai GPU và validation |
| P1 | Current loader/optimizer/checkpoint giả định MiniLM/SigLIP | Factory/protocol chung dưới CLI hiện có; native quantized loading, adapter-only checkpoint |
| P2 | Dropout flag 0 vẫn để tied output 0.25 | Kiểm actual modules; tách ablation transformer/output head |
| P2 | Resume thiếu config/RNG từng rank/scaler | Version config/state và round-trip next-update tests |
| P2 | Diagnosis có metric khác official ROUGE | Dùng official scorer cho mọi quality comparison; numeric metric bổ sung |

## Thứ tự ticket khi bắt đầu code

### T0 — khóa input, metric và environment

- Lưu pinned environment riêng cho CUDA T4; không lấy local Transformers 5.0.0 ngoài requirements làm chuẩn.
- Lưu source/data/checkpoint/prompt/decode hashes. Chênh stored checkpoint 47,05% với current evaluation 44,89% phải được giải thích, hoặc đánh dấu lịch sử chưa tái lập và dùng baseline mới cùng snapshot để so các run.
- Khóa input track: figure+caption/choices, có hay không metadata format; unanswerability gold không được dùng làm câu trả lời oracle.
- Không chặn correctness coding chỉ vì không khôi phục được môi trường lịch sử không có manifest.

### T1 — correctness fixes và regression tests

- Sửa numeric decoding/routing và compound prompt riêng từng thay đổi; prompt/decode versions cho checkpoint cũ.
- Dùng chung builder cho train/eval/predict; restore persistent tests.
- Kiểm gold-token coverage, binary/multilabel answers, PAD/causal masks, unchanged gold references, scorer consistency, checkpoint legacy round-trip.
- Nghiệm thu: tất cả fixtures đúng; full-dev report same-snapshot trước/sau. Không đặt điều kiện sửa lỗi bắt buộc tăng ROUGE để được giữ.

### T2 — training infrastructure tối thiểu cho T4

- AMP, accumulation, correct example/token weighting, `no_sync`, clipping/unscale, scheduler successful-update counting.
- Native prepared batch/factory/output protocol, freeze policy, optimizer groups, checkpoint adapter/scaler/config/RNG.
- Unit integration trên CPU/MPS khi hỗ trợ; CUDA/DDP fixture bắt buộc trên T4 trước run dài.
- Nghiệm thu: microbatch/large-batch deterministic equivalence; resume-next-update; overflow/last-group behavior; không đổi mục tiêu bằng normalization vô thức.

### T3 — native 2B + OT contextualizer

- Native wrapper no-op parity, original image processor, query-only frozen MiniLM, main + DeepStack mapping.
- Labels answer-only và native causal shift; generation KV/RoPE/request-state correctness; no answer leakage vào Q/T.
- Gradient flow qua frozen LLM tới OT, không dead gate; native LoRA/device mapping kiểm riêng.
- Nghiệm thu: finite forward/backward, correct shapes/order, no-op logits/generation parity, checkpoint reload đủ OT+LoRA.

### T4 — pilot 2 T4, không cam kết memory trước khi đo

- 100–200 optimizer updates + validation; worst permitted image/context/answer lengths.
- Visual budget 256 → 512 merged tokens; total cap 2.048 ban đầu, không truncate answer/question/choices silently.
- Log allocated/reserved/NVML, overflow, seconds/update, data/vision/OT/LLM/eval breakdown; khoảng 10–15% usable VRAM headroom mỗi GPU.
- Với 4B: cùng evidence/global batch; nếu không vừa hoặc quá quota, giữ 2B để phát triển và báo đề xuất GPU lớn theo nhu cầu quality.

### T5 — chỉ training variants có khả năng trả lời giả thuyết

- Current branch: baseline + no auxiliary và tối đa một visual-memory/head control theo evidence; không mở full grid.
- Pretrained branch: native LoRA / OT+LoRA / parameter-matched adapter, cùng data order/updates/evidence; finals reset cùng base.
- Plan interventions giữ feasibility, đánh giá overall + hai infinite groups; repeat seeds 1105/42/2026 cho winner.
- OCR/tiles/external-data/distillation/retrieval triển khai từng nhánh sau khi model cần đúng loại evidence đó. Strict và external-paper tracks không gộp.

## Nghiệm thu quality theo mức kỳ vọng

| Mức | Tiêu chí |
|---|---|
| Tính đúng | Không chặn đáp án hợp lệ, không chỉ sai panel, không leak gold, native/token semantics đúng |
| Tiến bộ thực nghiệm | Full-dev gain và CI/seeds; báo cả các nhóm bị giảm, không chỉ overall |
| Giá trị của OT | OT hơn control cùng backbone/data/updates; can thiệp plan có ích và capacity control không giải thích toàn gain |
| Đạt mục tiêu | Strict dev >0.80, recipe frozen, final held-out test >0.80; chỉ dev vượt thì chỉ báo dev |
| Khả thi trên T4 | Không OOM/overflow trong pilot + validation, model/evidence budget hợp lý, wall time vừa quota đã đo |

Các threshold memory, hyperparameters và thời gian trong plan là **đề xuất thí nghiệm**, không phải benchmark đã đo. Không có “đảm bảo 80%” trước training; cơ chế kiểm soát là dừng sớm giả thuyết không có bằng chứng và phân bổ compute cho khả năng đọc/sinh đáp án còn thiếu.

## Bằng chứng của lượt review này

- [Focused checks](../../results/scivqa/review_20261015/second_review_checks.json): AST/hash 28 sources không đổi, parser resume defaults, nonvisual routing, head dropout, dead gate, autocast dtype và numeric scoring fixture.
- [Validation report đã có](../../results/scivqa/review_20261015/current_checkpoint_dev_report.json): 1.680 QA, 240 figures, ba conditions.
- [Intervals](../../results/scivqa/review_20261015/verification_and_intervals.json): đã recompute 5.040 ROUGE values; CI group theo figure cho hai delta nhỏ đều chứa 0.
- Không thay production Python/config/checkpoint/dataset trong lượt này. Không chạy lại full-dev inference khi source hash không đổi. Không chạy training/CUDA/DDP và không báo pytest pass khi chưa có test suite.

**Trạng thái cho bước tiếp theo:** T0/T1 và code T2/T3 đã triển khai, kiểm chứng local.
Review này đã hoàn tất. CUDA portions của T2/T3, pilot T4 và quality T5 chưa nghiệm thu;
không đánh dấu toàn bộ mục tiêu accuracy là done.
