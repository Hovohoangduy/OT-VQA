# Review OT-VQA và kế hoạch đạt ROUGE-L F1 > 80% trên SciVQA

**Trạng thái cập nhật: PARTIALLY_COMPLETED.** Đã hoàn tất implementation prototype và correctness/local checks; kế hoạch accuracy >0.80 còn mở.

> Cập nhật implementation 15/10/2026: đã thêm correctness fixes, native OT wrapper, training infrastructure và persistent tests. Xem [runbook + trạng thái nghiệm thu](scivqa-implementation-runbook.md). CUDA/2 × T4 và mục tiêu quality >0.80 chưa được nghiệm thu; các nhận xét “chưa triển khai” bên dưới mô tả thời điểm review ban đầu.

Ngày: **15/10/2026**. Trạng thái: **ĐÃ TRIỂN KHAI PROTOTYPE + KIỂM CHỨNG LOCAL; CHỜ GPU/QUALITY EXPERIMENTS**.

**Revision 2 — sau review trước triển khai:** tài nguyên chuẩn **2 × T4, 16 GB/GPU**, effective batch mục tiêu 16. Ưu tiên Qwen3-VL-2B để xác minh pipeline, 4B là ứng viên chất lượng phải qua memory pilot; 8B không thuộc lộ trình huấn luyện mặc định trên T4. Xem §14 cho các sửa đổi kỹ thuật quan trọng sau lần rà soát thứ hai và [checklist triển khai](scivqa-preimplementation-review-2xt4.md).


## Tiến độ implementation đã xác nhận

- [x] Correctness fixes, prompt/decode versions và checkpoint legacy compatibility.
- [x] Native Qwen3-VL OT adapter cho main + DeepStack, query-only, answer-only labels.
- [x] AMP/accumulation/DDP infrastructure, checkpoint OT+LoRA/scaler/RNG và local regressions.
- [x] Rà soát NF4 visual scope, worker spawn, pilot gates, log và preflight.
- [x] Lệnh torchrun trực tiếp trên Kaggle 2 × T4; đã xóa notebook file theo yêu cầu.
- [ ] Full pretrained CUDA/NF4/native parity và pilot 2 × T4/worst shapes.
- [ ] Controlled quality comparisons/seeds và mục tiêu dev/test >0.80.

## 1. Quyết định đề xuất

**Không tiếp tục lấy việc chỉnh epsilon, thêm FGW/UOT, tăng patience hoặc thêm prompt dài làm chiến lược chính.** Code có những lỗi và hạn chế cần sửa trước, nhưng bước nhảy từ khoảng 45–47% tới trên 80% cần năng lực đọc biểu đồ, OCR và sinh câu trả lời đã được tiền huấn luyện. Hướng ưu tiên là **backbone VLM đã tiền huấn luyện + OT adapter có vai trò thực sự trong fusion + tinh chỉnh SciVQA + bằng chứng ablation**.

Ràng buộc đã xác nhận với người dùng:

- Ưu tiên độ chính xác; chấp nhận pretrained VLM/backbone/teacher.
- **Giữ OT là thành phần fusion chính trong mô hình nghiên cứu**.
- Tài nguyên hiện có đã được người dùng điều chỉnh thành **2 × T4 trên Kaggle notebook, 16 GB/GPU**. Dùng hai process DDP khi cả hai GPU cùng runtime; luôn kiểm tra `torch.cuda.device_count()` và VRAM thực tế trước launch.
- Có thể đề xuất GPU khác nếu tài nguyên hiện có hạn chế đường đạt chất lượng cao.

Hai luồng công việc, thực hiện tuần tự theo các cổng quyết định bên dưới:

1. **Luồng sửa đúng và đo đúng:** xử lý ràng buộc giải mã, metadata, tái lập baseline, khôi phục regression tests và chạy một số ablation nhỏ. Mục đích là xác định mức trần có thể cải thiện bằng sửa lỗi; không mặc định rằng sửa lỗi sẽ đưa kết quả tới 80%.
2. **Luồng tăng năng lực:** khảo sát VLM mạnh trên dev, chọn backbone phù hợp, tích hợp OT trước đường vào thị giác của bộ sinh đã tiền huấn luyện; bổ sung dữ liệu và bằng chứng OCR/biểu đồ. Đây là đường chính nhắm tới 80%.

**80% là mục tiêu nghiệm thu bằng số đo, không phải lời hứa của kiến trúc.** Không đánh dấu hoàn tất khi chỉ qua forward/backward, checklist hoặc có flag mới.

## 2. Phạm vi và nguồn bằng chứng

Đã đọc đường thực thi chính: `model/optimal_transport.py`, `model/vqa_model.py`, `model/features_extraction.py`, `model/decoder_model.py`, `train.py`, `test.py`, `predict.py`, `configs/`, các utility về dữ liệu, image transform, loss, metric, checkpoint, và các script diagnosis/comparison. Xem thêm downloader SciVQA, các utility tải dataset khác ở phạm vi liên quan, README, SCIVQA_TUNING, các báo cáo SciVQA đã lưu, metadata hai checkpoint và **12 kế hoạch trong `.agent/plans`**. Các plan về kỹ năng, tổ chức và đề cương được phân loại là công việc quy trình/tài liệu; không coi chúng là thực nghiệm tăng điểm.

Các kiểm chứng mới và mã tái lập nằm trong [`results/scivqa/review_20261015/`](../../results/scivqa/review_20261015/). Không chỉnh source training/model, không thay dữ liệu, không huấn luyện lại và không chạy thêm trên test trong đợt review này.

### 2.1 Ba nguồn kết quả phải tách riêng

| Nguồn | Mô hình / cấu hình đã biết | Kết quả | Diễn giải đúng |
|---|---|---|---|
| Log người dùng đính kèm | Peak decoder LR 2.5e-5, encoder LR 2e-6; example-normalized train loss; 25 epoch dự kiến | Best dev **0.4536 ở epoch 12**; dừng epoch 22 | Đây là run mới người dùng yêu cầu phân tích; không có đủ manifest để xác nhận tất cả flags |
| `data/weights/scivqa.pt` | SigLIP Base 384, pad, visual memory, context, tied output, MHOT 4 heads; question encoder mở; checkpoint epoch 20 | `best_metric.value = 0.4705307503`, epoch 20 | Là metadata checkpoint local; **không khớp log epoch 12**; giá trị lưu trong checkpoint chưa đồng nghĩa điểm tái lập bằng code hiện tại |
| `data/weights/OT_SciVQA.pt` và báo cáo cũ | MobileViTV2-2.0, kiến trúc/tiền xử lý cũ | Full-dev 0.45409; đổi ảnh 0.44816 | Bằng chứng yếu về visual grounding của **mô hình cũ**; không gán kết luận này cho SigLIP mới |

Không có `metrics.jsonl`/`run_config.json` đi kèm hai checkpoint local để đối chiếu toàn bộ lịch sử. Revision encoder của `scivqa.pt` còn là `null`. Phải lưu hash checkpoint, CSV, source snapshot, prompt version và decode version trong các run tiếp theo.

### 2.2 Log mới nhất thực sự cho thấy gì?

| Đại lượng | Đầu run | Điểm cần chú ý | Cuối run |
|---|---:|---:|---:|
| Dev ROUGE-L F1 | 0.2836, epoch 1 | **0.4536, epoch 12** | 0.4346, epoch 22 |
| Train generated ROUGE-L F1 | 0.2971 | 0.4888, epoch 12 | 0.4968 |
| Dev CE | 2.2966 | **min 1.1234, epoch 17** | 1.1315 |
| Train OT loss | 0.4580 | 0.2380, epoch 12 | 0.2186 |

![Training diagnosis](../../results/scivqa/review_20261015/training_diagnosis.png)

- Epoch 12 tới 22: OT loss giảm khoảng 8,2%, train loss giảm, nhưng dev ROUGE giảm **1,90 điểm phần trăm**. Việc giảm OT cost chưa chuyển thành tăng chất lượng câu trả lời.
- Chênh train/dev ROUGE ở epoch 12 khoảng **3,52 điểm phần trăm**. Train generated metric có thể chỉ trên tập con; chưa biết cỡ mẫu của log này. Không dùng chênh này để chứng minh chắc chắn underfitting hoặc loại trừ overfitting.
- Train loss có smoothing, dropout, OT term và normalization theo example; dev CE không smoothing và normalization theo token. **Không so trực tiếp 2.46 với 1.14 để kết luận lỗi/overfit.**
- `best_loss.pt` có thể hữu ích để nghiên cứu calibration; nó không tự động là checkpoint sinh câu trả lời tốt hơn. Mục tiêu ROUGE phải chọn checkpoint theo generated ROUGE.
- Patience đã cho thêm 10 epoch sau đỉnh mà không vượt đỉnh. Không có bằng chứng rằng tăng patience tiếp là đòn bẩy chính.
- BERTScore khoảng 0.94 không nghĩa là trả lời đúng 94%. Metric semantic dễ cho điểm cao với chuỗi ngắn sai số/nhãn; ROUGE và exact/numeric accuracy vẫn cần đo độc lập.

### 2.3 Kết quả kiểm chứng mới

Đã chạy **toàn bộ 1.680 QA / 240 figures** của dev với `scivqa.pt`, cache frozen image features, gọi đúng question encoder, OT fusion và generation của production. Không tính BERTScore trong lượt này. Xem [`audit.json`](../../results/scivqa/review_20261015/audit.json), [`report`](../../results/scivqa/review_20261015/current_checkpoint_dev_report.json) và [`per-example predictions`](../../results/scivqa/review_20261015/current_checkpoint_dev_predictions.csv).

| Nhóm — mỗi nhóm 240 mẫu | Code/constraint hiện tại | Bỏ mọi step constraint | Thay bằng figure khác, giữ constraint |
|---|---:|---:|---:|
| Binary nonvisual | 61.25% | 61.25% | 57.92% |
| Binary visual | 56.67% | 56.67% | 51.67% |
| Multiple-choice nonvisual | 40.29% | 40.29% | 42.18% |
| Multiple-choice visual | 32.52% | 32.52% | 31.48% |
| Infinite nonvisual | **6.80%** | **8.28%** | 8.12% |
| Infinite visual | **17.96%** | **18.43%** | 14.65% |
| Unanswerable | 98.75% | 98.75% | 98.33% |
| **Overall ROUGE-L F1** | **44.89%** | **45.17%** | **43.48%** |

Các kết luận được phép rút ra:

1. Bỏ constraints chỉ tăng overall **0,279 điểm phần trăm** trên checkpoint này. Vì các nhóm finite không đổi prediction score, lợi ích quan sát nằm ở nhóm infinite. Numeric mask là lỗi thật, nhưng sửa nó **không giải quyết phần lớn khoảng cách tới 80%**.
2. Đổi ảnh làm **34,52% predictions thay đổi**, nhưng overall chỉ giảm **1,413 điểm phần trăm**. Mô hình SigLIP nhạy ảnh hơn bản MobileViT cũ (12,38% trong report cũ), song đây không phải controlled architecture comparison vì weights/prompt/training khác. Useful visual grounding vẫn yếu qua thử nghiệm thay ảnh này; không kết luận hoàn toàn bỏ qua ảnh.
3. Metadata checkpoint ghi 47,05%, còn evaluation current code đạt 44,89%. **Chênh 2,16 điểm phần trăm chưa được giải thích**: training snapshot/prompt/constraint version không có, local Transformers **5.0.0** nằm ngoài `requirements.txt` hiện yêu cầu `>=5.5,<6`. Các probe/inference vẫn chạy thành công, nhưng cần tái lập trong environment khớp training trước khi kết luận regression từ kiến trúc.
4. Mean matched mass **0.599918** sát biên thấp 0.6; max row residual **1.516e-3**, max column residual **1.043e-7**. Sai số row cho thấy không thể gọi mọi plan đã hội tụ tới tol=1e-4: solver có thể hết 20 iterations. So 20/50/100 iterations trên cùng cached features và đo answer changes trước khi tăng mặc định. Matched mass thấp hơn 0.6 một chút là sai số feasibility, không phủ định công thức biên của plan hội tụ.
5. Audit input constraints: **203/278** dev samples bị route numeric có canonical gold-token sequence bị chặn (**73,02%**); train là **1.795/2.440**. Trong 278 mẫu dev bị route numeric có **76 mẫu gold dạng text/range/name** theo kiểm tra lexical của audit. Đây là lỗi heuristic routing; không phải phép đo numeric accuracy hoặc một trần ROUGE lý thuyết.
6. Prompt đầy đủ hiện tại dài tối đa **494 tokens train / 367 dev**, không mẫu nào quá 512. Vì vậy tăng token limit thêm không giải quyết truncation cho revision local hiện tại.
7. Probe ordinary OT: đổi nội dung PAD không đổi memory (max difference 0), backward hữu hạn, row/column residual khoảng 1.62e-6/5.96e-8 trên fixture nhỏ. Probe FGW: chỉ thêm padding làm active cost lệch **0.126662** ở fixture alpha=1. Positive-only auxiliary loss đạt **0** với constant directions và matched mass ~0.6. Các probe này kiểm chứng tính chất code; không chứng minh mô hình đã học collapse.

Bootstrap 2.000 lần theo 240 figure cho delta bỏ constraint có 95% CI **[-0,094; +0,673] điểm phần trăm**; delta ảnh đúng trừ ảnh khác có CI **[-0,245; +3,115] điểm phần trăm**. Cả hai khoảng đều chứa 0: **chưa đủ bằng chứng thống kê để khẳng định gain hoặc useful image contribution ở overall** từ một checkpoint và phép thay ảnh này. Tỷ lệ đổi predictions là một quan sát khác với tăng độ chính xác. Đã tính lại toàn bộ **5.040 giá trị ROUGE** từ predictions và kiểm tra khớp report; xem [`verification_and_intervals.json`](../../results/scivqa/review_20261015/verification_and_intervals.json).

Checkpoint SHA256: `fdd178e4c6b78a710ec859fe2984c086723ce08f55c32321b86c71c2d0b72d93`. Dev CSV SHA256: `6322dc4b28569278e2ad7d070c7f7ea8fdcca108c36ab792c8c137e7cc8593ce`. Full cached review mất khoảng 478 giây trên local MPS; **không dùng thời gian này làm benchmark inference** vì đã cache vision và chạy ba conditions.

## 3. Review code: phát hiện và thứ tự sửa

### P0-1. Numeric constraint chặn token hợp lệ của tokenizer

**Vị trí:** `model/vqa_model.py:207`, `model/vqa_model.py:270`; trigger được thêm ở `utils/vqa_dataset.py:65`.

`numeric_token_ids` chỉ lấy các token của từng ký tự `0123456789.-+`. Với tokenizer MiniLM thực tế:

| Đáp án | WordPiece | Vấn đề |
|---|---|---|
| `10` | `10` | Token `10` không nằm trong allowlist |
| `0.75` | `0`, `.`, `75` | Token `75` bị chặn |
| `-12.5` | `-`, `12`, `.`, `5` | Token `12` bị chặn |
| `1e-3` | `1`, `##e`, `-`, `3` | Exponent bị chặn |
| `10%` | `10`, `%` | Cả token nhiều chữ số và `%` bị chặn |

Đây là lỗi kỹ thuật có thể tái hiện; không cần đợi retrain. Model có thể học đúng nhãn ở teacher forcing nhưng bị cấm sinh nhãn đó khi đánh giá. Ghép token `1`, `0` không tương đương token `10`: decoder WordPiece thường cho `1 0`, và postprocessor hiện không khôi phục chuỗi đó thành `10`.

**Sửa đầu tiên:** tắt numeric hard constraint; đo lại checkpoint giữ nguyên mọi yếu tố khác. Nếu muốn giữ constraint về sau, xây automaton theo **chuỗi được tokenizer decode**, nhận số nhiều chữ số, phần thập phân, dấu, exponent, đơn vị khi task yêu cầu; mỗi allowed transition phải giữ prefix hợp lệ. Không thay bằng regex trên từng token rồi cho rằng đã đủ.

**Tiêu chí:** canonical target-token path phải được chấp nhận cho toàn bộ các mẫu được route numeric trong train/dev nếu gold là số thuộc grammar; trường hợp gold không phải số phải được route đúng. Coverage chỉ là kiểm tra tính đúng, không phải dự báo mức tăng ROUGE.

### P0-2. Dùng số lượng subplot làm chỉ số subplot đích

**Vị trí:** `utils/vqa_dataset.py:41` và `:118`.

`figs_numb` được dùng để thêm “Navigate to subfigure N”. Dataset card định nghĩa nó là **số lượng subfigures**, không phải vùng được hỏi. Ví dụ local: một figure có `figs_numb=3`, câu hỏi visual nói `(a)`, câu nonvisual nói `(b)`; code lại dẫn tất cả tới subfigure 3. Dev có **616/1.680 dòng compound**. Sửa thành thông báo trung tính về số panel; chỉ chọn panel từ câu hỏi, nhãn ảnh hoặc bằng bộ chọn panel đã được kiểm chứng. [Dataset card chính thức](https://huggingface.co/datasets/katebor/SciVQA).

`bool("False")` cũng là True nếu một CSV khác lưu chuỗi. CSV local hiện được pandas đọc đúng kiểu bool, nên đây là nguy cơ robustness cần test, chưa phải lỗi đã xảy ra trên dữ liệu này.

**Tiêu chí:** prompt không bịa panel đích; ví dụ `(a)/(b)/lower left` được bảo toàn; train/eval/predict dùng cùng builder và prompt version.

### P1-1. Chưa có năng lực tiền huấn luyện sinh đáp án tương ứng với target

**Vị trí:** `model/features_extraction.py:37`, `:84`; `model/vqa_model.py:125`; `model/decoder_model.py`.

- SigLIP luôn `requires_grad_(False)` và chạy `torch.no_grad()`. Tăng `encoder_lr` chỉ tác động nhánh text/answer, **không tinh chỉnh ảnh**.
- Decoder Transformer hai lớp được khởi tạo mới. Lấy embedding MiniLM và tie output không biến nó thành decoder ngôn ngữ đã pretrained.
- SigLIP vision feature + MiniLM đến từ hai hệ pretrained khác nhau; cosine cost chỉ có nghĩa sau khi projection học alignment có ích. SigLIP không tự căn chỉnh sẵn với token-space MiniLM.
- Pad 384 giữ biên ảnh nhưng có thể làm nhãn trục cực nhỏ, nhất là compound/wide figure. Thêm prompt không phục hồi được chữ/số đã mất độ phân giải.

**Quyết định:** chuyển nguồn lực chính sang pretrained generative VLM có năng lực chart/OCR, thay vì kỳ vọng OT tự tạo năng lực OCR từ 2.160 training figures.

### P1-2. Auxiliary OT loss có nghiệm suy biến và không có negative pairs

**Vị trí:** `model/optimal_transport.py:247`; `train.py:75`.

Loss hiện là `mean(sum(T_local * C_local))`, không phải đầy đủ entropic OT objective, không có negative image/text và không có supervision vùng/đối tượng. Khi các vector cùng hướng, cosine cost bằng 0 cho mọi cặp và auxiliary loss cũng bằng 0. Probe trong evidence tái hiện nghiệm này; **đó là khả năng của objective, chưa phải chứng minh checkpoint hiện tại đã collapse**.

Với balanced marginals và dustbin mỗi bên có mass `d`, đặt mass dustbin-to-dustbin là `x` thì local-local mass bằng `1 - 2d + x`; `0 <= x <= d`. Với `d=0.2`, local matched mass nằm trong **[0.6, 0.8]** nếu solver hội tụ. Vì vậy mass ~0.6 không tự chứng minh align tốt, cũng không thể kết luận model “đổ toàn bộ vào dustbin”.

**Thí nghiệm ưu tiên:** cùng backbone/input/seed, so `ot_loss_weight=0` với `0.05`. Log variance feature, positive-negative retrieval margin, entropy, gradient norm CE/OT và cosine giữa hai gradient trên parameters chung. Nếu auxiliary không tăng answer quality thì bỏ nó khỏi recipe, vẫn giữ **OT fusion**.

Nếu cần alignment loss mới, dùng contrastive/ranking có negative hợp lệ hoặc supervision bbox trên synthetic/OCR. In-batch negatives phải loại **cùng image, cùng paper và trường hợp cùng ngữ nghĩa/đáp án** phù hợp; 7 câu trên một figure không phải 7 cặp image-negative. UNITER là tiền lệ dùng OT trong hệ multi-objective pretrained; không phải bằng chứng rằng positive-only cost hiện tại sẽ tăng SciVQA. [UNITER](https://arxiv.org/abs/1909.11740).

### P1-3. FGW tùy chọn phụ thuộc chiều padding và chưa chứng minh ý nghĩa hình học

**Vị trí:** `model/optimal_transport.py:11`, `:28`, `:189`.

`D_Q` được tạo từ `linspace(0,1,num_q_tokens)`. `num_q_tokens` là chiều dài batch sau padding. Cùng một câu có thể nhận khoảng cách giữa token hợp lệ khác nhau khi ghép với câu dài hơn. Probe giữ nguyên active plan và chỉ thêm hai cột padding làm active FGW cost thay đổi. Padded positions có zero transport không giải quyết việc **active positions bị rescale**.

Ngoài ra:

- Khoảng cách Euclid patch và khoảng cách thứ tự từ không thể mặc nhiên coi là cùng cấu trúc. Từ gần nhau trong câu không đồng nghĩa thuộc vị trí gần nhau trên đồ thị.
- Hiện có một lượt sửa cost theo plan cũ và giải lại, không phải bằng chứng đạt nghiệm tối ưu FGW hội tụ.
- `plan.detach()` là lựa chọn xấp xỉ gradient cần công bố/đối chiếu, không phải tự động sai.
- Với `fgw_alpha=0` như mặc định hiện tại, lỗi này **không giải thích baseline**. Chỉ có thể liên quan log mới nếu manifest xác nhận đã bật FGW.

**Quyết định:** giữ FGW tắt trong recipe chính. Chỉ mở lại sau khi có test padding/batch invariance, alpha/tau validation, oracle nhỏ đối chiếu công thức/POT và ma trận cấu trúc có nghĩa từ OCR-box/panel/đồ thị dữ liệu. [POT](https://pythonot.github.io/gen_modules/ot.gromov.html).

### P1-4. Nhiều thành phần được đánh dấu done nhưng chưa có trong model

Không thấy OT option candidate scorer, dedicated binary head, learned unanswerable gate, numeric regression head hoặc Smooth-L1 numeric loss trong đường forward/training. Hiện có **mask token khi generate**, không tương đương các module đã nêu trong plan.

MHOT có chia projection thành bốn subspaces và giải bốn plan, nhưng **không có supervision hay cơ chế bắt buộc** các head lần lượt học axis/curve/legend/OCR. Không thể dùng tên head trong sơ đồ làm bằng chứng specialization.

### P1-5. Metadata/prompt và đường inference chưa có hợp đồng thống nhất

`VQADataset` đọc `qa_pair_type` để tạo format prompt; `predict.py` không làm việc đó và cũng không nhận compound/panel metadata. Vì vậy không đúng khi nói hiện tại qa-type “chỉ dùng cho reporting”. Type có thể là metadata được phép trong benchmark, nhưng phải báo rõ model nhận gì. **Không truyền nhãn `unanswerable` để trả thẳng đáp án chuẩn** trong mô hình được đánh giá năng lực answerability.

“Infinite answer set” không có nghĩa là tất cả đáp án đều số; local có `S-shaped`, tên phương pháp, nhãn trục và thực thể. “Non-visual” trong taxonomy cũng không đồng nghĩa đáp án nằm trong caption: vẫn có thể cần đọc axis label, legend hoặc thực hiện phép tính. Do đó nhận định cũ “60% câu có đáp án ngay caption” không có bằng chứng; ba nhóm nonvisual bằng 3/7 số mẫu cũng không chứng minh điều đó.

**Sửa:** tạo input contract gồm question, choices, caption, figure metadata, optional predicted answer format; tách nhãn phục vụ reporting. Với benchmark có cung cấp qa-type, lưu một track metadata rõ ràng và một control không dùng type. Không đưa annotation ID, paper ID hay answerability gold vào nội dung model.

### P2. Huấn luyện, tái lập và tốc độ

1. `train.py` chưa có AMP, gradient accumulation hoặc gradient checkpointing. Thêm chúng trước khi tăng kích thước backbone. Sinkhorn/cost normalization/logsumexp giữ FP32; CUDA T4 dùng FP16 autocast + GradScaler, unscale trước clipping, scheduler chỉ step khi có optimizer update hợp lệ.
2. AdamW đang áp weight decay chung cho bias và normalization. Tách no-decay nhóm này; đo tác động như một optimizer change riêng, không gọi đây là nguyên nhân chính khi chưa có ablation.
3. Checkpoint chỉ lưu training config `loss_normalization`, chưa lưu đầy đủ auxiliary weights/prompt/decoder constraints. Resume một checkpoint cũ với preset mới có thể thay `ot_loss_weight` ngoài ý muốn. Một số defaults context/FGW/tau cũng cần phân biệt explicit flags với preset. Lưu toàn bộ resolved config và kiểm tra khi resume.
4. Dual-best bookkeeping khởi tạo lại một best phụ khi resume theo metric còn lại; cần khôi phục cả hai best để tránh ghi đè bằng checkpoint kém hơn. Đây là lỗi bookkeeping, không phải cơ chế tăng accuracy.
5. `tools/diagnose_scivqa_breakdown.py --compute_bertscore` dùng mặc định builder `bert-base-uncased`/rescale, trong khi SciVQA pipeline dùng `roberta-large`/không rescale. Đồng bộ settings/hash trước khi so số.
6. `infinite_answer_errors_count` hiện tính **mọi** mẫu infinite, kể cả đúng; sửa filter và phân loại OCR/numeric/text/routing/answerability riêng. Không dùng count hiện tại làm số lỗi thực.
7. `test.py` luôn chạy BERTScore cho SciVQA; rank 0 phải chờ scorer/validation và các rank khác đợi. Cho phép ROUGE-only trong vòng screening, BERTScore ở checkpoint tốt nhất/final; không đổi metric chọn model.
8. Frozen image encoder được tính lại cho 7 QA/figure mỗi epoch; cache feature theo image+transform+encoder revision khi encoder frozen. Cache không hợp lệ nếu unfreeze hoặc augmentation thay pixel. Chỉ xem đây là tiết kiệm compute.
9. Không tăng SigLIP384 sang 512 chỉ bằng flag: cần xác nhận positional embedding/interpolation và processor của backbone. `image_size` được constructor chấp nhận không bảo đảm backbone chạy đúng resolution.
10. MiniLM là sentence encoder, không phải bộ instruction-following pretrained. Card cho biết pretraining sentence pairs dùng giới hạn 128 tokens; input 512 có thể hợp lệ kỹ thuật nhưng không chứng minh có năng lực làm theo prompt phức tạp. [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2).

### Những phần nên giữ

Teacher forcing hiện dịch BOS/target đúng; causal mask ngăn nhìn token tương lai; PAD/special question tokens bị loại khỏi OT marginal; Sinkhorn chạy FP32; kiểm tra loss hữu hạn và gradient clipping đã có. `compare_scivqa.py` join theo ID và kiểm tra thiếu/trùng là tốt. Example-normalized CE là giả thuyết hợp lý để cân bằng câu trả lời ngắn/dài. Full-figure padding là nền tảng hợp lý. Probe OT thường cần xác nhận masking/gradient tiếp tục đúng sau mỗi thay đổi. Không viết lại các phần này chỉ vì muốn kiến trúc trông mới hơn.

## 4. Kiểm toán các kế hoạch cũ

| Kế hoạch | Những gì thấy trong code/bằng chứng | Điều cần sửa trong kết luận |
|---|---|---|
| `scivqa-improvement-plan.done.md` | SigLIP, padding, context, tied output và regularization đã có | Chưa có controlled proof từng thay đổi tăng điểm; frozen SigLIP không mặc nhiên đọc số tốt; nhận định 60% caption không được chứng minh |
| `ot-enhancement-plan.done.md` | MHOT, positive auxiliary loss, adaptive iterations đã có | Không có bằng chứng heads tự chuyên biệt; không có benchmark xác nhận giảm thời gian 30–40%; memory một tensor không phải toàn bộ backward graph |
| `scivqa-breakthrough-plan.done.md` | Dual best checkpoint và breakdown CLI đã có | “CE giảm nên không overfit”, “patience là bẫy” và mốc >55% chưa được log mới xác nhận; chưa thấy dedicated task heads |
| `scivqa-breakthrough-80-rouge-plan.done.md` | FGW/UOT tùy chọn, metadata prompts và decoding masks có một phần | Candidate scorer/gate/numeric loss chưa có; panel metadata dùng sai; không có thí nghiệm đạt 68/75/80%; đánh dấu done không có nghĩa accuracy đạt |
| `scivqa-checkpoint-load-and-bias-diagnosis.md` | Adapter SigLIP prefix đã có; load probe local thực sự chạy được | Vẫn pending trên tài liệu; cần report mới có hash, không suy từ báo cáo MobileViT |
| `cleanup-test-folders-plan.done.md` | Không còn thư mục `tests/` trong working tree | Xóa regression tests sau khi pass làm mất khả năng nghiệm thu lặp lại; cần khôi phục |
| `test-cleanup-skill-plan.done.md` | Quy trình archive khuyến khích xóa test | Không áp dụng cho công việc mới; theo AGENTS hiện tại giữ tests/artifacts |
| `ot-skills-plan.done.md` | Có hướng dẫn OT/architecture/performance | Đây là tài liệu; các con số gain trong ví dụ không phải số đo của repo |
| `plan-verification-skill-plan.done.md` | Có checklist quy trình | Checklist hiện cho qua khi không có tests, không kiểm chứng accuracy/hội tụ FGW |
| `plan-organization-and-done-lifecycle.done.md` | Có tổ chức lifecycle plan | Tách “implemented”, “validated correctness”, “validated quality”; chỉ trạng thái cuối có measured gain |
| `thesis-proposal-alignment.done.md` | Định hướng OT core và đề cương | Không phải evidence thực nghiệm SciVQA; giữ ràng buộc OT core |
| `proposal-refinement-and-proofreading.done.md` | Chỉnh nội dung đề cương | Không ảnh hưởng inference/training; không tính vào số cải tiến thuật toán |

Không đổi tên/xóa các kế hoạch lịch sử trong đợt review này. Kế hoạch mới nằm ở `.agents/plans` vì `.agent/` được AGENTS hiện tại xác định là archive; `docs/*` và `results` hiện đang được Git ignore, nên giữ kế hoạch trong `.agents/plans` để có thể quản lý cùng repository.

## 5. Chốt metric, dữ liệu và điều kiện >80%

### 5.1 Hợp đồng đánh giá

- Metric chính: **ROUGE-L F1 macro theo câu trả lời**, `rouge-score`, stemming bật, thang [0,1]. Báo thêm ROUGE-1 F1, exact match, numeric exact/relaxed accuracy và BERTScore.
- Không gọi BERTScore hoặc mean(ROUGE-1, ROUGE-L, BERTScore) là ROUGE. Không đổi scoring để đạt 80%.
- Giữ nguyên official train/dev/test và gold. Chỉ dùng train để update trọng số; dev để chọn thiết kế/threshold/checkpoint; test dùng sau khi chốt.
- Bộ local tải JSON **tháng 7/2025**, trong khi dataset card còn lưu **competition archive tháng 3–4/2025**. Score leaderboard là tham khảo; phải đối chiếu version/IDs trước khi tuyên bố vượt paper cùng điều kiện.
- ROUGE hiện khớp stemmer và mean của [official scoring.py](https://github.com/esborisova/SciVQA/blob/main/src/eval/scoring.py). Không cần sửa metric để giải thích khoảng cách 30+ điểm.
- SciVQA có 7 câu/figure. Bootstrap confidence interval và split nội bộ phải group theo **figure**, ưu tiên kiểm tra thêm paper. Audit train/dev hiện không trùng image/annotation ID nhưng có **72 paper ID** giao nhau; không tự ý đổi official split. Với external pretraining/RAG phải chống duplicate figure và xem xét paper overlap.

### 5.2 Ngân sách điểm theo nhóm

Mỗi nhóm dev hiện có 240 mẫu, nên:

`R_overall = (R_unanswerable + R_bin_v + R_bin_nv + R_mc_v + R_mc_nv + R_inf_v + R_inf_nv) / 7`.

Ví dụ **điều kiện mục tiêu**, không phải dự báo:

| Nhóm | Mục tiêu tham chiếu |
|---|---:|
| Unanswerable | 0.98 |
| Binary visual | 0.90 |
| Binary nonvisual | 0.90 |
| Multiple-choice visual | 0.85 |
| Multiple-choice nonvisual | 0.85 |
| Infinite visual | 0.58 |
| Infinite nonvisual | 0.68 |
| Trung bình | **0.82** |

Tăng unanswerable từ 0.96 lên 1.0 chỉ thêm **0,57 điểm phần trăm tổng**. Nếu bốn finite groups chỉ ở 0.85 và unanswerable ở 0.98, hai infinite groups phải có trung bình **trên 0.61** để overall vượt 0.80. Do đó tập trung hoàn toàn vào format/unanswerable không đủ. Mỗi tăng 7 điểm ở một nhóm chỉ thêm khoảng 1 điểm overall.

## 6. Kiến trúc chính: pretrained VLM có OT contextualizer bắt buộc

### 6.1 Chọn backbone bằng dữ liệu, không bằng tên model

Screening trên 2 × T4: **hai ứng viên train chính là 2B và 4B**, cùng input hợp lệ/scorer, cùng 80 figure/560 QA dev được chọn phân tầng trước khi xem predictions; ứng viên tốt chạy đủ 240 figure/1.680 QA. Không bắt buộc tải hoặc chạy 8B để bắt đầu triển khai.

| Ứng viên | Vai trò | Lý do thử | Ràng buộc |
|---|---|---|---|
| **Qwen3-VL-2B-Instruct** | Phát triển interface, kiểm tra native parity/gradient và baseline tài nguyên | Ưu tiên triển khai đầu tiên trên T4; dùng native processor và generator | Chưa có pilot CUDA; không hứa đạt 80% với 2B |
| **Qwen3-VL-4B-Instruct** | Ứng viên tăng chất lượng trên cùng họ backbone | Chuyển lên sau khi 2B hoạt động và 4B vượt memory/throughput gate | QLoRA + OT phải vừa **mỗi** GPU, giữ đủ visual evidence; không ép giảm ảnh để vừa rồi coi là so sánh ngang bằng |
| InternVL3-8B | Tham chiếu SciVQA / optional teacher hoặc thử nghiệm trên GPU lớn | Paper Infyn báo standalone test ROUGE-L 0.740 | Không nằm trên critical path T4; chỉ thêm khi có tài nguyên/pilot phù hợp |

Backbone version/revision phải pin. Không thay `--image_model` bằng tên VLM rồi dùng `AutoModel.last_hidden_state` như hiện tại: cần loader, processor, multimodal connector, tokenizer và generator riêng. Nguồn: [Qwen3-VL chính thức](https://github.com/QwenLM/Qwen3-VL), [mã finetune](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune), [InternVL3 model card](https://huggingface.co/OpenGVLab/InternVL3-8B), [nghiên cứu Infyn](https://arxiv.org/abs/2507.06183).

Baseline VLM nguyên bản chỉ là **control thực nghiệm** để biết năng lực pretrained và kiểm tra mất mát do adapter. Mô hình nghiên cứu/final tiếp tục dùng OT làm đường fusion; không chuyển default repo sang baseline không OT.

### 6.2 Interface và đường dữ liệu

```text
Native high-resolution image processor -> pretrained vision encoder -> V
Question / caption / choices -> role-aware text representation -> Q
OCR tokens + bbox + confidence -------------------------------> O (giai đoạn sau)

V, Q, optional O -> learned projections -> masked partial OT T
               -> OT-conditioned visual memory V_ot
               -> native pretrained VLM connector / language model
               -> concise answer, supervised only on answer tokens
```

Thiết kế khởi đầu giữ số visual tokens mà connector native mong đợi:

1. Lấy `V` từ đúng stage của backbone, giữ native image-grid/tile layout và pixel mask.
2. Tạo `Q` từ question và lựa chọn hợp lệ. Caption/context có role embedding và mask riêng. Ban đầu không dùng hidden state của **gold answer** để tạo plan.
3. Tính `C_ij = 1 - cos(Wv(V_i), Wq(Q_j))`, projections/cost FP32, solver log-domain FP32. Tránh áp OT toàn chiều LLM; dùng cost space 128/256 trước, values giữ native dimension.
4. Tạo aligned text tại visual positions từ `T @ Q / matched_per_patch`; tạo `V_ot = V + g * P(aligned_text)`, sau đó đưa **toàn bộ visual paths, gồm DeepStack nếu có**, qua contextualizer đúng native dimensions. Trên Qwen3-VL ưu tiên sau native merger: giữ nguyên main embeddings và các DeepStack outputs rồi áp OT conditioning; không có raw DeepStack nối vòng ngoài adapter. Chi tiết §14.2.
5. **Không zero-init đồng thời gate và projection:** khi `g=0` và `P=0`, cả hai gradient đều bằng 0. Khởi đầu parity test với gate 0 và projection random nhỏ; kiểm tra bước đầu gate có gradient, sau khi gate mở thì cost projections có gradient. Hoặc zero-init output projection nhưng gate khác 0. Không thêm normalization trên toàn residual ở trạng thái no-op vì sẽ phá native parity. Sau warmup, chứng minh đóng góp bằng feasible plan controls cùng mask/marginals (§14.3), OT-disabled và retrained native control. Gate khác 0 hoặc prediction change tự nó chưa chứng minh tăng độ chính xác.
6. Giữ residual bên trong OT module để bảo toàn chi tiết ảnh. OT phải đóng góp qua **cả plan và giá trị**; không chỉ thêm một auxiliary loss lên model vẫn dùng ảnh theo đường riêng.
7. Với Qwen dùng native grid và số merged visual tokens; với InternVL dùng native tiles và thumbnail. Lưu mapping bbox về tọa độ ảnh gốc. Bỏ giả định `sqrt(patches)` luôn nguyên; multi-tile/rectangular/OCR nodes cần explicit coordinates, valid patch mask và token role.
8. Attention mask, position IDs, image placeholder token count và train/infer chat template phải khớp native VLM. Supervision chỉ answer; không tính loss trên image/question/padding. Test một batch heterogeneous aspect ratio trước khi DDP.

Đây là **thiết kế đề xuất chưa được triển khai/đo**. Nếu adapter làm baseline pretrained giảm sâu, sửa interface/gate/staging trước, không bù bằng thêm loss hoặc tăng epoch.

### 6.3 Làm OT phù hợp với evidence của biểu đồ

Ưu tiên lần lượt, mỗi mục một ablation:

- **Role-aware marginals:** question trọng số riêng, choices riêng, caption riêng, OCR riêng. Không phân đều mass cho hàng trăm boilerplate/caption tokens ngang với token nội dung câu hỏi. Dùng masked learned salience nhưng đặt lower/upper bound và entropy monitoring để tránh dồn một token.
- **Question-conditioned relevance:** learn patch marginal từ câu hỏi; padding/background không có bằng chứng nên có thể dùng valid pixel/foreground prior. Không đồng nhất nền trắng với vô dụng: chữ/đường mảnh cũng nằm trên nền trắng.
- **Fine-grained evidence:** OCR token có bbox/confidence, panel ID và nguồn. Dùng OT giữa query tokens và visual/OCR nodes; coordinate encodings dùng tọa độ gốc, không index trong list.
- **Giữ chi tiết:** không ép toàn bộ 576+ patches thành một vector mean; giữ OT-conditioned visual memory hoặc evidence slots đủ độ phân giải.
- **Số head:** thử 1 so 4 trước; không tăng 8/16 trước khi 4 head có lợi. Theo dõi pairwise similarity của plans để xem các heads có trùng nhau không.
- **FGW/UOT:** nhánh nghiên cứu sau cùng. UOT cần kiểm tra tau dương, tổng mass/normalization và gate semantics; số residual tới fixed marginals không còn là tiêu chí hội tụ phù hợp khi relaxed. Không mặc định UOT tự loại background đúng.

## 7. Lộ trình triển khai có cổng nghiệm thu

### Giai đoạn A — 1–2 ngày làm việc: chuẩn hóa bằng chứng và sửa correctness

1. Khóa baseline: copy code snapshot/commit, hash CSV/checkpoint, environment, prompt/decode versions. Tách kết quả log đính kèm và checkpoint epoch 20.
2. Khôi phục regression tests **được lưu trong repo**: numeric tokenizer coverage, gold answer không bị numeric-route sai, compound metadata, multiple-answer choices, PAD isolation, BOS/EOS shift, causal leakage, input equivalence train/eval/predict, checkpoint round-trip và optimizer resume.
3. Tắt numeric hard mask và sửa compound prompt trong **hai experiment riêng**, giữ baseline hiện tại làm control. Cung cấp decode config để đánh giá checkpoint cũ đúng phiên bản; prompt mới dùng fresh run hoặc báo rõ inference-only input shift.
4. Sửa diagnosis BERTScore settings/count lỗi; tạo report đầy đủ theo 7 nhóm và theo compound/figure type/answer length/unseen answer.
5. Đo nguyên bản / đổi ảnh khác / ảnh trắng / bỏ caption / chỉ caption / bỏ lựa chọn trên validation được phân tầng. Counterfactual đổi ảnh giữ question+caption, nên chỉ đo incremental image contribution khi caption còn nguyên; caption-only và visual-only cần báo riêng.
6. Audit native image transform, original resolution, tỷ lệ nhãn nhỏ, OCR coverage, canonical target constraint coverage. Kiểm tra 50–100 lỗi/nhóm trọng yếu bằng hình gốc, không chỉ đọc text prediction.

**Cổng A:** có baseline cùng snapshot/environment để tái lập; chênh checkpoint stored/current được giải thích hoặc được ghi rõ là lịch sử chưa tái lập vì thiếu manifest, không dùng làm baseline đối chứng. Không còn lỗi chặn target; tests chạy thực sự. Chỉ khi đó mới chạy fresh training để đo sửa lỗi. Không nghiệm thu bằng checklist “No tests found (skipped)”.

### Giai đoạn B — 2–3 ngày: xác định giới hạn kiến trúc hiện tại với ít run

Các run phải có cùng seed, batches, số optimizer updates, transform và decoder. Sau correctness fixes, tạo baseline mới B0 và chỉ thay một yếu tố mỗi run:

| ID | Thay đổi so với baseline phù hợp | Giả thuyết | Giữ lại khi |
|---|---|---|---|
| B0 | Correctness-fixed current architecture | Mốc so sánh mới | Ghi đầy đủ 7 nhóm và counterfactual |
| B1 | `ot_loss_weight=0` so với 0.05 | Positive-only OT cost có thể không hữu ích | Dev ROUGE cao hơn và/hoặc visual numeric không giảm |
| B2 | `ot_num_heads=1` so với 4 | Multihead chưa tạo specialization có ích | Có measured quality/compute tradeoff |
| B3 | Visual memory off so với on | Detail retention có thể quan trọng hơn sharper plan | Infinite visual và overall cải thiện |
| B4 | Transformer dropout 0.10 so với 0.25, giữ output-head dropout 0.25 | Reg ở decoder layers có thể hạn chế fitting | Báo rõ output dropout đang bị clamp; nếu thay output dropout thì tách ablation khác |
| B5 | LR decoder 1e-4 so với 2.5e-5 | New layers có thể học quá chậm | Pilot hữu hạn, small-set fit tốt hơn, full-dev tăng |
| B6 | Unfreeze 1–2 vision blocks hoặc vision LoRA | Representation cần thích nghi | Chỉ sau hỗ trợ freeze/no_grad/optimizer mới; chart evidence tốt hơn |

Không chạy full factorial B1–B6. B0/B1 bắt buộc, B2/B3 ưu tiên theo counterfactual; B4/B5 chỉ khi small-set fitting chỉ ra vấn đề. Chỉ một ứng viên tối ưu hiện tại được chạy full 20–25 epoch; screening 3–5 epoch chỉ để loại lỗi/đánh giá xu hướng, không được tuyên bố xếp hạng cuối vì phương pháp có thể hội tụ khác nhau.

**Small-set fitting test:** chọn 16–32 figure (112–224 QA), tắt smoothing và **mọi** dropout trong fixture chẩn đoán, cho đủ updates, đo teacher-forced token accuracy và generated answer EM. `--drop_prob 0` hiện chưa tắt output dropout vì VQAModel clamp tối thiểu 0.25 khi tie weights; phải kiểm tra actual modules, không chỉ args. Nếu không đạt gần-memorization (mục tiêu chẩn đoán ~95% generated EM, có ngoại lệ annotation mâu thuẫn), tìm lỗi input/target/gradient/LR trước. Không dùng kết quả memorization để tuyên bố generalization.

**Cổng B:** nếu sau correctness fixes và 2–3 controlled comparisons, full-dev vẫn khoảng dưới 0.60 và nhóm infinite yếu, dừng tinh chỉnh nhỏ. Mốc 0.60 là **quy tắc phân bổ compute đề xuất**, không phải tiên đoán trần lý thuyết. Chuyển sang C, không tiếp tục 10 recipe OT không có evidence.

### Giai đoạn C — 2–4 ngày: baseline pretrained và tích hợp OT

1. Chạy khảo sát VLM theo §6.1; so cả inference FP16 và quantized nếu định dùng QLoRA. Chọn bằng dev cùng protocol, không bằng ChartQA leaderboard.
2. Implement adapter interface chung và đúng processor. Validate native control trước và sau wrapper, rồi mới bật OT. Native wrapper phải tái lập logits/generation trong sai số số học hợp lý.
3. **Đầu tiên 100–200 optimizer updates pilot** trên train với frozen vision/LLM, chỉ train OT module mới; native connector cũng giữ frozen. Sau khi gradient/memory ổn mới cho chạy tối đa 1 epoch adapter stage. LR OT `1e-4`, warmup 5%, không auxiliary cost, smoothing 0.0 là điểm khởi đầu, không phải recipe đã xác nhận.
4. Mở LoRA cho **LLM attention projections**, native connector vẫn frozen ở comparison đầu, trong 1–3 epoch. Bắt đầu rank 16/alpha 32/dropout 0.05, LR LoRA `2e-5`; OT `1e-4`; native connector nếu cần mở phải có ablation riêng và LR pretrained riêng (khởi điểm `1e-5`). Dùng cùng stage schedule/total updates cho native control: stage adapter-only được so bằng matched zero-LR/no-update control hoặc báo rõ budget; không âm thầm cho OT variant nhiều training hơn.
5. Nếu visual OCR vẫn yếu, mở vision LoRA hoặc 1–2 block cuối ở LR `1e-6`–`5e-6`, so với frozen control. Việc này phải bỏ hard `no_grad` có điều kiện và test gradient tới đúng parameters.
6. Tất cả các nhánh final đi qua OT contextualizer. So native-finetune control và OT-finetune với cùng data/updates. Không thêm external data trong comparison này.

**Cổng C:** adapter không phá pretrained baseline; candidate tăng dev và có lợi ở nhóm khó; causal plan ablation phát hiện tác động có ích của OT. Nếu OT variant chưa thắng native control, báo trung thực hai con số, tiếp tục sửa alignment/evidence thay vì chỉ đổi tên kiến trúc.

### Giai đoạn D — 3–7 ngày: OCR, vùng ảnh, chart-to-table và dữ liệu học bổ sung

Thứ tự ưu tiên theo lỗi thực tế:

1. **High-resolution + tiles:** ảnh toàn cục để giữ bố cục, 2–4 crop/panel phù hợp để đọc chữ và số. Sử dụng crop từ original image, không phóng lại bản 384 đã mất chi tiết. Crop selector không dùng gold answer; thêm panel bounding boxes dự đoán hoặc quy tắc từ nhãn vị trí câu hỏi.
2. **OCR:** cache text+bbox+confidence theo image hash; đánh giá riêng chữ trục, tick, legend, dấu âm, decimal. Cho model xem cả token OCR và native pixels, cùng OT evidence mapping. Candidate OCR engine phải pilot trên 50–100 hình, không chọn chỉ vì tên mới.
3. **Numeric copy:** pointer/copy từ OCR/caption/context, mixture với vocab; đáp án như tên thuật toán cũng có thể copy. Copy phải theo evidence được model chọn, không dò gold. Model vẫn autoregressively generate; không thay mọi infinite bằng scalar regression.
4. **Chart-to-table/digitizer:** thử với line/bar/pie phù hợp, có confidence và kiểm tra axis log/linear, min/max, unit. Nếu table sai thì làm hại reasoning. Đo table extraction và QA cuối độc lập. DePlot là ứng viên có mã/model gốc; không mặc định tốt cho tất cả scientific compound/diagram. [DePlot/MatCha](https://github.com/google-research/google-research/tree/master/deplot).
5. **Data curriculum:** thêm train-only ChartQA/PlotQA hoặc synthetic charts có dữ liệu thật + bbox. Tạo random decimals, negative/log axes, legend colors, multi-panel và phép so sánh. Mỗi câu phải có đáp án suy ra từ rendered data. Giữ một phần held-out generator styles để phát hiện shortcut template. [ChartQA](https://github.com/vis-nlp/ChartQA).
6. Bắt đầu warmup **5k–20k** chart/OCR samples, sau đó SciVQA; hoặc mixing 20–40% external và 60–80% SciVQA. So warmup/mixing với **cùng tổng updates** và một control SciVQA-repeat để tách lợi ích thêm compute khỏi thêm data. Dải tỷ lệ là proposal, không phải recipe đã xác nhận.
7. Distillation từ teacher mạnh trên **train**: lấy answer/evidence/table; giữ gold làm supervision chính. Sample mà teacher bất đồng numeric/gold phải kiểm tra/loại khỏi pseudo-target; không tự sửa nhãn dev/test. Không train vào reasoning dài mà teacher bịa; ưu tiên evidence ngắn kiểm chứng được.

**Cổng D:** đếm lỗi OCR/numeric giảm và official dev metric tăng; không tăng chỉ nhờ hậu xử lý. Nếu native chart model kém trên SciVQA thì không dùng chỉ vì nó đạt ChartQA cao. Một preprint RE-SciVQA báo R1-F1 73.9% cùng lightweight adaptation/retrieval và thử digitization; đây là bằng chứng hướng nghiên cứu, không bảo đảm hoặc chứng minh ROUGE-L >80%. [RE-SciVQA, SSRN 2026](https://ssrn.com/abstract=6302829).

### Giai đoạn E — tùy kết quả D: retrieval và quyết định đáp án

**Retrieval chia hai protocol rõ ràng:**

- **Track strict figure+caption:** chỉ OCR/figure/caption và retrieval demonstrations từ **training figures**. Không truy xuất gold QA của chính figure đánh giá; index train không có validation/test answers. Đây là track chính để giữ nghĩa câu “dựa trên dữ liệu được cung cấp”.
- **Track mở rộng source-paper context:** parse PDF từ `pdf_url`, lấy caption và đoạn nhắc đúng figure, BM25/dense retrieval+rereank lấy 2–4 đoạn ngắn; cache nguồn/trang/vị trí. Phải gắn nhãn external context và kiểm tra quy định/version benchmark. Với unanswerable, văn bản paper có thể chứa câu trả lời ngoài evidence được phép; cần answerability gate dựa **figure+caption** và đánh giá false-answer riêng. Không gộp score track này vào strict protocol mà không công bố.

**Multiple choice:** parse `answer_options` thành cấu trúc; vì có nhiều đáp án đúng, bốn lựa chọn có **15 tập con không rỗng**. Khởi đầu score teacher-forced log-prob trên 15 canonical answer strings, tính cả EOS; so sum và length normalization trên dev với calibration. Dùng cached encoding. Đây là exact search nhỏ trên decoder hiện tại, không đòi thêm arbitrary Wasserstein ranking head. Candidate scorer về sau phải conditioned vào question+evidence, có supervision phân biệt đúng/sai; không chọn option chỉ bằng Wasserstein nhỏ nhất khi không có train discriminative.

**Binary:** so candidate `Yes`/`No` có cùng input; current step-0 mask đã làm phần lớn việc này nên không kỳ vọng tự tăng từ 58 lên 88% nhờ mask. Dành công sức cho visual reasoning và negative/negation.

**Answerability:** classifier/generation học từ train với calibrated confidence dev; không dùng `qa_pair_type=unanswerable` làm oracle. Báo false abstention trên answerable và false answer trên unanswerable.

**Postprocessing:** numeric spacing, case và answer format phải prediction-only. Không tự động bỏ dấu âm, đơn vị, decimal hoặc đổi số cho gần gold. Output chuẩn chỉ khi chính model kết luận insufficient evidence. Numeric relaxed accuracy chỉ là metric phụ; official ROUGE giữ nguyên.

**Ensemble:** chỉ thử sau khi một model tốt và error complementarity có bằng chứng. Ba checkpoint/seed hoặc hai OT-backbone có logit/probability calibration; weights/route chọn từ dev. Đo oracle union chỉ để biết room, không báo oracle là final score. Không route bằng gold qa-category nếu category không có ở deployment. Đo thêm 2–3× latency.

## 8. Tài nguyên chuẩn: 2 × T4, 16 GB/GPU

**Đây là baseline hardware của toàn bộ kế hoạch.** Hai GPU chạy DDP nhân bản model; không có một GPU 32 GB. Trước launch ghi device count, tên GPU, usable VRAM, CUDA/driver, RAM và disk trống. Nếu chỉ một GPU khả dụng, dừng recipe hai process và ghi lại cấu hình khác; không mặc định tương đương. [Thông số T4](https://www.nvidia.com/en-us/data-center/tesla-t4/).

### 8.1 Recipe phát triển và điều kiện tăng kích thước

| Thành phần | Khởi điểm trên 2 × T4 | Khi nào thay đổi |
|---|---|---|
| Current SigLIP OT-VQA | Batch 8/GPU, accumulation 1, global 16 | Nếu OOM: batch 4 + accumulation 2 **sau khi triển khai accumulation**, giữ global 16 |
| Pretrained baseline/OT đầu tiên | Qwen3-VL-2B-Instruct; frozen vision + native connector | Hoàn thành native parity, finite backward và pilot trước khi thêm LoRA |
| Quantization | CUDA NF4 + double quantization cho LLM được hỗ trợ; OT không quantize | So quantized vs native FP16 cùng evidence trên screening dev; không suy quality parity từ bit width |
| Dtype | Backbone/LoRA compute FP16, gradient scaling; OT cost/solver/critical divisions FP32 trong vùng **tắt autocast** | BF16 không phải default cho T4; nếu FP16 overflow phải điều tra, không chỉ giảm loss scale vô hạn |
| VLM microbatch | 1/GPU; accumulation 8; effective batch **16** | Kiểm tra đúng group cuối và optimizer/scheduler updates; không tăng batch trước khi có headroom |
| Context | Khởi đầu tổng sequence cap 2.048, tối đa 64 answer tokens, greedy `do_sample=False` | Đo độ dài **bằng tokenizer VLM và image placeholders**; không cắt answer/choice silently; tăng cap khi dữ liệu/evidence cần và memory cho phép |
| Visual budget | Pilot một ảnh toàn cục, **256 rồi 512 merged visual tokens**/sample | Log actual `image_grid_thw` và token count; 768/1.024 hoặc thêm crop chỉ khi qua profile/quality gate |
| OT | 1 head, cost space 128; query-only content tối đa 256 **nếu audit không truncate nội dung**; epsilon 0.05, 20 iterations ban đầu | Tăng iterations khi residual yêu cầu; context roles tách riêng, không bóp mất choice/question để vừa budget |
| LoRA | LLM attention projections rank 16/alpha 32/dropout 0.05; vision và connector frozen | Connector/vision LoRA là ablation riêng, không bật tất cả cùng lúc |
| Backward | Gradient checkpointing không reentrant nếu integration hỗ trợ; `use_cache=False` khi train | Kiểm tra gradients thực sự qua frozen backbone tới OT/LoRA; không bọc toàn LLM trong `no_grad()` |
| Attention | SDPA/memory-efficient backend thực sự hỗ trợ trên T4; eager chỉ khi profile vừa | Không mặc định FlashAttention-2 có trên Turing; dense eager attention có thể quyết định OOM |
| Evaluation | Batch 1/GPU để pilot, ROUGE-only screening; BERTScore CPU ở checkpoint cần thiết | Full dev có thể chạy tuần tự hoặc shard không trùng và merge theo ID; tránh BERTScore chiếm VRAM train |

Các số cap là **cấu hình profile đề xuất**, chưa được đo trên T4. Nếu 256/512 visual tokens không đọc được labels/ticks, không ép tiếp để vừa memory rồi tuyên bố kiến trúc đủ tốt. Có thể dùng OCR offline, high-resolution vision cache nếu backbone frozen, hoặc GPU lớn hơn.

Qwen3-VL 2B/4B config hiện cho patch size 16 và spatial merge 2; với ảnh t=1, số merged tokens là `grid_h × grid_w / 4`. Pixel heuristic phải lấy từ processor/config pinned, không copy công thức `28×28` của model khác. 512 merged tokens tương ứng xấp xỉ 524.288 pixels nếu grid phù hợp; native processor mới quyết định kích thước cuối. [Config 2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json), [config 4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json).

### 8.2 Memory budget phải tính đầy đủ

Không dùng phép tính “4-bit weights chỉ 1–2 GB” để kết luận training vừa 16 GB. Ghi riêng:

`peak = quantized weights + nonquantized embeddings/vision/head + quantization metadata + trainable params/grads/optimizer + activations + OT autograd graph + logits/loss + communication/workspace`.

- Với `P_q` quantized parameters, `P_q/2` bytes chỉ là **lower bound raw NF4 storage**, không bao gồm scale/metadata hoặc module còn FP16/FP32. Không quantize OT projections và solver; chọn phạm vi quantization rõ ràng cho vision/connector.
- Vocab Qwen3-VL config là 151.936. Chỉ một tensor full-sequence logits `[1,2048,151936]` ở FP32 đã khoảng **1,16 GiB**. `labels=-100` trên prompt không tự loại tensor logits. Nếu profile cho thấy cần, chunk loss hoặc chỉ project đúng các answer-prediction positions, kiểm chứng cùng loss/gradient với full logits; không dùng `last_token_only` của inference cho training.
- Một dense attention score tensor 16 heads × 2048² ở FP16 là **128 MiB/layer**, chưa tính gradients/softmax và các layers khác. Giới hạn total tokens và backend attention quan trọng hơn chỉ đếm weights.
- Current worst-shaped OT tensor `[8×4,577,513]` FP32 khoảng **36,13 MiB**; nhiều tensors và 20 bước autograd lớn hơn nhiều. Prototype VLM với `[1,513,257]` chỉ khoảng **0,503 MiB/tensor**, nhưng đây không phải total peak của OT. Không copy custom implicit backward từ ví dụ skill; cần gradient/oracle validation riêng nếu thật sự cần kỹ thuật đó.
- Frozen vision có thể cache outputs native, **gồm main và tất cả DeepStack levels**, lưu CPU/disk và chỉ move batch cần dùng. Không cache OT-conditioned features vì còn phụ thuộc question/weights. Nếu vision/merger mở train hoặc pixel augmentation thay đổi, cache đó mất hiệu lực.

**Memory gate đề xuất:** sau warmup và optimizer-state allocation, peak `max_memory_reserved` + non-PyTorch usage quan sát qua NVML phải còn khoảng **10–15% headroom** so với usable VRAM trên **cả hai GPU**, với batch dài/ảnh lớn nhất được phép. Ghi allocated/reserved/system memory; chúng không cộng cơ học với nhau. Không đặt một số GB cố định rồi gọi là đã đo vừa. Bắt đầu bằng 100–200 optimizer updates pilot và một validation pass; giữ log OOM/overflow.

**4B chỉ được thăng cấp** khi cùng token/evidence budget như 2B, cùng effective batch 16, qua memory gate và dev quality đáng giá. Nếu OOM: dùng checkpointing/backend phù hợp, answer-only/chunked loss khi cần, LoRA scope nhỏ hơn; không coi giảm accumulation là cách giảm activation peak. Mỗi microbatch vốn chỉ có một ảnh.

### 8.3 Batch, update count và wall time

| Recipe | Microbatch/GPU | GPUs | Accumulation | Effective batch | Optimizer updates/epoch với 15.120 QA |
|---|---:|---:|---:|---:|---:|
| Current baseline | 8 | 2 | 1 | **16** | **945** |
| Current memory fallback, sau khi có accumulation | 4 | 2 | 2 | **16** | **945** |
| VLM 2B/4B pilot | 1 | 2 | 8 | **16** | **945**, khi flush đúng group cuối |

Với toàn bộ 15.120 QA, VLM mỗi rank nhận 7.560 samples; `7560 / 8 = 945` nên **mọi accumulation group đều đủ**, không có group cuối thiếu. Với subset/filter/độ dài dataset khác không chia hết, phải chuẩn hóa và flush group cuối theo số example/target thực tế rồi mới sang epoch mới. Nếu token-normalized loss, dùng tổng valid answer tokens toàn global effective batch, không trung bình các microbatch means không cùng độ dài. `global_step` và scheduler đếm **optimizer updates thành công**, không đếm 7.560 microsteps.

Profile p50/p95 step seconds, data/vision/OT/LLM/eval time và actual visual/text tokens. `wall_hours ≈ updates × seconds_per_update / 3600 + evaluation_hours`; `GPU-hours = 2 × wall_hours` khi dùng cả hai GPU. Ví dụ toán học cho 3 epoch = 2.835 updates: 10/30/60 giây mỗi update lần lượt là **7,88 / 23,63 / 47,25 giờ** chưa tính eval; **đây không phải throughput đã đo**.

Screening ban đầu giới hạn 100–200 updates smoke/profile; run học nhanh 1 epoch; chỉ top candidate train 1–3 epoch đầy đủ và repeat seeds. Current branch chỉ tối đa 2–3 controls ưu tiên, không full grid B0–B6. Các mốc ngày ở §7 mô tả công việc kỹ thuật, không hứa hoàn tất training trong quota notebook. Sau pilot ghi số giờ/quota thực còn lại và chỉ launch run vừa ngân sách.

### 8.4 Khi nào đổi GPU

2 × T4 đủ để **bắt đầu** sửa correctness, đo baseline, phát triển OT adapter và thử 2B/4B có điều kiện; chưa có bằng chứng đủ để đạt 80%. Nếu model chất lượng cần high-res/context khiến mỗi T4 OOM, hoặc pilot dự báo vượt quota dù đã dùng cấu hình đúng, đề xuất theo thứ tự:

1. **GPU 48 GB** cho 4B/8B LoRA khi cần memory lớn hơn nhưng vẫn giới hạn ngân sách.
2. **1 × A100 80 GB** cho high-res 8B, connector/vision adaptation và teacher extraction; là lựa chọn ưu tiên khi cần tiến xa về quality. [Thông số A100](https://www.nvidia.com/en-us/data-center/a100/).
3. H100/2 × A100 chỉ sau khi một pilot có trend đủ tốt và wall time cần giảm; không mặc định cần chúng để bắt đầu.

Không thuê GPU lớn chỉ để lặp lại recipe thiếu evidence. QLoRA DDP phải load đúng local device mỗi rank; `device_map=auto` không phải chiến lược DDP training. Checkpoint adapter-only và base revision pinned để tránh nhân bản hàng GB weights vào bốn checkpoint mỗi epoch. FSDP/ZeRO/CPU offload không phải default trên hai T4; cân nhắc khi có profiling và compatibility tests. [bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes), [PEFT FSDP](https://huggingface.co/docs/peft/en/accelerate/fsdp).

## 9. Ma trận ablation bắt buộc trước khi công bố

| So sánh | Câu hỏi nghiên cứu | Những gì phải giữ cố định |
|---|---|---|
| Native pretrained VLM vs cùng backbone + OT | OT có giúp sau khi đã có năng lực pretrained? | Data, processor, updates, LoRA targets, seed, decode |
| OT learned plan vs uniform/batch-shuffled plan | Kết quả có dùng alignment? | Backbone, params, evidence, memory size |
| OT loss 0 vs positive-only vs contrastive có negative hợp lệ | Supervision nào có ích? | Fusion architecture và task loss |
| Question-only vs question+caption | Lợi ích ngữ cảnh | Images, tokenizer budgets, train/eval inputs |
| Global image vs native high-res/tiles | Có phải giới hạn OCR/resolution? | Backbone, available evidence và training budget |
| Pixels vs pixels+OCR | OCR thực sự giúp đọc số/tên? | QA, processor, decoder và total updates |
| SciVQA only vs external warmup/mixing | Thêm data có ích ngoài thêm compute? | Total updates và control SciVQA-repeat |
| Frozen vision vs vision LoRA | Có cần thích nghi visual domain? | Language LoRA, evidence, updates |
| Single OT model vs ensemble | Sai số có bổ sung và đáng latency? | Full dev split và fixed ensemble recipe |

Run winners với seeds **1105, 42, 2026**. Lưu paired predictions theo `anno_id`; bootstrap 2.000 lần theo **figure**, báo CI của delta. Với repeated tuning trên 240 dev figures, có rủi ro overfit dev; giới hạn shortlist, ghi mọi experiment, chỉ final evaluate test một lần sau khi freeze thiết kế. Mức tăng <1 điểm phần trăm cần seed/CI thuyết phục hơn; không tự động kết luận significance chỉ từ một run.

## 10. File cần sửa/tạo khi triển khai

| File/module | Thay đổi dự kiến | Nghiệm thu |
|---|---|---|
| `utils/vqa_dataset.py` | Sửa subplot metadata; shared prompt/data contract; tách format metadata khỏi gold reporting | Prompt fixtures và train/eval/predict parity |
| `model/vqa_model.py` | Decode policy config; numeric mask fix; candidate sequence scoring; dispatch current/pretrained implementation | Token grammar, multi-label choices, deterministic generation |
| `model/features_extraction.py` | Freeze policy có điều kiện; explicit gradient enable; native processor adapter | Gradient vào vision khi mở, frozen parity khi tắt |
| `model/optimal_transport.py` | Valid visual masks/coordinates; role-aware cost/marginals; FGW pad fix chỉ cho research | Residual, gradients, padding/batch invariance, small oracle |
| **Mới:** `model/pretrained_ot_vqa.py` | Pretrained VLM adapter và OT contextualizer bắt buộc | Native wrapper parity, OT causal ablation |
| **Mới:** backbone adapter module | Loader/connector cho **một** backbone thắng screening trước | Processor/token count/position mapping đúng |
| `train.py`, `configs/` | AMP/accumulation, per-module LR, optional LoRA, ROUGE-only screening, resolved config manifest | SingleGPU/DDP update equivalence; resume parity |
| `utils/checkpoint.py` | Versioned architecture/input/decode/objective config; migration rõ | Old checkpoints vẫn strict-load, new resume đúng |
| `test.py`, `predict.py` | Dùng chung input contract; full breakdown/evidence source; configurable decoding | Cùng model+input cho cùng prediction |
| `tools/diagnose_scivqa_breakdown.py`, `diagnose_training.py` | Correct metric settings; real error counts; full counterfactuals/plan effects | Score khớp scorer chính thức, different-figure đúng |
| **Mới:** OCR/cache preprocessing | Image hash, bbox coordinate mapping, provenance | Cache invalidation, no label dependence |
| **Khôi phục:** `tests/` | Persistent mathematical + pipeline regressions | Chạy thực sự trong CI/local; không xóa sau pass |
| README / docs experiments | Current supported recipes và experiment table | Không chứa flag chưa triển khai như thể chạy được |

Chỉ thêm capability cần cho phase hiện tại; không xây mọi module trước khi screening cho biết backbone nào đáng đầu tư. Checkpoint cũ giữ loader/config cũ; backbone mới cần architecture identifier/version mới, không load một phần weights rồi gọi đó là resume.

## 11. Lệnh dùng được ngay và phần chưa có CLI

Các lệnh sau chỉ review/evaluate; không tự khởi chạy training tốn GPU. Đã kiểm tra `--help` của CLI liên quan. `test.py` hiện chưa có flag đổi constrained policy; dùng review script để đo cả hai, không tự bịa flag.

```bash
# Tái lập kiểm chứng tokenizer, metadata, checkpoint và OT trên train/dev.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python results/scivqa/review_20261015/audit_readonly.py

# Full validation, 3 điều kiện; cache frozen images; không tính BERTScore.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python results/scivqa/review_20261015/evaluate_cached_validation.py

# Breakdown của checkpoint bằng CLI hiện có.
python tools/diagnose_scivqa_breakdown.py \
  --checkpoint data/weights/scivqa.pt \
  --dev_csv_path data/scivqa_dataset/val.csv \
  --img_path data/scivqa_dataset/images \
  --samples -1 --device auto \
  --output_json results/scivqa/siglip_dev_breakdown_new.json

# Scorer chính, khi cần BERTScore và prediction export.
python test.py --dataset scivqa --split dev --device auto \
  --checkpoint data/weights/scivqa.pt \
  --dev_csv_path data/scivqa_dataset/val.csv \
  --img_path data/scivqa_dataset/images \
  --report_json results/scivqa/siglip_dev_report_new.json \
  --predictions_csv results/scivqa/siglip_dev_predictions_new.csv
```

Lệnh training **control current architecture** dưới đây dùng flags đã có, giữ gần cấu hình checkpoint local; nó chưa bao gồm các correctness fixes đề xuất. Chỉ dùng sau cổng A, thư mục output phải mới. Với hai run chỉ khác `--ot_loss_weight` và `--model_path`, giữ mọi thiết lập khác giống nhau.

```bash
# Hai T4 cùng runtime; batch 8/GPU -> global batch 16.
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset scivqa --device cuda --fusion ot \
  --image_model google/siglip-base-patch16-384 \
  --text_model sentence-transformers/all-MiniLM-L12-v2 \
  --epochs 25 --batch_size 8 --seed 1105 \
  --d_model 384 --ffn_hidden 1024 --num_layers 2 --num_heads 4 \
  --lr 2.5e-5 --encoder_lr 2e-6 --warmup_ratio 0.05 \
  --weight_decay 0.05 --label_smoothing 0.1 --drop_prob 0.25 \
  --loss_normalization example \
  --freeze_answer_embeddings --no-freeze_question_encoder \
  --gradient_clip 1.0 --early_stopping_patience 10 \
  --selection_metric rouge_l_f1 --train_eval_samples 512 --num_workers 2 \
  --figure_context --max_question_tokens 512 --max_answer_tokens 64 \
  --image_size 384 --image_preprocessing pad --ot_visual_memory --tie_weights \
  --ot_num_heads 4 --ot_epsilon 0.05 --ot_iterations 20 \
  --ot_dustbin_mass 0.2 --ot_dustbin_cost 1.0 \
  --fgw_alpha 0.0 --ot_loss_weight 0.0 \
  --model_path /kaggle/working/scivqa_current_no_aux_seed1105
```

Override CSV/image paths theo mount Kaggle thực tế; không đoán đường mount từ máy local. Recipe này được chốt cho **2 GPU**, global batch 16; accumulation hiện chưa có CLI. Với VLM, kế hoạch là microbatch 1/GPU × accumulation 8 × 2 GPU = 16; command VLM/LoRA/AMP/accumulation chỉ viết sau khi implement và `--help` xác nhận. Không đưa command không tồn tại vào “recipe đã hoàn tất”.

## 12. Bằng chứng nghiên cứu và cách áp dụng

- **ExpertNeurons / SciVQA 2025:** paper báo test ROUGE-L **0.7750** với InternVL3-14B finetuning, **0.7983** khi thêm RAG và **0.8043** ở cấu hình đầy đủ. Đây là cơ sở ưu tiên pretrained generation + context/data, thay vì kỳ vọng chỉ mask output làm tăng 30 điểm. Không ngoại suy số đó sang OT-VQA hoặc dataset revision khác. [Paper gốc](https://aclanthology.org/2025.sdp-1.20/).
- **Infyn:** pretrained InternVL3-8B là tham chiếu trực tiếp tốt hơn việc ghép vision encoder và decoder random; ensemble không mặc định thắng model mạnh nhất. Nguồn đã liên kết ở §6.1; không coi prompting cho VLM tương đương thêm instruction vào frozen MiniLM.
- **UNITER:** OT có thể hữu ích trong pretraining đa nhiệm có alignment objectives; điều kiện dữ liệu/negative/task phối hợp khác xa positive-only loss nhỏ hiện tại.
- **ChartQA / DePlot:** thêm supervision data-table/OCR cho đúng khả năng thiếu; cần kiểm tra transfer sang scientific diagrams, compound figures, tên phương pháp và infinite nonvisual.
- **RE-SciVQA:** cân nhắc context relevance/digitization như ablation; preprint và metric/compute khác, không bằng chứng rằng model nhỏ chart-only sẽ vượt 80%.

Các bài và tài liệu được kiểm tra trên web ngày 15/10/2026. Các cơ chế mới trong tài liệu này là **đề xuất của review**, không phải kết quả thực nghiệm đã được các nguồn đó đo cho repo này.

## 13. Điều kiện hoàn thành mục tiêu

- [ ] Reproduce baseline với hash và config đầy đủ; giải thích stored-vs-current difference.
- [x] Correctness regressions pass thật; numeric targets hợp lệ không bị chặn và panel không bị chỉ sai.
- [ ] Model final có OT trong visual fusion path; plan intervention và controlled retraining cho thấy đóng góp, hoặc báo rõ nếu chưa có lợi.
- [ ] Full dev >0.80 ROUGE-L F1 với đúng scoring và input protocol; báo 7 nhóm, exact/numeric metrics, counterfactuals, latency/memory.
- [ ] Winner được lặp seed và báo mean/std; không chọn một seed may mắn rồi coi là ổn định.
- [ ] Freeze recipe, thresholds, ensemble và sources trước final test; giữ cả score strict và external-context track nếu có.
- [ ] **Final held-out test >0.80** mới tuyên bố đạt mục tiêu test. Nếu chỉ dev vượt, báo chính xác chỉ dev vượt.
- [ ] Lưu per-example predictions, confidence intervals theo figure, checkpoint/config/source revision và toàn bộ run thất bại để tránh lặp lại kế hoạch thiếu bằng chứng.

Đề xuất hành động đầu tiên sau review: **sửa decoder numeric + subplot metadata, khôi phục tests và khóa baseline**, sau đó **xác minh Qwen3-VL-2B và memory/quality pilot Qwen3-VL-4B trên 2 × T4**. Không dành tiếp phần lớn GPU budget cho FGW/UOT trước khi giải quyết năng lực đọc/sinh câu trả lời. Các mốc thời gian theo ngày ở §7 là ước lượng công việc kỹ thuật; wall time huấn luyện phải tính lại từ pilot T4, không suy từ local MPS.

## 14. Kết quả rà soát lần hai trước triển khai — 2 × T4

### 14.1 Những phát hiện bổ sung đã kiểm tra

Lần rà này kiểm tra lại **28 Python source files** bằng AST/hash, đọc lại đường model, OT, train, config, dataset, metrics, checkpoint và các entry points. Hash của **toàn bộ 28 file khớp đợt review đầu**. Do đó không chạy lại inference 1.680 QA chỉ để tái tạo cùng số; dùng report đã lưu, kiểm tra lại liên kết/provenance và tập trung vào các rủi ro thiết kế còn thiếu. Các utility download GQA/PlantExpert/JSON không nằm trên đường SciVQA training; chúng không phải ứng viên tối ưu accuracy trong phase này. Bằng chứng probe bổ sung: [`second_review_checks.json`](../../results/scivqa/review_20261015/second_review_checks.json).

| Phát hiện | Bằng chứng / ảnh hưởng | Điều chỉnh bắt buộc trong triển khai |
|---|---|---|
| `"visual" in "non-visual"` là True | `format_qa_prompt` áp nhánh infinite numeric cả cho nonvisual; fixture “Which method achieves the maximum score?” bị thêm yêu cầu số | Dùng schema/type enum rõ ràng; tốt nhất bỏ hard numeric routing mặc định; hỗ trợ text/range/name và hai nhóm infinite |
| `--drop_prob 0` chưa tắt mọi dropout | VQAModel gọi tied head với `max(drop_prob,0.25)`; actual output dropout vẫn 0.25 | Small-set diagnostic phải kiểm actual dropout modules; phân biệt transformer dropout và output dropout trong B4 |
| Zero gate + zero output projection tạo dead branch | Tiny autograd probe cho cả gradient gate/weight/bias bằng 0 | Chỉ zero một phía; có gate-open/gradient-to-cost test, không chỉ forward hữu hạn |
| `.float()` không thắng outer autocast | Probe CPU autocast: bmm với hai input `.float()` vẫn trả BF16 | Tắt autocast quanh cost bmm và solver; đây là minh họa dtype rule, không phải CUDA parity test |
| Resume defaults vẫn mang preset mới | Parser với `--resume` chưa explicit giữ `figure_context=True`, `fgw_alpha=0`, `ot_loss_weight=0.05` | Lưu/resume đầy đủ input/decode/objective config; phân biệt absent flag và preset; chặn silent objective drift |
| Evaluation có thể áp preset context lên checkpoint cũ | `get_args(evaluation=True)` chưa reset `figure_context`; `test.py` so requested True với checkpoint False rồi báo mismatch | Absent eval flags phải inherit checkpoint; test cả old no-context và new context models, explicit conflict vẫn phải bị chặn |
| Metric diagnosis không đồng nhất | `diagnose_training._lexical_summary` dùng custom `lexical_scores.rouge_l`, khác SciVQA stemmed ROUGE | Scorer chính thức phải được dùng cho mọi delta/selection; metric phụ gắn tên khác |
| ROUGE bỏ mất một số khác biệt số học | Fixture `-12` vs `12` có ROUGE-L 1.0 vì punctuation tokenizer | Báo numeric exact/sign-aware bên cạnh ROUGE, không tối ưu score bằng cách làm sai dấu/đơn vị |
| `global_memory` và raw visual residual không phụ thuộc T hoàn toàn | Current fusion có global concat image/question và image residual | Model/adapter có mặt không đủ chứng minh OT hữu ích; thêm memory-component và plan controls |
| Checkpoint hiện lưu cả frozen backbone | `model.state_dict()` và bốn best/last payload có thể nhân bản base weights | VLM lưu OT/LoRA/trainable modules + base revision; kiểm reload prediction/optimizer state |
| `.model.to(device)` và legacy batch tensor là giả định hiện tại | Quantized VLM có local-device loader, processor dict và native labels | Cần factory/protocol mới dưới CLI hiện có; không cố gọi y nguyên loader/optimizer legacy |

Các vấn đề trên chia thành **lỗi đã tái hiện**, **hạn chế code có thể đọc trực tiếp** và **rủi ro của thiết kế chưa triển khai**. Không quy mọi vấn đề thành nguyên nhân đã đo của 45% ROUGE. Chẳng hạn AMP và dead-gate chưa tồn tại trong training hiện tại; chúng là lỗi phải tránh khi thêm capability mới.

### 14.2 Hợp đồng adapter cụ thể cho Qwen3-VL

Qwen3-VL có **main image embeddings và DeepStack embeddings** được đưa vào các lớp language model. Config 2B/4B hiện ghi ba DeepStack visual indexes 5/11/17; dimensions language là 2.048 cho 2B và 2.560 cho 4B. Chỉ sửa main embedding sẽ để các nhánh DeepStack native đi vòng quanh OT. Nguồn implementation: [Transformers Qwen3-VL](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py). Đây là code upstream được đọc khi review, **không phải phiên bản đã cài/chạy trên T4**; implementation phải pin revision và kiểm tra API thật, không dựa vào symbol mới nhất của branch main.

Prototype được chốt đủ cụ thể để code tuần tự:

1. **Input:** raw RGB figure + original question + parsed choices + optional caption; native processor tạo pixel tensor/grid/placeholder positions. Không đưa ảnh đã pad/resize 384 của legacy pipeline vào processor VLM. Giữ legacy dataset/transform khi chạy checkpoint cũ.
2. **Question representation:** tái sử dụng frozen MiniLM contextual encoder cho **question + choices**, không ghép caption/boilerplate vào mọi OT marginal ở prototype đầu. Audit mới cho question+choices không caption: train max **210 tokens**, dev max **140**; cap 128 cắt **31 train / 4 dev**, cap 256 không cắt mẫu nào. Chốt cap 256 cho revision local này; audit lại nếu prompt/data đổi. Q không bao giờ được tạo từ teacher-forced answer states. Caption vẫn xuất hiện trong native language input, và usage track được ghi rõ.
3. **Cost/plan:** lấy native **merged** main image features V ở language dimension; chiếu V và Q xuống cost dimension 128; một masked partial OT plan T/sample, ban đầu 1 head. Values Q có projection riêng sang native dimension. Khởi đầu dùng marginals hiện tại với mask hợp lệ; role-aware marginals là thí nghiệm sau, không thêm cùng lúc.
4. **Injection:** giữ native main/DeepStack shapes và positions. Với mỗi level `l`, dùng `V_l_ot = V_l + g_l * P_l(T_local @ Q_value / matched_per_patch)`. Có thể dùng chung T từ main level để tránh giải bốn Sinkhorn; mỗi level có projection/gate riêng. Kiểm tra merged token ordering/grid khớp mọi level trước khi reuse T. Nếu không khớp, chưa được bật shared-plan injection.
5. **Native interface:** wrapper phải truyền modified features theo API mà **pinned version** hỗ trợ, giữ image placeholders, MRoPE, attention mask, padding side và DeepStack injection. Không sửa thư viện global bằng monkeypatch dễ ảnh hưởng model khác. Không xóa DeepStack để tiết kiệm memory rồi gọi đó là native-equivalent.
6. **Generation:** compute V/Q/T ở prefill; preserve conditioned visual state và KV cache đúng cho decode. Không tính lại encoder/OT mỗi generated token, không để gold answers đi vào prefix; reset per-request state để batch sau không dùng RoPE/cache của batch trước.
7. **Training:** native causal LM tự shift labels theo hợp đồng của nó. Prompt/image/PAD labels là -100, target answer+EOS hợp lệ. Không dùng target pre-shifted của legacy decoder rồi shift thêm lần nữa. Giữ example-normalized answer CE nếu muốn so objective với baseline; native token-mean CE là control riêng. Native tokenizer có thể trùng PAD/EOS, nên không mask bằng phép so ID nếu điều đó xóa EOS thật; dùng positions/attention và explicit answer span.
8. **Gate/gradient:** không đặt đồng thời `g=0` và `P=0`. Frozen LLM parameters vẫn phải cho autograd đi qua activations về adapter; `no_grad()` chỉ dùng cho feature extraction thật sự frozen trước trainable OT, không bọc toàn bộ forward LLM. Với checkpointing/quantization, kiểm trainable parameters, finite nonzero grad cho OT, LoRA và gate đúng stage.
9. **Compatibility:** interface chung cho train/eval nhận prepared batch và trả task loss, logits/targets khi cần, predictions, OT diagnostics. Factory chọn legacy/pretrained path, optimizer groups theo đúng modules, checkpoint schema mới dispatch riêng. Không hardcode `.answer_embedding`, `.question_encoder.text_encoder` hoặc `.image_model.model` trong common VLM path. Cần giữ `train.py`, `test.py`, `predict.py` làm entry points, tránh pipeline song song không đồng bộ.

**Vai trò OT cần được chứng minh:** residual vẫn chứa native V, nên việc mọi nhánh đi qua một module có tên OT chưa đủ. Sau training, nếu plan controls không làm giảm quality hoặc OT không hơn native/parameter-matched control, kết luận đúng là “pretraining giúp, chưa chứng minh OT giúp”. Không ép gate lớn chỉ để tạo prediction change.

### 14.3 Đối chứng OT và đánh giá có thể bác bỏ giả thuyết

- Không dùng một tensor toàn `1/(N×L)` làm control với PAD/dustbin hoặc learned marginals; nó phá feasibility và trộn hai yếu tố. Một control giữ matched mass và local row/column sums: đặt `r=T_local.sum(-1)`, `c=T_local.sum(-2)`, `m=T_local.sum()`, rồi `T_control = r[:,None] * c[None,:] / m` cho m>0; giữ nguyên dustbin blocks, zero PAD. Đây là independent coupling với **cùng local marginals**, làm mất pairwise alignment nhưng giữ mass/gates. Thêm test residuals trước khi so predictions.
- Control này vẫn giữ relevance qua marginals của T; vì vậy so thêm **native no-OT** và **parameter-matched adapter không dùng transport** được retrain. Cùng params/updates/evidence là control cho extra capacity; inference-only can thiệp T là một câu hỏi khác về model reliance, có thể tạo distribution shift.
- Shared T across DeepStack levels chỉ là prototype tiết kiệm compute; so main-only vs all-level conditioning để xác định nhánh nào có ích. Main-only được ghi là ablation có bypass native, không gọi default OT core hoàn chỉnh.
- Các giai đoạn adapter-only/LoRA ở §7 dùng để phát triển ổn định. **Final architecture comparison** nên reset về cùng pretrained base rồi native LoRA vs OT+LoRA cùng lịch updates, warmup và data order; thêm param-matched adapter control. Nếu dùng adapter warmup riêng, báo riêng treatment/training budget và chạy control tương ứng, không quy toàn gain cho solver.
- Lưu primary full-dev overall delta và hai infinite-group deltas; correctness fixes được giữ dù một checkpoint không tăng ROUGE, nhưng không tính chúng là accuracy gain khi CI chưa hỗ trợ.
- ROUGE primary vẫn giữ đúng benchmark; numeric exact/sign-aware và false abstention là guardrails. Không sửa references, regroup test hay dùng gold unanswerability để vượt score.
- Chốt sample manifest 80 figure để screening, full 240 figure để confirm; bootstrap group theo figure, kiểm tra nhạy cảm khi group theo paper. Không xem 7 QA của cùng figure là 7 observations độc lập.

### 14.4 Accumulation, AMP, DDP và checkpoint: yêu cầu trước full run

1. **Loss weighting:** accum CE theo tổng loss từng example / tổng examples effective batch; token normalization phải chia tổng valid answer tokens toàn group/global ranks. Aux weight áp một lần đúng normalization, không bị scale sai bởi accumulation. Test batch lengths khác nhau, not only equal-size fixtures.
2. **FP16:** scaler.scale/backward mỗi microstep, DDP `no_sync()` cho các microsteps chưa cuối group; unscale đúng một lần, clip, step/update, zero-grad ở boundary. Scheduler/global step chỉ theo optimizer update thật; test overflow skip và distributed agreement. [PyTorch AMP examples](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/amp_examples.md).
3. **FP32 island:** disable autocast cho cost construction, normalization và Sinkhorn; inspect actual dtypes. Small matched masses/division cần FP32 hoặc safe casting để tránh gate/denominator underflow. Không gọi `.float()` rải rác rồi giả định bmm không bị autocast.
4. **Distributed equivalence:** so 1GPU effective batch 16 với 2GPU microbatch×accum tương đương trên deterministic fixture, dropout tắt, tolerance theo dtype. Kiểm dataset sampler duplication/last batch, uneven lengths và per-rank RNG. Full train 15.120 QA chia hết effective batch 16; subsets vẫn phải test group cuối thiếu.
5. **Validation:** chỉ rank 0 evaluate thì ranks khác đồng bộ/barrier có timeout đủ, không gọi DDP wrapper cho forward rank 0 đơn lẻ tạo collectives; hoặc shard dev không duplicate và merge theo ID. Chỉ rank 0 ghi reports/checkpoints. Inference uses same prompt/decode config and answer-only extraction.
6. **Checkpoint:** lưu base ID+immutable revision, tokenizer/processor config, quantization config, OT modules, LoRA state, optimizer, scheduler, GradScaler, best metrics, epoch/update/sampler position, world size/effective batch và RNG **từng rank**. Nếu resume không hỗ trợ giữa accumulation group thì save ở update boundary và ghi rõ.
7. **Legacy restriction:** checkpoint hiện chỉ lưu RNG từ rank 0, scheduler có thể load state với schedule function tạo từ args mới; resume sang `--epochs`/world size khác chưa bảo đảm exact continuation. Phải validate hoặc yêu cầu fresh run cho recipe mới; không gọi đó là bug accuracy đã đo khi chưa có evidence.
8. **Output safety:** fresh model_path không được silently overwrite metrics/best checkpoint cũ. Adapter checkpoint phải reload đủ cả OT và native LoRA/modules-to-save; không chỉ `save_pretrained()` phần LoRA rồi mất OT weights. Round-trip phải khớp predictions và next-update state.
9. **Disk/host RAM:** đo bytes mỗi checkpoint/cache và quota Kaggle. Native base giữ một bản pinned; hai adapters/checkpoints có thể refer cùng base. Không lưu mọi epoch đầy đủ cả base theo recipe cũ.

### 14.5 Các cổng quyết định cuối cùng

| Cổng | Điều kiện bắt buộc | Nếu không đạt |
|---|---|---|
| R0 — review | Source/config/metric/provenance rõ; baseline lịch sử chưa tái lập được đánh dấu riêng; plan 2 T4 thống nhất | Sửa tài liệu/input contract, không hứa model gain |
| R1 — correctness code | Regression tests thực sự chạy; numeric/text không bị cấm sai; subplot không bịa; checkpoint cũ load | Sửa lỗi trước training dài |
| R2 — native adapter | No-op wrapper khớp native; tất cả visual/DeepStack routes được kiểm; gate/OT gradients đúng | Sửa integration; không cộng thêm losses để che lỗi |
| R3 — tài nguyên | 100–200 optimizer updates + eval, worst allowed shapes, finite grads, không OOM, 10–15% headroom cả hai GPU | Giảm lãng phí memory có kiểm chứng, dùng 2B hoặc đề xuất GPU lớn nếu quality cần evidence lớn |
| R4 — improvement | Full-dev gain so với **matched** baseline, CI/seeds và per-group evidence | Không gọi thành công; bỏ variant hoặc điều chỉnh hypothesis |
| R5 — 80% | Strict full-dev >0.80, winner repeat seeds, frozen recipe và held-out final test >0.80 | Báo đúng điểm đang đạt; không biến mục tiêu thành kết quả dự kiến |

**Kết luận review trước code:** đã đủ cơ sở để bắt đầu **correctness fixes + tests + native 2B adapter/pilot** theo thứ tự; chưa có cơ sở để launch full 4B/8B hoặc cam kết >80%. Lộ trình tăng chất lượng được giữ, nhưng mỗi bước phải qua cổng đo tương ứng. Chưa có kiểm chứng CUDA/DDP/T4; R2/R3 là điều kiện nghiệm thu trong giai đoạn triển khai.
