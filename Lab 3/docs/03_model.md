# Hồ sơ model — Lab 3

## 1. Thông tin model

| Trường | Nội dung |
| --- | --- |
| Name | DistilBERT fine-tuned trên IMDB — Lab 3 |
| Created date | 2026-11-06 — ngày bản fine-tuned được ghi nhận trong báo cáo |
| Created by | HoangDang |
| Last updated | 2026-11-07 — cập nhật hồ sơ model |
| Updated by | HoangDang |
| Ngày huấn luyện bản full đang sử dụng | 2026-11-06 |
| Trạng thái | Đã huấn luyện và đánh giá; checkpoint `best` đang được sử dụng |

Tên người tạo/người cập nhật được xác nhận ngày 2026-11-07. Lần cập nhật hồ sơ này không thay trọng số hoặc chạy lại full training. Run cũ chưa có timestamp bắt đầu/kết thúc chính xác; không lấy ngày commit hoặc ngày sửa file làm thời điểm train.

## 2. Purpose — mục đích

Phân loại đánh giá phim tiếng Anh thành hai nhãn: `NEGATIVE` và `POSITIVE`. Dùng model sau khi fine-tune để dự đoán câu mới và đánh giá chất lượng trên tập test độc lập.

Exercise 1 dùng một model pretrained khác để thử phân tích cảm xúc đa ngôn ngữ. Model đó được nêu riêng ở cuối mục 3; không phải bản DistilBERT IMDB được tạo trong bài này.

## 3. Initial model — mô hình ban đầu

### 3.1. Model và nguồn

| Nội dung | Giá trị |
| --- | --- |
| Pretrained model ID | `distilbert/distilbert-base-uncased` |
| Nguồn trọng số | [DistilBERT model card trên Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased) |
| Tokenizer | WordPiece, `AutoTokenizer.from_pretrained(MODEL_NAME)` |
| Cách tạo model cho bài | `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)` |
| Phần được kế thừa | Trọng số backbone pretrained |
| Phần khởi tạo mới | Head phân loại 2 lớp, chưa được học trên IMDB ở thời điểm khởi tạo |

Model gốc học biểu diễn ngôn ngữ; nạp backbone không có nghĩa head phân loại IMDB đã được huấn luyện. Bài này fine-tune cả backbone và head.

### 3.2. Model architecture — kiến trúc

```text
Văn bản → Tokenizer → input_ids + attention_mask
       → 6 tầng Transformer encoder
       → vector token đầu tiên (768 chiều)
       → Linear 768→768 → ReLU → Dropout 0.2
       → Linear 768→2 → logits
       → Softmax khi cần xác suất dự đoán
```

| Thông số | Giá trị đối chiếu từ checkpoint |
| --- | ---: |
| Lớp model | DistilBertForSequenceClassification |
| Số tầng encoder | 6 |
| Hidden size | 768 |
| Attention heads mỗi tầng | 12 |
| Feed-forward hidden size | 3072 |
| Vocabulary size | 30.522 |
| Giới hạn vị trí của kiến trúc | 512 token |
| Độ dài sử dụng trong notebook | Tối đa 256 token |
| Số lớp đầu ra | 2 |
| Tổng tham số | 66.955.010, đếm từ `model.safetensors` hiện có |

Kiến trúc đối chiếu [config.json của best](../outputs/full/checkpoints/best/config.json) và [tài liệu DistilBERT](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/distilbert). Khi train, loss cross entropy nhận logits; không áp softmax trước loss.

### 3.3. Input / output model

| Loại | Nội dung / kích thước |
| --- | --- |
| Input từ người dùng | Một chuỗi hoặc danh sách chuỗi đánh giá phim tiếng Anh |
| `input_ids` | Tensor số nguyên `[batch_size, sequence_length]` |
| `attention_mask` | Tensor `[batch_size, sequence_length]`; phân biệt token thật và padding |
| `labels` khi train | Tensor `[batch_size]`, giá trị `0` hoặc `1` |
| Output trực tiếp | Logits `[batch_size, 2]` |
| Output khi suy luận | Nhãn và score sau softmax, nằm trong khoảng 0–1 |
| Quy ước nhãn | `0 = NEGATIVE`, `1 = POSITIVE` |

`sequence_length` tối đa 256 trong notebook; padding động theo batch. Model không có lớp `NEUTRAL`, nên một câu trung tính vẫn buộc được phân vào một trong hai lớp.

### 3.4. Source file — file mã nguồn

- [Lab_3 (1).ipynb](../../Lab_3%20%281%29.ipynb): nguồn thực hành chính; Exercise 2 Step 3 nạp model/tokenizer, Step 4 tokenize, Step 5–6 cấu hình và train, Step 7 đánh giá/demo.
- [lab3.py](../lab3.py): pipeline CLI bổ sung; `train_run` tạo model/Trainer, `load_demo` nạp model đã lưu.
- [04_training.md](04_training.md): cấu hình train và cách ghi thời gian.
- [10_sources.md](10_sources.md): nguồn API, pretrained model và dataset.

### 3.5. Model có sẵn của Exercise 1

| Nội dung | Giá trị |
| --- | --- |
| Name / model ID | `tabularisai/multilingual-sentiment-analysis` |
| Bên phát hành | `tabularisai`, theo model card |
| Kiến trúc nền | DistilBERT multilingual cased, head 5 lớp |
| Input / output | Văn bản đa ngôn ngữ → token, nhãn và score |
| Nhãn | Very Negative, Negative, Neutral, Positive, Very Positive |
| Cách dùng trong bài | Nạp qua pipeline, chỉ suy luận; không train lại |
| Source file | Ô Exercise 1 của notebook |
| Nguồn | [Model card](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) |

Thông tin HoangDang ở mục 1 áp dụng cho bản fine-tuned IMDB của dự án, không xác nhận quyền tác giả của hai pretrained model bên ngoài.

## 4. Trained model — mô hình sau huấn luyện

### 4.1. Update after training — thay đổi sau train

- Cập nhật trọng số của toàn bộ backbone và head bằng dữ liệu IMDB.
- Giữ nguyên kiến trúc 6 tầng và đầu ra 2 lớp; không thêm tầng sau huấn luyện.
- Chọn model theo validation accuracy, sau đó lưu model/tokenizer để nạp lại.
- Lưu metadata thời gian trong `training_time.json`. Việc cập nhật file tài liệu hoặc metadata không đồng nghĩa huấn luyện lại trọng số.

### 4.2. Completion description training — mô tả hoàn tất

| Nội dung | Kết quả đã ghi nhận |
| --- | --- |
| Dữ liệu | 20.000 train / 5.000 validation / 25.000 test |
| Cấu hình | Seed 42, batch 16, LR 2e-5, AdamW, weight decay 0.01, max length 256 |
| Huấn luyện | Đủ 3 epochs, 3.750 optimizer steps |
| Model được chọn | `best_val_acc`, epoch 3 |
| Validation accuracy | 91,34% |
| Test accuracy | 91,012% |
| Test macro F1 | 0,910118 |
| Tổng thời gian train | 334,46 giây, khôi phục từ báo cáo cũ |
| Nhận xét | Train loss giảm nhưng validation loss tăng ở các epoch sau; có dấu hiệu overfitting |

Nguồn kết quả: [06_verification.md](06_verification.md) và [nguồn thời gian khôi phục](../outputs/full/timing_sources.md). Bản full đã có không được train lại trong lần cập nhật hồ sơ này. Tệp thời gian cũ có `status=recovered_from_documentation`; timestamp chưa được ghi nhận vẫn để `null`. Các run mới ghi thời gian thực theo [04_training.md §6](04_training.md).

### 4.3. Saving path — đường dẫn lưu

Các đường dẫn dưới đây tính từ thư mục repo `Lab/`.

| Mục | Đường dẫn | Tình trạng |
| --- | --- | --- |
| Model full đang dùng | `Lab 3/outputs/full/checkpoints/best/` | Có config, trọng số, tokenizer và file thời gian |
| Trọng số | `Lab 3/outputs/full/checkpoints/best/model.safetensors` | Có |
| Cấu hình | `Lab 3/outputs/full/checkpoints/best/config.json` | Có |
| Tokenizer | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` trong thư mục best | Có |
| Thời gian model/run cũ | `Lab 3/outputs/full/checkpoints/best/training_time.json` | Khôi phục từ báo cáo, không có timestamp gốc |
| Best theo accuracy | `Lab 3/outputs/full/checkpoints/best_val_acc/` | Có trọng số |
| Best theo loss | `Lab 3/outputs/full/checkpoints/best_val_loss/` | Có trọng số; epoch 1 |
| `best_val_pause` | `Lab 3/outputs/full/checkpoints/best_val_pause/` | Chỉ có config; chưa phải checkpoint nạp được |
| `final_epoch` của run cũ | `Lab 3/outputs/full/checkpoints/final_epoch/` | Không có trong bản sao hiện tại |
| Notebook train mới | `Lab 3/outputs/simple-training/<run>/best/` | Đường dẫn sẽ được tạo khi `TRAIN=True` |
| Thời gian train mới | `Lab 3/outputs/simple-training/<run>/training_time.json` | Cùng run; từng checkpoint cũng có file thời gian |

Để nạp bản full, dùng `AutoModelForSequenceClassification.from_pretrained('Lab 3/outputs/full/checkpoints/best')` cùng tokenizer ở chính thư mục đó.

## Nguồn và phạm vi

- [08_implementation.md §8](08_implementation.md): các trường hồ sơ và xác nhận HoangDang ở vai trò người tạo/người cập nhật.
- [01_lab_3_description.md](01_lab_3_description.md), [02_data.md](02_data.md), [04_training.md](04_training.md), [05_evaluation.md](05_evaluation.md): mục đích, dữ liệu và phương pháp.
- [06_verification.md](06_verification.md): kết quả full và giới hạn của dữ liệu lịch sử.
- [10_sources.md S-04, S-05, S-12](10_sources.md): model card và API chính thức.
- `outputs/full/checkpoints/best/config.json`, `model.safetensors`, `training_time.json`: cấu hình, số tham số và tệp model được đối chiếu tại lần rà soát 2026-11-07.
- [11_AI_Usage.md](11_AI_Usage.md): phạm vi hỗ trợ AI và quy tắc ghi nguồn.
