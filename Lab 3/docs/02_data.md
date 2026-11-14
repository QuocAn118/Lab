# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Dữ Liệu Thực Nghiệm (Data Specification) — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Nguồn dữ liệu và nhãn

Exercise 2 dùng `stanfordnlp/imdb`: 25.000 đánh giá phim tiếng Anh trong split train và 25.000 trong split test. Nhãn `0 = NEGATIVE`, `1 = POSITIVE`. Không sử dụng split `unsupervised`. Exercise 1 chỉ nhận văn bản đầu vào, không dùng dataset này.

## 2. Tiền xử lý hiện hành

- Tokenizer: `distilbert/distilbert-base-uncased`, hoặc tokenizer đi kèm checkpoint khi đánh giá model đã lưu.
- `truncation=True`, `max_length=256`; giới hạn 512 vị trí của kiến trúc không phải cấu hình chạy của bài.
- Padding động bằng `DataCollatorWithPadding`; riêng CLI train trên CUDA pad đến bội 8. Notebook không đặt `pad_to_multiple_of`.
- Bản chính tokenize trực tiếp văn bản IMDB, không thêm bước xóa thẻ HTML. Bước thay `<br />` thuộc pipeline Quốc An riêng trong `docs demo/data.md`.

## 3. Cấu hình chia tập đã chốt

Seed 42; chia phân tầng train gốc theo tỷ lệ 80/20, giữ test độc lập. Validation chọn checkpoint và early stopping; test chỉ đánh giá cuối, không dùng chọn epoch hoặc chỉnh tham số.

| Luồng | Train | Validation | Test | Phạm vi |
| --- | ---: | ---: | ---: | --- |
| Notebook chính / CLI `full` | 20.000 | 5.000 | 25.000 | Đánh giá chất lượng full |
| CLI `quick` | 1.000 | 200 | 200 | Kiểm tra rút gọn, không nghiệm thu KPI chất lượng full |
| CLI `smoke` | 32 | 16 | 16 | Hai optimizer steps, kiểm tra cơ chế train/save/load |

CLI ghi dataset revision, class counts, fingerprint và SHA-256 index từng split vào `run_config.json`. Notebook ghi kích thước/fingerprint dataset trong report đánh giá; không tuyên bố notebook có toàn bộ manifest của CLI. Các tập tách biệt theo index và split gốc; không khẳng định đã loại mọi review trùng nội dung của IMDB.

## 4. Phân biệt cấu hình kế thừa

Các lựa chọn cũ như smoke 2.000/500, max length 512 hoặc padding cố định không phải cấu hình hiện hành. Bộ này lấy mục 2–3 làm chuẩn. Pipeline Quốc An dùng 1.000/200/200 theo [tài liệu riêng](docs%20demo/data.md), không thay thế full split của bản chính.

## Nguồn và phạm vi

- [01_lab_3_description.md](01_lab_3_description.md), [08_implementation.md §2, §11](08_implementation.md): phạm vi bản chính và cấu hình đã chốt.
- [10_sources.md S-03](10_sources.md): nguồn dataset; [11_AI_Usage.md](11_AI_Usage.md): quy tắc truy nguồn.
- Bản đặc tả kế thừa và lịch sử Git giữ nguồn các lựa chọn ban đầu; lần chuẩn hóa không thay dữ liệu hay thuật toán.
