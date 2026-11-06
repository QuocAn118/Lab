# Practice 3 — Get Started with Hugging Face

Bài lab gồm 2 phần:
- **Exercise 1**: Phân tích cảm xúc của một câu bằng model có sẵn trên Hugging Face Hub.
- **Exercise 2**: Fine-tune `distilbert-base-uncased` để phân loại cảm xúc nhị phân trên IMDB.

## Cấu Trúc Thư Mục

```text
Lab 3/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes               # Lưu file model bằng Git LFS
├── configs/exercise_2.yaml      # Tham số của Exercise 2
├── docs/                        # 7 file đặc tả
├── notebooks/
│   ├── Lab_3_colab.ipynb        # Notebook chạy trên Colab, chỉ gọi code trong src/
│   └── Lab_3 (1).ipynb          # Notebook cũ, giữ làm tham khảo
├── src/
│   ├── exercise_1/
│   │   └── sentiment_analysis.py
│   └── exercise_2/
│       ├── common.py            # Đọc cấu hình, ghi báo cáo, đo RAM
│       ├── data.py              # Bước 2, 4
│       ├── model.py             # Bước 3
│       ├── training.py          # Bước 5, 6
│       ├── evaluation.py        # Bước 7
│       └── demo.py
├── tests/
│   ├── test_exercise_2.py       # Test logic
│   └── test_smoke.py            # Smoke test 2 batch
├── reports/                     # Tạo khi chạy: metrics.json, figures/
└── checkpoints/                 # Tạo khi chạy; file model lưu bằng Git LFS
```

## Cách Chạy Trên Google Colab

1. Tải cả thư mục `Lab 3` lên `MyDrive` trên Google Drive. Drive cần trống khoảng 1 GB cho checkpoint.
2. Mở `notebooks/Lab_3_colab.ipynb` bằng Google Colab.
3. Chọn **Runtime → Change runtime type → T4 GPU**.
4. Chọn **Run all** và cho phép Colab truy cập Drive khi được hỏi.
5. Lấy kết quả trong `reports/metrics.json` và `reports/figures/` để điền vào `docs/evaluation.md`, `docs/demo.md`.

Nếu đặt thư mục ở nơi khác trên Drive, hãy sửa `PROJECT_DIR` trong cell đầu của notebook.

Muốn chạy từng phần, hãy đứng trong thư mục `Lab 3` và đặt `PYTHONPATH=src`:

```bash
python -m pytest tests -q -s
python -m exercise_1.sentiment_analysis --text "Hôm qua em tuyệt lắm!"
python -m exercise_2.data
python -m exercise_2.model
python -m exercise_2.training
python -m exercise_2.evaluation
python -m exercise_2.demo
```

## Tài Liệu Đặc Tả

1. **[`lab_3_description.md`](./docs/lab_3_description.md)**: Bài toán, đối chiếu đề bài, thiết kế, KPI, checkpoint, bảng nghiệm thu.
2. **[`data.md`](./docs/data.md)**: Nguồn dữ liệu, cách chia tập, tiền xử lý.
3. **[`model.md`](./docs/model.md)**: Model của từng Exercise, lớp phân loại, số tham số.
4. **[`training.md`](./docs/training.md)**: Siêu tham số, smoke test, early stopping, xử lý tình huống.
5. **[`evaluation.md`](./docs/evaluation.md)**: Báo cáo kết quả và đối chiếu KPI.
6. **[`demo.md`](./docs/demo.md)**: Lệnh chạy và kết quả thử nghiệm thực tế.
7. **[`plan_progress_status.md`](./docs/plan_progress_status.md)**: Tiến độ, nhật ký, việc dùng AI.

## Giới Hạn

- Bài dùng 1.000 mẫu train và 200 mẫu test nên kết quả có sai số khoảng ±5 điểm phần trăm.
- Nếu Colab ngắt phiên giữa lúc huấn luyện thì phải chạy lại từ đầu.
