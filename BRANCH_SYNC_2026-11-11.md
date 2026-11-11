# Nhật ký đồng bộ nhánh — 2026-11-11

## Thông tin lần đồng bộ

- Nhánh nhận: **hoangdang**; remote: origin (https://github.com/QuocAn118/Lab.git).
- Bắt đầu: **2026-11-11T08:14:59+07:00**.
- Hoàn tất merge và kiểm tra nội dung: **2026-11-11T08:23:49+07:00**. Mọi thời gian hiển thị theo UTC+07:00 (Asia/Bangkok).
- Người tích hợp theo cấu hình Git: **HoangDang**, email dangnh3549@ut.edu.vn; hỗ trợ **OpenAI Codex**.
- Commit trước đồng bộ: 054a9f15857045a9966207d3d4e67fabe68ca2e8.
- Nhánh dự phòng: **backup/hoangdang-before-sync-20261111-081459**.
- Commit sau ba lần merge: 06fbc7af3c62ea19a08e0c6792b22e3c495686cb. Commit ghi nhật ký được tạo sau mốc này; dùng git log để xem hash/thời gian của chính nhật ký.
- Cây làm việc sạch lúc bắt đầu. Chỉ đồng bộ các nhánh remote đã fetch; không push lên GitHub.

## Cách hiểu người sửa và thời gian

Tác giả gốc và AuthorDate được giữ nguyên. Committer và CommitDate ghi người/thời điểm ghi commit vào Git. Người tích hợp được ghi riêng trong ba merge commit, không thay tác giả code bằng tên người merge.

Đây là metadata Git, không phải thời điểm chính xác từng lần gõ/sửa file. Không dùng ngày sửa file trên Windows làm ngày huấn luyện hoặc sửa code. JSON đi kèm giữ đầy đủ hash, email tác giả/committer, thời gian và danh sách file của từng commit.

## Các nhánh đã kiểm tra

| Nhánh | Commit đầu nhánh | Tác giả đầu nhánh | AuthorDate | Commit chưa có trước đồng bộ | Kết quả |
| --- | --- | --- | --- | --- | --- |
| origin/QuocAn | 32ca8b1 | Quoc An | 2026-11-10T21:18:59+07:00 | 17 | Đã có trong hoangdang |
| origin/ThanhBinh | eb86ab8 | Thanh Binh | 2026-11-10T16:24:05+07:00 | 0 | Đã có trong hoangdang |
| origin/main | 502a84b | Quoc An | 2026-11-10T09:42:43+07:00 | 0 | Đã có trong hoangdang |
| origin/vananh | b323b78 | Vân Anh | 2026-11-10T00:14:24+07:00 | 4 | Đã có trong hoangdang |
| origin/xbao | b02aa81 | hbao2123-debug | 2026-11-10T20:09:14+07:00 | 5 | Đã có trong hoangdang |

origin/hoangdang khớp commit ban đầu. main và ThanhBinh không có commit mới cần merge. Ba nhánh còn lại bổ sung 26 commit vào lịch sử; một số là bản sao hoặc đã có nội dung tương đương, nên số commit không đồng nghĩa số thay đổi chức năng mới.

## Các lần tích hợp

| Merge commit | Người tích hợp | CommitDate | Nội dung |
| --- | --- | --- | --- |
| e1616ad | HoangDang | 2026-11-11T08:15:32+07:00 | merge: integrate origin/vananh and preserve reviewed Lab 1 changes |
| 890b888 | HoangDang | 2026-11-11T08:16:32+07:00 | merge: integrate origin/xbao model, tests and reports for Lab 2 |
| 06fbc7a | HoangDang | 2026-11-11T08:19:26+07:00 | merge: integrate origin/QuocAn and preserve current Lab 1 and Lab 3 work |

## Xử lý conflict và giữ dữ liệu

1. **vananh:** model/model I/O đã được lấy về từ trước. Giữ header nguồn Spec, tài liệu đã rà và khả năng lưu snapshot code data của bản hiện hành.
2. **xbao:** nhận Lab 2/model.py, test.py, docs/model_vi.md, báo cáo model và ignore cache. Giữ docs/model.md làm đặc tả giao diện chung; lưu đầy đủ tài liệu triển khai tiếng Anh của Bảo ở docs/model_implementation.md. Ghi rõ đường dẫn model dùng chung chưa được ghép.
3. **QuocAn — Lab 1:** giữ nguyên toàn bộ cây Lab 1 trước đồng bộ; các bản nhập cũ hơn phần gom data và cập nhật tài liệu.
4. **QuocAn — Lab 2:** nhận src/lab_2/evaluation.py và khai báo nguồn AI/kiểm thử do tác giả ghi trong tiến độ. Giữ model/test vừa nhận từ xbao.
5. **QuocAn — Lab 3:** nhận source/config/tests, notebook bổ sung, checkpoint LFS, reports/ và bộ tài liệu không đánh số. Giữ notebook gốc, pipeline hiện hành, tài liệu đánh số và bằng chứng outputs/, report/. Báo cáo nhập vào không phải kết quả vừa chạy.
6. Giữ requirements.txt đã khóa phiên bản; danh sách thư viện của QuocAn ở requirements_quocan.txt. Kết hợp LFS với quy tắc giữ nguyên byte log/report. Giữ README gốc và ghi rõ hai bản Lab 3.
7. Git LFS tự tạo bốn hook post-checkout, post-commit, post-merge, pre-push trong .githooks/; giữ pre-commit có sẵn. AGENTS.md tại gốc được nhận từ QuocAn.
8. Không sửa thuật toán thành viên, không sửa tay artifact thực nghiệm, không reset/rebase hoặc đổi lịch sử.

## Kiểm tra và giới hạn

- Tất cả đầu nhánh remote đều là ancestor của hoangdang sau merge.
- Không còn file unmerged hoặc marker conflict trong các file text thay đổi.
- Parse thành công **12 file Python**, **14 file JSON**, **2 notebook** thay đổi; đây là kiểm tra cú pháp/cấu trúc, không phải kiểm thử runtime.
- Git xác nhận **Lab 1**, **notebook gốc**, **Lab 3/outputs/** và **Lab 3/report/** không thay đổi so với nhánh dự phòng.
- Git LFS tải đủ **4 tệp trọng số** (2 đối tượng duy nhất, khoảng 536 MB tải xuống; khoảng 1,07 GB cho 4 tệp làm việc). SHA-256 và kích thước từng tệp khớp pointer; git lfs fsck: **OK**.
- Provenance Lab 1: **0 lỗi, 18 cảnh báo** về hash nguồn docs/lab_1_description.md thay đổi qua f560d1d và 054a9f1. Đây là cảnh báo có trước lần merge; không chạy --ack để giả định đã nghiệm thu lại.
- Chưa chạy test/train: Python hiện thiếu torch, torchvision, transformers, datasets.
- Lab 2 chưa ghép toàn pipeline: evaluation.py cần lab_2.common, data, model, training; các module package chưa đủ. Model của Bảo vẫn ở Lab 2/model.py.
- Hai bản Lab 3 được giữ riêng; chưa xác nhận chạy chung môi trường. Nội dung tài liệu/báo cáo nhập từ nhánh chưa được coi là bằng chứng runtime mới.

## Nhánh local và backup giữ nguyên

Nhánh local ThanhBinh có commit chưa push **40acced**, do HoangDang tạo lúc **2026-11-07T07:33:28+07:00**, chứa snapshot output/checkpoint cũ (177 file). Không gộp snapshot local này trong lần đồng bộ remote; toàn bộ vẫn còn trên nhánh local.

backup/hoangdang-before-push-20261107-091915 còn 5 commit riêng; giữ nguyên vì là backup. Nhánh backup mới giữ toàn bộ trạng thái trước lần đồng bộ. Metadata các commit local nằm trong JSON.

## Commit gốc chưa có trước lần đồng bộ

### origin/QuocAn

| Commit | Tác giả gốc | AuthorDate | Committer | CommitDate | Nội dung |
| --- | --- | --- | --- | --- | --- |
| d339891 | Quoc An | 2026-11-06T16:47:12+07:00 | Quoc An | 2026-11-06T16:47:12+07:00 | delete unnesscesary |
| 186582e | Quoc An | 2026-11-06T20:53:12+07:00 | Quoc An | 2026-11-06T20:53:12+07:00 | Add Exercise 2: Implement sentiment analysis, data handling, training, and evaluation scripts |
| 7fea49d | Quoc An | 2026-11-06T21:43:04+07:00 | Quoc An | 2026-11-06T21:43:04+07:00 | Them checkpoint Lab 3 bang Git LFS |
| 3a86e93 | Quoc An | 2026-11-07T10:30:02+07:00 | Quoc An | 2026-11-07T10:30:02+07:00 | Lab 2: hoan thien dac ta theo de bai, them cau hinh va test |
| 00b8d04 | Quoc An | 2026-11-07T11:49:25+07:00 | Quoc An | 2026-11-07T11:49:25+07:00 | Lab 2: xoa code, dac ta chi tiet tung ham cho 5 thanh vien |
| fa27469 | Quoc An | 2026-11-07T16:34:50+07:00 | Quoc An | 2026-11-07T16:34:50+07:00 | try to retore demo data.md cause accidentally delete |
| 0f71b4f | Quoc An | 2026-11-07T16:36:28+07:00 | Quoc An | 2026-11-07T16:36:28+07:00 | Lab 2: khoi phuc cac muc da mat trong model.md |
| 9096223 | Quoc An | 2026-11-07T16:36:33+07:00 | Quoc An | 2026-11-07T16:36:33+07:00 | Lab 2: khoi phuc cac muc da mat trong training.md |
| f9f2fd8 | Quoc An | 2026-11-07T16:36:38+07:00 | Quoc An | 2026-11-07T16:36:38+07:00 | Lab 2: khoi phuc cac muc da mat trong evaluation.md |
| d0cd42f | Quoc An | 2026-11-07T16:36:43+07:00 | Quoc An | 2026-11-07T16:36:43+07:00 | Lab 2: khoi phuc cac muc da mat trong demo.md |
| 7876e0c | Quoc An | 2026-11-07T16:44:25+07:00 | Quoc An | 2026-11-07T16:44:25+07:00 | Lab 2: doi quy trinh day code sang origin/main |
| 33bf735 | Quoc An | 2026-11-09T08:35:35+07:00 | Quoc An | 2026-11-09T08:35:35+07:00 | Lab 1: lay thu muc Lab 1 tu nhanh hoangdang (phan cong nhom, dac ta requirements) |
| fc5aa39 | Quoc An | 2026-11-09T08:52:00+07:00 | Quoc An | 2026-11-09T08:52:00+07:00 | Lab 1: lay model.md tu nhanh vananh |
| 48f73bf | Quoc An | 2026-11-10T09:03:50+07:00 | Quoc An | 2026-11-10T09:03:50+07:00 | Lab 1: lay model.py, model_io.py tu nhanh vananh |
| 773a5a1 | Quoc An | 2026-11-10T09:11:52+07:00 | Quoc An | 2026-11-10T09:11:52+07:00 | Lab 1: them src/train.py va tests/test_train.py (TV3 training) |
| e2b6402 | Quoc An | 2026-11-10T09:42:35+07:00 | Quoc An | 2026-11-10T09:42:35+07:00 | Lab 1: train.py theo dung dac ta (bo train_summary.json va cac khoa config thua) |
| 32ca8b1 | Quoc An | 2026-11-10T21:18:59+07:00 | Quoc An | 2026-11-10T21:18:59+07:00 | Lab 2: Evaluation.py and test 6 passed |
### origin/vananh

| Commit | Tác giả gốc | AuthorDate | Committer | CommitDate | Nội dung |
| --- | --- | --- | --- | --- | --- |
| d55bff6 | Vân Anh | 2026-11-08T23:32:29+07:00 | Vân Anh | 2026-11-08T23:32:29+07:00 | Update model.md |
| c77ae34 | Vân Anh | 2026-11-09T09:07:45+07:00 | Vân Anh | 2026-11-09T09:07:45+07:00 | Update the file scope |
| 908d0f3 | Vân Anh | 2026-11-10T00:09:26+07:00 | Vân Anh | 2026-11-10T00:09:26+07:00 | Update progress on model.md file |
| b323b78 | Vân Anh | 2026-11-10T00:14:24+07:00 | Vân Anh | 2026-11-10T00:14:24+07:00 | Add Lab 1 source code and output |
### origin/xbao

| Commit | Tác giả gốc | AuthorDate | Committer | CommitDate | Nội dung |
| --- | --- | --- | --- | --- | --- |
| cfef080 | hbao2123-debug | 2026-11-10T17:54:13+07:00 | hbao2123-debug | 2026-11-10T17:54:13+07:00 | feat(lab2): implement ResNet18 model |
| 60bc6c9 | hbao2123-debug | 2026-11-10T17:54:27+07:00 | hbao2123-debug | 2026-11-10T17:54:27+07:00 | docs(lab2): document model architecture |
| 749c935 | hbao2123-debug | 2026-11-10T17:54:38+07:00 | hbao2123-debug | 2026-11-10T17:54:38+07:00 | docs(lab2): add Vietnamese model documentation |
| 9c9db62 | hbao2123-debug | 2026-11-10T20:06:11+07:00 | hbao2123-debug | 2026-11-10T20:06:11+07:00 | test(lab-2): pass model implementation and unit tests |
| b02aa81 | hbao2123-debug | 2026-11-10T20:09:14+07:00 | hbao2123-debug | 2026-11-10T20:09:14+07:00 | chore: ignore Python cache files |

## Nguồn thay đổi theo file

Bảng đối chiếu blob sau merge với đầu nhánh nguồn. JSON giữ đầy đủ các nguồn khi nội dung trùng nhau. File xử lý thủ công ghi người/thời gian của commit tích hợp. Hook LFS và hai file nhật ký được ghi ở commit kết thúc lần đồng bộ này.

| File | Nguồn | Người sửa/tác giả | AuthorDate | Commit |
| --- | --- | --- | --- | --- |
| .gitignore | origin/xbao | hbao2123-debug | 2026-11-10T20:09:14+07:00 | b02aa81 |
| AGENTS.md | Xử lý tích hợp | HoangDang | 2026-11-11T08:19:26+07:00 | 06fbc7a |
| Lab 2/docs/model.md | Xử lý tích hợp | HoangDang | 2026-11-11T08:16:32+07:00 | 890b888 |
| Lab 2/docs/model_implementation.md | origin/xbao | hbao2123-debug | 2026-11-10T20:06:11+07:00 | 9c9db62 |
| Lab 2/docs/model_vi.md | origin/xbao | hbao2123-debug | 2026-11-10T20:06:11+07:00 | 9c9db62 |
| Lab 2/docs/plan_progress_status.md | origin/QuocAn | Quoc An | 2026-11-10T21:18:59+07:00 | 32ca8b1 |
| Lab 2/model.py | origin/xbao | hbao2123-debug | 2026-11-10T20:06:11+07:00 | 9c9db62 |
| Lab 2/reports/metrics.json | origin/xbao | hbao2123-debug | 2026-11-10T20:06:11+07:00 | 9c9db62 |
| Lab 2/src/lab_2/evaluation.py | origin/QuocAn | Quoc An | 2026-11-10T21:18:59+07:00 | 32ca8b1 |
| Lab 2/test.py | origin/xbao | hbao2123-debug | 2026-11-10T20:06:11+07:00 | 9c9db62 |
| Lab 3/.gitattributes | Xử lý tích hợp | HoangDang | 2026-11-11T08:19:26+07:00 | 06fbc7a |
| Lab 3/.gitignore | Xử lý tích hợp | HoangDang | 2026-11-11T08:19:26+07:00 | 06fbc7a |
| Lab 3/README.md | Xử lý tích hợp | HoangDang | 2026-11-11T08:19:26+07:00 | 06fbc7a |
| Lab 3/checkpoints/best/config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best/model.safetensors | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best/tokenizer.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best/tokenizer_config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_acc/config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_acc/model.safetensors | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_acc/tokenizer.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_acc/tokenizer_config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_loss/config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_loss/model.safetensors | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_loss/tokenizer.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/best_val_loss/tokenizer_config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/final_epoch/config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/final_epoch/model.safetensors | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/final_epoch/tokenizer.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/checkpoints/final_epoch/tokenizer_config.json | origin/QuocAn | Quoc An | 2026-11-06T21:43:04+07:00 | 7fea49d |
| Lab 3/configs/exercise_2.yaml | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/docs/11_AI_Usage.md | Xử lý tích hợp | HoangDang | 2026-11-11T08:19:26+07:00 | 06fbc7a |
| Lab 3/docs/12_traceability.md | Xử lý tích hợp | HoangDang | 2026-11-11T08:19:26+07:00 | 06fbc7a |
| Lab 3/docs/data.md | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/docs/demo.md | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/docs/evaluation.md | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/docs/lab_3_description.md | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/docs/model.md | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/docs/plan_progress_status.md | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/docs/training.md | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/notebooks/Lab_3 (1).ipynb | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/notebooks/Lab_3_colab.ipynb | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/reports/figures/confusion_matrix.png | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/reports/figures/loss_curve.png | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/reports/metrics.json | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/requirements_quocan.txt | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/src/exercise_1/sentiment_analysis.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/src/exercise_2/common.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/src/exercise_2/data.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/src/exercise_2/demo.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/src/exercise_2/evaluation.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/src/exercise_2/model.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/src/exercise_2/training.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/tests/test_exercise_2.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |
| Lab 3/tests/test_smoke.py | origin/QuocAn | Quoc An | 2026-11-06T20:53:12+07:00 | 186582e |

## Nguồn và phạm vi

- [BRANCH_SYNC_2026-11-11.json](BRANCH_SYNC_2026-11-11.json): snapshot đầu nhánh, metadata commit, danh sách file, quyết định xử lý, kiểm tra hash.
- Lịch sử Git các commit/nhánh nêu trên; yêu cầu người dùng ngày 2026-11-11 về kiểm tra, đưa thay đổi nhánh về nhánh cá nhân và ghi người/thời gian.
- Nhật ký chỉ xác nhận việc đồng bộ; không xác nhận các tuyên bố kiểm thử/chỉ số thực nghiệm của tác giả khi chưa chạy lại.
