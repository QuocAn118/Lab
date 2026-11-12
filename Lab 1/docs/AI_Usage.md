# Khai báo AI — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Khai báo AI — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-10T16:24:05+07:00 ; Thanh Binh ; eb86ab8 |
| Last updated | 2026-11-11T10:16:25+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

| Ngày | Công cụ | Phần hỗ trợ | Prompt chính | Review |
| --- | --- | --- | --- | --- |
| 2026-11-10 | Codex, theo yêu cầu người dùng | prepare_data.py, data_pipeline.py, tài liệu data và kiểm thử | Triển khai 3.1 và 3.2 | Đối chiếu đặc tả và giao diện train; kiểm chứng ghi trong tiến độ. Chờ review của nhóm. |
| 2026-11-10 | Codex, theo yêu cầu người dùng | README, đặc tả gốc, tài liệu thành phần, yêu cầu 3.1–3.6, phân công, tiến độ và registry | Kiểm tra và đồng bộ tài liệu Lab 1 | Đọc code, đối chiếu giao diện/đường dẫn/checkpoint; gỡ marker conflict Lab 1, giữ lịch sử lưu vết. Test train không thu thập được do thiếu torch; còn 3 lỗi provenance cũ, chưa chạy training. Chờ nhóm review. |

| 2026-11-11 | OpenAI Codex, theo yêu cầu người dùng; người tích hợp HoangDang | Chuẩn hóa tài liệu sau đồng bộ nhánh | Đồng nhất metadata, đường dẫn, trạng thái, vòng đời Run và tiêu chí nghiệm thu | Rà tĩnh với code/config; kiểm liên kết, registry và hash artifact. Chưa chạy lại test/train vì thiếu thư viện. |

Lịch sử registry cũng ghi nhận Claude Code hỗ trợ công cụ lưu vết và train; tác giả/công cụ tạo model và model I/O chưa được xác nhận đầy đủ. Nhóm cần bổ sung khai báo phần mình, không suy tác giả code từ người commit.

## Implementation contribution (2026-11-11)

Ghi nhận nguồn bổ sung 2026-11-12: **Nguyễn Quốc An (Quoc An)** thực hiện phần training và test training Lab 1 tại `773a5a1` và `e2b6402`. `src/train.py` đã được registry ở `773a5a1` ghi `created_by: Quoc An`, `method: ai-assisted`, với Claude Code (Claude Opus 5.5). HoangDang/OpenAI Codex đối chiếu và bổ sung ghi nhận này; phần hoàn thiện pipeline và chạy thực nghiệm dưới đây là đóng góp riêng. Không quy model/data hoặc kết quả chạy của thành viên khác thành phần Quốc An thực hiện.

HoangDang integrated code with OpenAI Codex: best checkpoint loading/reload verification, model summary sidecar, test evaluation/performance, sample demo, plots, comparison and read-only manifest verifier. Existing data/model/train contributions remain in Git history. 17 tests passed before real training; no pre-existing metric is presented as a new result.

## Completion evidence (2026-11-11)

HoangDang, assisted by OpenAI Codex, executed 17 tests, real smoke and four full-data normal runs on RTX 5060 Laptop, verified artifacts, generated comparison/demo outputs and prepared the local delivery ZIP. Metrics are read from the new run files, not copied from historical logs. No Drive upload or unconfirmed student authorship is claimed.

## Repository product update

2026-11-11T10:16:25+07:00 — HoangDang, assisted by OpenAI Codex: tracked existing input and all 25 checkpoint files with Git LFS at their original paths, updated delivery documentation and kept finalized artifacts unchanged. No retraining or changed metric is claimed. The previous ZIP entry records an earlier handoff phase.
