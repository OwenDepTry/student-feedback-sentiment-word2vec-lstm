# Sentiment Classification of Student Feedback with Word2Vec and LSTM

> Đồ án NLP – Phân loại cảm xúc phản hồi của sinh viên bằng Word2Vec và LSTM

Classifying Vietnamese student feedback on using AI in learning into **negative / neutral / positive**.
Compares an LSTM with randomly initialized embeddings (E3) against one initialized from Skip-gram Word2Vec (E4),
and benchmarks the selected model against a TF-IDF + Logistic Regression baseline (E1).

- Học phần: Xử lý ngôn ngữ tự nhiên (841448) · Lớp DCT1243 · HK I 2026–2027
- Sinh viên: 3124410066
- Thời gian: 15-11-2026 → hạn nộp 05-12-2026 (Moodle)

## Dữ liệu
Vietnamese AI Feedback OOD 3125 (Nguyen và cộng sự), giấy phép **CC BY 4.0**.
Bản chia của lớp (seed 42, phân tầng theo nhãn, ~80/10/10):

| Tập | Số câu | negative | neutral | positive |
|---|---|---|---|---|
| train | 2.500 | 800 | 900 | 800 |
| validation | 312 | 100 | 112 | 100 |
| test (nhận ở tuần 3) | 313 | 100 | 113 | 100 |

Đặt các tệp CSV trong `data/`. Notebook tự kiểm tra SHA-256.

## Cấu trúc
```
3124410066_week1.ipynb   C1 (E1: TF-IDF + LogReg), C2 (Word2Vec Skip-gram)
3124410066_week2.ipynb   C3 (E3 random vs E4 Word2Vec), khóa checkpoint
3124410066_week3.ipynb   C4 (đánh giá test, phân tích lỗi)
3124410066_lstm.pt       (tạo ở tuần 2)
3124410066_decision.json (tạo ở tuần 2)
data/                    CSV của lớp
docs/                    Đề bài
templates/               Notebook gốc giảng viên phát (không sửa)
notes/NHAT_KY.md         Nhật ký làm bài, số liệu, giả thuyết
```

## Cách chạy lại
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Chạy lần lượt week1 → week2 → week3, mỗi notebook "Restart & Run All".
Tuần 3 cần `data/ood3125_test.csv` và hai tệp `.pt`, `.json` do tuần 2 tạo (ở thư mục gốc).

## Tiến độ
- [x] Tuần 1 – C1: hoàn thành lệnh `scores`, chạy E1
- [x] Tuần 1 – C2: tách token, huấn luyện Word2Vec, chọn cặp token
- [x] Tuần 1 – viết nhận xét C1, C2 vào báo cáo
- [x] Tuần 2 – chép vector Word2Vec vào ma trận Embedding (E4)
- [x] Tuần 2 – chạy E3, ghi giả thuyết có ngày, chạy E4, khóa checkpoint
- [ ] Tuần 3 – nhận test, đánh giá E1 + LSTM, phân tích lỗi
- [ ] Báo cáo ≥ 15 trang, đóng gói `3124410066_NLP.zip` (6 tệp, không thư mục con)
