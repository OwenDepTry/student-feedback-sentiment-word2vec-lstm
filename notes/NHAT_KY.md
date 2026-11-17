# Nhật ký làm đồ án

Ghi ngày, việc đã làm, số liệu đọc được từ notebook. Mỗi lần ghi xong thì commit.

## Môi trường
- Máy / Colab:
- Python:          numpy:      pandas:      scikit-learn:      gensim:      torch:

---

## Tuần 1 (15–21/11)

### C1 – Phân bố nhãn và E1
- Ngày:
- Phân bố nhãn train / validation (tỉ lệ):
- 3 câu mẫu (row_id):
- E1 trên validation: Accuracy = ……  Macro-F1 = ……  F1 neg = ……  neu = ……  pos = ……
- Nhận xét 1 (phân bố nhãn):
- Nhận xét 2 (vì sao xem cả macro-F1 lẫn F1 từng nhãn):

### C2 – Word2Vec
- Ngày:
- Số token có vector:
- Cặp token: ……, ……
- 3 láng giềng gần nhất của token thứ nhất:
- Cosine của cặp:
- Giới hạn khi suy ra cảm xúc từ độ gần vector:

---

## Tuần 2 (22–28/11)

### C3 – Kích thước tensor (batch = 64, MAX_TOKENS = 50)
- Đầu ra Embedding:
- Trạng thái ẩn cuối của LSTM:
- Đầu ra lớp phân loại:
- Vì sao cần độ dài thật trước khi đệm:

### E3 (Embedding ngẫu nhiên)
- Ngày chạy:
- Bảng epoch / train_loss / macro-F1:
- Epoch tốt nhất:

### Giả thuyết E4 (VIẾT TRƯỚC KHI CHẠY E4, commit riêng)
- Ngày viết:
- Cơ chế (vì sao dự đoán như vậy):
- Dự đoán đo được:
- Kết quả nào sẽ KHÔNG ủng hộ dự đoán:

### E4 (Embedding khởi tạo Word2Vec)
- Ngày chạy:
- Bảng epoch / train_loss / macro-F1:
- Epoch tốt nhất:

### So sánh và khóa mô hình
- Macro-F1 (chưa làm tròn) E3 = ……  E4 = ……  → chọn ……
- F1 từng nhãn E3 vs E4:
- Đối chiếu với giả thuyết:
- SHA-256 checkpoint (từ decision.json):

---

## Tuần 3 (29/11–05/12)

### C4 – Đánh giá test
- Ngày nhận test:
- Bảng test (E1 vs LSTM):
- Ma trận nhầm lẫn LSTM:
- Một điểm khác biệt E1 vs LSTM (có số):
- Nhãn có F1 thấp nhất của LSTM + đối chiếu ma trận:
- Macro-F1 validation → test: …… → ……; giới hạn khi diễn giải:

### Phân tích lỗi (seed = 66)
| row_id | Nhãn thật | Dự đoán | Dấu hiệu trong câu | Giả thuyết nguyên nhân |
|---|---|---|---|---|
| | negative | | | |
| | neutral | | | |
| | positive | | | |

### (Tự chọn) Cặp câu tự viết
- Câu A: ……  nhãn kỳ vọng: ……  dự đoán: ……
- Câu B: ……  nhãn kỳ vọng: ……  dự đoán: ……

### C5 – Báo cáo
- [ ] Đối chiếu mọi số liệu trong báo cáo với output notebook
- [ ] Phụ lục: môi trường, thư viện, thứ tự chạy, vị trí dữ liệu, ô notebook chứa kết quả
- [ ] Ghi nguồn + giấy phép CC BY 4.0 + cách chia dữ liệu
