# Nhật ký làm đồ án

Ghi ngày, việc đã làm, số liệu đọc được từ notebook. Mỗi lần ghi xong thì commit.

## Môi trường
- Máy / Colab: Laptop Windows (chạy cục bộ, CPU), VS Code + Jupyter
- Python: 3.12.0
- numpy: 2.5.4
- pandas: 3.0.6
- scikit-learn: 1.9.1
- gensim: 4.4.0
- torch: 2.15.0+cpu
---

## Tuần 1 (15–21/11)

### C1 – Phân bố nhãn và E1
- Ngày: 18/11/2026
- Phân bố nhãn train / validation (tỉ lệ): train 0,32 / 0,36 / 0,32; validation 0,3205 / 0,3590 / 0,3205
- 3 câu mẫu (row_id): ood:00196, ood:01731, ood:01446
- E1 trên validation: Accuracy = 0,9904  Macro-F1 = 0,9905  F1 neg = 1,0000   neu = 0,9865  pos = 0,9851
- Nhận xét 1 (phân bố nhãn): Ba nhãn khá cân bằng. neutral nhiều nhất (36%), negative và positive mỗi nhãn 32%. Tỉ lệ ở validation (0,3205 / 0,3590 / 0,3205) gần như trùng với train, nên có thể tập đã được chia theo tầng (stratified) và validation đại diện tốt cho train. Vì không lệch nhãn nặng, accuracy ở đây không bị "thổi phồng" bởi nhãn đa số. Mô hình chỉ đoán toàn bộ là neutral thì cũng chỉ đạt khoảng 0,36.
- Nhận xét 2 (vì sao xem cả macro-F1 lẫn F1 từng nhãn): Macro-F1 lấy trung bình F1 của 3 nhãn với trọng số bằng nhau, nên không bị nhãn nhiều mẫu lấn át như accuracy. Nhưng macro-F1 vẫn là một con số gộp và có thể che mất nhãn yếu, nên cần xem F1 từng nhãn để biết mô hình sai ở đâu. Ở E1, negative đạt 1,0000, còn neutral (0,9865) và positive (0,9851) thấp hơn một chút. Lỗi tập trung ở ranh giới neutral–positive, ví dụ câu khen nhẹ hoặc câu "khá ổn, nhưng…". Điểm gần 0,99 là rất cao, có thể dữ liệu validation khá dễ hoặc giống mẫu train, nên cần kiểm tra thêm trên dữ liệu khó hơn (OOD).

### C2 – Word2Vec
- Ngày: 18/11/2026
- Số token có vector: 1214
- Cặp token: tốt, hay 
- 3 láng giềng gần nhất của token thứ nhất: đợi (0,4772), kỳ (0,4696), khen (0,4612)
- Cosine của cặp: 0,2171
- Giới hạn khi suy ra cảm xúc từ độ gần vector: Word2Vec học vector từ ngữ cảnh xuất hiện chung, không học cực tính cảm xúc. "tốt" và "hay" cùng mang nghĩa tích cực nhưng cosine chỉ 0,2171. Láng giềng của "tốt" lại là "đợi", "kỳ", tức là các từ hay đứng gần nó trong câu chứ không phải từ đồng nghĩa; chỉ "khen" là có liên quan về cảm xúc. Kho dữ liệu nhỏ (2500 câu, 1214 token) khiến vector chưa ổn định. Ngoài ra, các từ trái nghĩa như "tốt"/"tệ" thường xuất hiện trong ngữ cảnh giống nhau nên có thể có cosine cao. Mỗi từ chỉ có một vector cố định nên mô hình không xử lý được phủ định ("không tốt") hay sắc thái theo ngữ cảnh. Vì vậy độ gần vector chỉ cho biết hai từ hay dùng trong ngữ cảnh giống nhau, không đủ để kết luận chúng cùng cảm xúc.

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
