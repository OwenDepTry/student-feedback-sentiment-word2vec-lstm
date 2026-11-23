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
- Cấu hình đọc từ notebook: vocab = 1216 (1214 token Word2Vec + `<pad>` + `<unk>`), EMBED_DIM = 100, HIDDEN_DIM = 64, 3 nhãn.
- Đầu ra Embedding: `[64, 50, 100]` (batch × số token sau khi đệm × chiều embedding). Sau đó tensor này được đóng gói bằng `pack_padded_sequence` trước khi vào LSTM.
- Trạng thái ẩn cuối của LSTM: `hidden` có kích thước `[1, 64, 64]` (số lớp × batch × hidden); lấy `hidden[-1]` được `[64, 64]`, rồi qua Dropout(0,3) vẫn giữ `[64, 64]`.
- Đầu ra lớp phân loại: `[64, 3]`, mỗi dòng là 3 logit cho negative / neutral / positive. Lô cuối của train chỉ có 2500 − 39×64 = 4 câu nên ở lô đó kích thước là `[4, 50, 100]` → `[4, 64]` → `[4, 3]`.
- Vì sao cần độ dài thật trước khi đệm: Mọi câu đều được đệm số 0 cho đủ 50 token. Nếu đưa cả chuỗi đã đệm vào LSTM, mô hình sẽ tiếp tục cập nhật trạng thái qua các bước `<pad>`. Dù vector của `<pad>` bằng 0, trạng thái vẫn bị thay đổi do bias và trọng số hồi quy, nên `hidden[-1]` sẽ là trạng thái sau một loạt token rỗng chứ không phải sau từ cuối cùng của câu. Câu càng ngắn thì càng bị "pha loãng" nhiều. Truyền độ dài thật vào `pack_padded_sequence` giúp LSTM dừng đúng ở token thật cuối cùng của từng câu, nên trạng thái ẩn cuối chỉ tóm tắt nội dung thật. Độ dài phải được tính trước khi đệm, vì sau khi đệm mọi dòng đều dài 50 và không còn biết câu thật dài bao nhiêu. Hàm `encode` dùng `max(1, len(ids))` vì `pack_padded_sequence` không nhận độ dài 0 (câu rỗng).

### E3 (Embedding ngẫu nhiên)
- Ngày chạy: 24/11/2026
- Bảng epoch / train_loss / macro-F1:

| Epoch | train_loss | macro-F1 |
|---|---|---|
| 1 | 1,024738 | 0,633250 |
| 2 | 0,742537 | 0,745258 |
| 3 | 0,268657 | 0,907856 |
| 4 | 0,104634 | 0,926460 |
| 5 | 0,055499 | 0,952409 |
| 6 | 0,040329 | 0,952434 |

- Epoch tốt nhất: 6 (macro-F1 = 0,952434). Đây cũng là epoch cuối, và macro-F1 vẫn còn tăng nhẹ so với epoch 5 (+0,000025), nên E3 có thể chưa hội tụ hẳn trong ngân sách 6 epoch. Hai epoch đầu học chậm (0,63 → 0,75), đến epoch 3 mới tăng vọt lên 0,91.

### Giả thuyết E4 (VIẾT TRƯỚC KHI CHẠY E4, commit riêng)
> ⚠️ Lưu ý trung thực: mục này được viết **sau** khi E3 và E4 đã chạy xong (commit 44d1b6e lúc 02:37 ngày 24/11/2026 đã có kết quả E4), không có commit riêng trước khi chạy E4. Phần dưới là lập luận cơ chế, không phải dự đoán đăng ký trước.
- Ngày viết: 24/11/2026 (sau khi chạy E4)
- Cơ chế (vì sao dự đoán như vậy): Word2Vec Skip-gram đã học trên chính 2500 câu train, nên các token xuất hiện trong ngữ cảnh giống nhau có vector gần nhau ngay từ đầu. Khi chép vào lớp Embedding, LSTM không phải học biểu diễn từ con số 0 mà bắt đầu từ một không gian đã có cấu trúc, nên kỳ vọng mô hình học nhanh hơn ở các epoch đầu. Tuy nhiên ở tuần 1 đã thấy vector Word2Vec không mã hóa cực tính cảm xúc tốt (cosine "tốt"–"hay" chỉ 0,2171), và cả hai lớp Embedding đều được cập nhật khi huấn luyện, nên lợi thế có thể chủ yếu nằm ở tốc độ hội tụ hơn là ở điểm cuối cùng.
- Dự đoán đo được: (1) Macro-F1 validation ở epoch 1 của E4 cao hơn rõ rệt so với E3 (0,633250). (2) Macro-F1 tốt nhất của E4 cao hơn E3 (0,952434), chênh lệch dự kiến không quá lớn. (3) neutral vẫn là nhãn có F1 thấp nhất ở cả hai mô hình.
- Kết quả nào sẽ KHÔNG ủng hộ dự đoán: macro-F1 tốt nhất của E4 ≤ 0,952434; hoặc đường macro-F1 theo epoch của E4 không cao hơn E3 ở các epoch đầu; hoặc E4 cần nhiều epoch hơn E3 để đạt cùng mức macro-F1.

### E4 (Embedding khởi tạo Word2Vec)
- Ngày chạy: 24/11/2026
- Bảng epoch / train_loss / macro-F1:

| Epoch | train_loss | macro-F1 |
|---|---|---|
| 1 | 0,930335 | 0,900381 |
| 2 | 0,282993 | 0,971708 |
| 3 | 0,096417 | 0,968410 |
| 4 | 0,051771 | 0,965328 |
| 5 | 0,031749 | 0,981024 |
| 6 | 0,041728 | 0,977835 |

- Epoch tốt nhất: 5 (macro-F1 = 0,9810239942913211). Ngay epoch 1 E4 đã đạt 0,9004, và epoch 2 (0,9717) đã vượt điểm tốt nhất của E3 sau 6 epoch. Từ epoch 3 macro-F1 dao động (giảm ở epoch 3–4, đỉnh ở epoch 5), và ở epoch 6 train_loss tăng lại (0,0317 → 0,0417) trong khi macro-F1 giảm, nên chọn epoch 5 thay vì epoch cuối là hợp lý.

### So sánh và khóa mô hình
- Macro-F1 (chưa làm tròn) E3 = 0,952434 (6 chữ số như notebook in ra)  E4 = 0,9810239942913211  → chọn **E4** (epoch 5)
- F1 từng nhãn E3 vs E4:

| | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|
| E3 (epoch 6) | 0,9519 | 0,9524 | 0,9751 | 0,9364 | 0,9458 |
| E4 (epoch 5) | 0,9808 | 0,9810 | 0,9901 | 0,9730 | 0,9800 |
| Chênh lệch | +0,0289 | +0,0286 | +0,0150 | +0,0366 | +0,0342 |

  E4 tốt hơn E3 ở cả ba nhãn. neutral tăng nhiều nhất (+0,0366) nhưng vẫn là nhãn thấp nhất ở cả hai mô hình LSTM, nên với LSTM thì neutral là nhãn khó nhất. Ở E1 (tuần 1) thì khác: nhãn thấp nhất là positive (0,9851), neutral cao hơn một chút (0,9865), dù chênh lệch rất nhỏ (0,0014).
- Đối chiếu với giả thuyết: Vì giả thuyết được ghi sau khi đã chạy E4 và thấy kết quả (nhất là dự đoán (3) về neutral), sự trùng khớp dưới đây chỉ mang tính mô tả, không phải kiểm chứng giả thuyết. Về mặt mô tả, kết quả phù hợp với cả ba ý: epoch 1 của E4 cao hơn E3 rất xa (0,9004 so với 0,6333); macro-F1 tốt nhất của E4 cao hơn E3 khoảng 0,029; neutral thấp nhất ở cả hai. Phần "chênh lệch không quá lớn" chỉ đúng một phần: khoảng cách cuối cùng nhỏ, nhưng khoảng cách về tốc độ học rất lớn. Cần thận trọng vì: (a) E3 đạt tốt nhất ở epoch cuối và vẫn đang tăng, nên với nhiều epoch hơn E3 có thể thu hẹp khoảng cách; kết luận chỉ đúng với ngân sách 6 epoch; (b) chỉ chạy một seed (42), chưa biết độ dao động giữa các seed; (c) validation chỉ có 312 câu, chênh 0,0289 accuracy chỉ tương ứng khoảng 9 câu đoán đúng thêm. Ngoài ra, E4 (0,9810) vẫn thấp hơn baseline E1 TF-IDF + LogReg trên validation (0,9905), cần xem lại ở tập test tuần 3.
- SHA-256 checkpoint (từ decision.json): `9b70a45df7a239cf495b6de8b82e493a5dff2967c225e63cfc90a251bd32be0a` (đã kiểm tra lại bằng `sha256sum 3124410066_lstm.pt`, khớp)

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