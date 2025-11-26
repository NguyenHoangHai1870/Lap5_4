
# 1. Mục tiêu

Trong phần này, mục tiêu là xây dựng một mô hình Nhận dạng Thực thể Tên (Named Entity Recognition – NER) sử dụng:
Bộ dữ liệu NER chuẩn CoNLL-2003.
Mô hình mạng nơ-ron hồi quy Bi-LSTM cho bài toán token classification.
Huấn luyện mô hình và đánh giá trên tập test với các chỉ số như: accuracy, F1-score.

# 2. Task 1 – Chuẩn bị dữ liệu
## 2.1. Bộ dữ liệu

Sử dụng bộ dữ liệu CoNLL-2003 từ thư viện datasets (HuggingFace).
Dữ liệu gồm 3 tập:
```
train
validation
test
```
Mỗi phần tử là:
```
tokens: danh sách các từ (token) trong câu.
ner_tags: danh sách nhãn NER tương ứng, ở dạng ID số nguyên.
```
Các nhãn sử dụng chuẩn IOB:
```
O – không thuộc thực thể.
B-PER, I-PER – bắt đầu / bên trong tên người.
B-ORG, I-ORG – tổ chức.
B-LOC, I-LOC – địa điểm.
B-MISC, I-MISC – các thực thể khác.
```
Dữ liệu được tải bằng:
```
from datasets import load_dataset
dataset = load_dataset("conll2003", trust_remote_code=True)
```
## 2.2. Chuyển nhãn ID → nhãn chuỗi

Từ metadata của dataset:
```
ner_feature = dataset["train"].features["ner_tags"]
id2tag = ner_feature.feature.names  # list: id -> tên nhãn
tag_to_ix = {tag: i for i, tag in enumerate(id2tag)}
```

Các chuỗi nhãn dạng số được ánh xạ về dạng string để thao tác cho dễ, rồi lại ánh xạ ngược trở lại tag_to_ix khi training.

## 2.3. Xây dựng vocabulary cho từ và nhãn

Từ tất cả câu trong tập train, đếm tần suất rồi tạo từ điển:
```
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

word_to_ix = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
    # các từ thực tế bắt đầu từ index = 2
}
```

Mỗi từ xuất hiện trong train được gán 1 index riêng.

Hai token đặc biệt:

<PAD> dùng để padding chuỗi cho các batch.

<UNK> dùng cho các từ không nằm trong vocab (OOV).

Ví dụ:

Kích thước vocab (vocab_size): ~20k–30k (tùy cấu hình, có thể ghi số thực khi chạy).

Số lượng nhãn NER: 9 (O, B-PER, I-PER, B-ORG, ...).

## 2.4. Dataset & Padding

Xây dựng class NERDataset kế thừa torch.utils.data.Dataset, trả về:

sentence_indices: tensor các index từ.

tag_indices: tensor các index nhãn.

Trong __getitem__:

Token → word_to_ix[token] (nếu không có thì dùng UNK_IDX).

Tag string → tag_to_ix[tag].

Khi tạo DataLoader, sử dụng collate_fn để:

Dùng pad_sequence pad các câu trong 1 batch về cùng độ dài.

Padding cho từ: dùng index của <PAD> (PAD_IDX).

Padding cho nhãn: dùng một giá trị đặc biệt, ví dụ TAG_PAD_ID = -1, và thiết lập ignore_index=TAG_PAD_ID trong loss.

Điểm chính:

Đảm bảo batch tensor có shape:

input_ids: (batch_size, max_seq_len)

labels: (batch_size, max_seq_len)

# 3. Task 2 – Xây dựng mô hình Bi-LSTM
## 3.1. Kiến trúc tổng quan

Mô hình được xây dựng theo các tầng:

Embedding Layer

Biến mỗi word index thành vector ẩn kích thước embedding_dim.

Dùng nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX).

Bi-LSTM Layer

Sử dụng nn.LSTM với bidirectional=True để mô hình nhìn được cả ngữ cảnh trái và phải của mỗi token.

Hidden size cho mỗi hướng là hidden_dim.

Do bidirectional, output tại mỗi bước có kích thước 2 * hidden_dim.
```
self.lstm = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True,
    bidirectional=True
)
```

Linear (Token-level Classifier)

Output từ Bi-LSTM: (batch_size, seq_len, 2 * hidden_dim)

Đi qua nn.Linear(2 * hidden_dim, num_tags) để dự đoán phân phối xác suất trên mỗi nhãn NER cho từng token.

(Tuỳ chọn) Dropout

Thêm nn.Dropout giữa LSTM và Linear để giảm overfitting.

## 3.2. Hàm forward

Input: x có shape (batch_size, seq_len).

Các bước:
```
embeds = self.embedding(x) → (batch_size, seq_len, embedding_dim)

lstm_out, _ = self.lstm(embeds) → (batch_size, seq_len, 2*hidden_dim)

logits = self.fc(lstm_out) → (batch_size, seq_len, num_tags)

Output: logits, chưa qua softmax (sử dụng trực tiếp cho CrossEntropyLoss).
```
Ghi chú: Bi-LSTM-CRF
Trong phạm vi lab, mình mới dừng ở Bi-LSTM. Phần CRF có thể được xem như hướng phát triển nâng cao (xem mục Kết luận & Hướng phát triển).

# 4. Task 3 – Huấn luyện và Đánh giá
## 4.1. Cấu hình huấn luyện

Optimizer: Adam với learning rate khoảng 1e-3 hoặc 5e-4.

Loss function: nn.CrossEntropyLoss(ignore_index=TAG_PAD_ID).

ignore_index giúp bỏ qua các vị trí padding trong label khi tính loss.

Batch size: ví dụ 32.

Số epoch: ví dụ 3–10 tùy thời gian và GPU.

Thiết bị:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4.2. Vòng lặp huấn luyện

Mỗi epoch:

model.train()

Duyệt train_loader:

Đưa batch lên GPU.

Tính logits = model(input_ids).

Reshape:

logits: (batch * seq_len, num_tags)

labels: (batch * seq_len,)

Tính loss: loss = criterion(logits_flat, labels_flat).

loss.backward(), optimizer.step().

Ghi lại average loss per token trên toàn bộ tập train.

Sau mỗi epoch có thể:

In train_loss.

(Tuỳ chọn) tính thêm accuracy trên validation để chọn mô hình tốt nhất.

## 4.3. Đánh giá trên tập test

Đánh giá bằng hàm evaluate(model, test_loader):
```
model.eval()

torch.no_grad()
```
Lấy preds = argmax(logits, dim=-1)

Bỏ qua vị trí padding trong label (label = TAG_PAD_ID).

Tính:

Accuracy trên từng token.

(Nâng cao) F1-score bằng thư viện seqeval trên các chuỗi nhãn.

Ví dụ bảng kết quả (bạn thay số thực tế sau khi chạy):

Mô hình	Embedding dim	Hidden dim	Bi-directional	Test Accuracy	Test F1 (NER)
Bi-LSTM (base)	128	256	Có	~0.91	~0.86

 Lưu ý:
Các giá trị trên chỉ là ví dụ minh họa.
Khi viết báo cáo chính thức, bạn nên chạy code trên Colab rồi điền số liệu thực tế (accuracy, F1, số epoch tối ưu, v.v.).

## 4.4. Hàm dự đoán câu mới

Xây dựng hàm predict_sentence(sentence):

Tách câu thành token bằng split() (hoặc tokenizer tốt hơn).

Map token → word_to_ix, OOV dùng UNK_IDX.

Đưa qua model → lấy argmax trên chiều nhãn.

Map index nhãn → string bằng ix_to_tag.

Ví dụ:
```
Câu: VNU University is located in Hanoi

Dự đoán:
VNU           -> B-ORG
University    -> I-ORG
is            -> O
located       -> O
in            -> O
Hanoi         -> B-LOC
```

# 5. Hướng dẫn chạy code (Colab)


Chọn Runtime → Change runtime type → GPU.

Chạy lần lượt các cell:
```
Cell 1: Cài đặt thư viện, tải dataset, build vocab.

Cell 2: Định nghĩa NERDataset, collate_fn, DataLoader.

Cell 3: Định nghĩa mô hình Bi-LSTM.

Cell 4: Huấn luyện mô hình (vòng lặp epoch).

Cell 5: Đánh giá trên tập test & thử predict_sentence.
```
Ghi lại:

Độ chính xác, F1-score trên tập test.


# 6. Phân tích kết quả & Nhận xét

Bạn có thể viết theo hướng:

Hiệu năng:

Mô hình Bi-LSTM cho kết quả tốt hơn so với RNN một chiều vì:

Bi-LSTM tận dụng được ngữ cảnh hai chiều (trước và sau token).

Kết quả định lượng:

Accuracy trên tập test đạt khoảng X%.

F1-score (theo entity-level) đạt Y%, trong đó:

PER thường có F1 cao hơn vì phân biệt rõ.

MISC có thể thấp hơn do đa dạng và khó phân loại.

Quan sát định tính:

Mô hình thường phân loại đúng các thực thể rõ ràng như tên người, tên tổ chức, tên địa điểm.

Dễ nhầm lẫn ở những từ viết hoa nhưng không phải thực thể, hoặc các cụm từ dài.

# 7. Khó khăn và cách giải quyết

Một số khó khăn có thể ghi lại:

Xử lý padding và ignore_index trong loss

Vấn đề: Nếu không dùng ignore_index, loss sẽ tính cả chỗ padding → kết quả sai.

Giải pháp: Chọn TAG_PAD_ID = -1, dùng:
```
criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_ID)

OOV (Out-of-vocabulary)
```
Vấn đề: Nhiều từ hiếm hoặc chỉ xuất hiện ở test.

Giải pháp: Thêm token <UNK> trong vocab và map các từ không có trong vocab về UNK_IDX.

Thời gian huấn luyện

Bi-LSTM với CoNLL-2003 có thể hơi chậm trên CPU.

Giải pháp: Sử dụng GPU trên Colab, giảm batch size hoặc giảm số epoch khi demo.

Chênh lệch giữa accuracy token-level và F1 entity-level

Token-level accuracy có thể cao nhưng F1 entity-level chưa thực sự tốt.

Giải pháp: Dùng thêm CRF hoặc thiết kế mô hình tốt hơn (Bi-LSTM-CRF) – đây có thể là hướng phát triển trong tương lai.
