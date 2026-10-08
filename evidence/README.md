# Phân tích RAGAS — Prompt V1 và V2

Đánh giá được thực hiện trên 50 cặp câu hỏi/đáp án chuẩn với cùng knowledge base,
retriever `k=3` và bốn metric RAGAS.

| Metric | Prompt V1 | Prompt V2 | Kết quả |
|---|---:|---:|---|
| Faithfulness | 0.9565 | 0.9541 | V1 cao hơn 0.0023 |
| Answer relevancy | 0.9201 | 0.8929 | V1 cao hơn 0.0273 |
| Context recall | 1.0000 | 1.0000 | Hòa |
| Context precision | 0.9383 | 0.9417 | V2 cao hơn 0.0033 |

## Nhận xét

- Cả hai phiên bản đều đạt mục tiêu faithfulness `>= 0.8`; V1 đạt 0.9565 và V2 đạt 0.9541.
- Context recall bằng 1.0 ở cả hai phiên bản, cho thấy retriever đã lấy được đầy đủ thông tin cần thiết theo bộ QA.
- V2 có context precision nhỉnh hơn rất nhỏ, nhưng V1 có faithfulness và answer relevancy cao hơn, đặc biệt answer relevancy chênh 0.0273.

## Kết luận

Chọn **Prompt V1** làm phiên bản ưu tiên cho pipeline hiện tại vì câu trả lời bám sát context tốt hơn và phù hợp với câu hỏi hơn. Prompt V2 vẫn là lựa chọn phù hợp nếu ưu tiên trình bày có cấu trúc và context precision cao hơn một chút.

Nguồn điểm: `data/ragas_report.json`.
