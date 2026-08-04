# RAG Evaluation Results

Framework: **DeepEval** · Judge model: `gpt-4o-mini` · Golden dataset: **6 câu hỏi**

Sinh tự động bằng `python -m group_project.evaluation.eval_pipeline`. Đừng sửa tay — chạy lại script.

## 1. Bảng điểm theo config

| Config | Faithfulness | Answer Relevancy | Contextual Recall | Contextual Precision | Trung bình |
|---|---|---|---|---|---|
| `hybrid_rerank` | 0.750 | 0.667 | 1.000 | 0.721 | **0.784** |

## 2. So sánh A/B

Mốc so sánh: `hybrid_rerank` (trung bình 0.784)

| Config | Trung bình | Chênh so với mốc |
|---|---|---|
| `hybrid_rerank` | 0.784 | +0.000 ← mốc |

Config tốt nhất: **`hybrid_rerank`**

## 3. Worst performers

5 câu hỏi điểm thấp nhất ở config `hybrid_rerank`:

**Where is the Saigon South campus library located?** — trung bình `0.583`

- Faithfulness: `1.000`
- Answer Relevancy: `0.000`
- Contextual Recall: `1.000`
- Contextual Precision: `0.333`
- Lý do (Answer Relevancy): The score is 0.00 because the response fails to address the question about the location of the Saigon South campus library, instead mentioning an inability to verify information, which is irrelevant to the inquiry.

**What are the Saigon South campus library opening hours during semester?** — trung bình `0.750`

- Faithfulness: `0.000`
- Answer Relevancy: `1.000`
- Contextual Recall: `1.000`
- Contextual Precision: `1.000`
- Lý do (Faithfulness): The score is 0.00 because the actual output contradicts the retrieval context by stating that the Beanland library is closed on weekends, which is consistent with the context, but fails to mention its weekday hours of operation, leading to a lack of alignment.

**Is the RMIT library open during semester break weekends?** — trung bình `0.750`

- Faithfulness: `1.000`
- Answer Relevancy: `0.000`
- Contextual Recall: `1.000`
- Contextual Precision: `1.000`
- Lý do (Answer Relevancy): The score is 0.00 because the output contains multiple irrelevant statements that directly answer the question negatively, such as confirming the library's closure on weekends and providing weekday hours, which do not address the inquiry about weekend availability.

**What are the library hours during exam time?** — trung bình `0.831`

- Faithfulness: `1.000`
- Answer Relevancy: `1.000`
- Contextual Recall: `1.000`
- Contextual Precision: `0.325`
- Lý do (Contextual Precision): The score is 0.33 because while the relevant nodes (fourth and fifth) provide useful information about library locations and hours, they are ranked lower than three irrelevant nodes. The first node ranks highest but discusses unrelated topics like payment deadlines, which detracts from the overall relevance. The second and third nodes also focus on library charges, further lowering the score as they do not contribute to answering the question about library hours.

**Is the library open on public holidays?** — trung bình `0.833`

- Faithfulness: `0.500`
- Answer Relevancy: `1.000`
- Contextual Recall: `1.000`
- Contextual Precision: `0.833`
- Lý do (Faithfulness): The score is 0.50 because the actual output inaccurately asserts that both libraries are closed on public holidays, while the retrieval context only specifies this for the Beanland Library, leaving the status of the Hanoi Library unclear.

## 4. Nhận xét & đề xuất

- Điểm Contextual Recall/Precision thấp nghĩa là retriever chưa lấy đúng đoạn văn — sửa ở Task 3/4 (chất lượng corpus, chunking) chứ không phải ở prompt Task 10.
- Faithfulness cao nhưng Answer Relevancy thấp nghĩa là câu trả lời bám context nhưng lạc đề; thường do retriever đưa nhầm tài liệu.
- Câu hỏi tiếng Việt hỏi về tài liệu tiếng Anh là nhóm yếu nhất — đây là lý do nhóm chọn embedding đa ngôn ngữ `BAAI/bge-m3`.

