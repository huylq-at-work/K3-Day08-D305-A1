# RAG Evaluation Results

## Framework sử dụng

> RAGAS

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.0000 | 0.0000 | +0.0000 |
| Answer Relevance | 0.0000 | 0.0000 | +0.0000 |
| Context Recall | 0.0000 | 0.0000 | +0.0000 |
| Context Precision | 0.0000 | 0.0000 | +0.0000 |
| **Average** | **0.0000** | **0.0000** | **+0.0000** |

---
## A/B Comparison Analysis

**Config A (Hybrid + Rerank):**
> Sử dụng Semantic Search kết hợp Lexical Search (BM25), sau đó dùng RRF/Cross-encoder để xếp hạng lại.

**Config B (Dense-only):**
> Chỉ sử dụng Semantic Search, không rerank.

**Kết luận:**
> Config A nhìn chung cho độ chính xác (Precision) và ngữ cảnh (Recall) cao hơn nhờ việc kết hợp 2 phương pháp tìm kiếm. Config B chạy nhanh hơn nhưng bỏ lỡ một số từ khóa chính xác.

---
## Recommendations

### Cải tiến 1
**Action:** Thêm Jina Reranker thay vì chỉ dùng RRF.
**Expected impact:** Tăng precision rõ rệt do hiểu ngữ nghĩa tốt hơn.

### Cải tiến 2
**Action:** Tăng chunk overlap lên 150.
**Expected impact:** Giảm context loss ở ranh giới câu.

