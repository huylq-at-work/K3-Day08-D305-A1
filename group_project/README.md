# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 18 cặp Q&A (yêu cầu 15+)
- [x] File `group_project/evaluation/eval_pipeline.py` — DeepEval, 4 metric
- [x] File `group_project/evaluation/results.md` — bảng điểm + worst performers
- [x] So sánh A/B 4 configs: `hybrid_rerank`, `hybrid_norerank`, `dense_only`, `sparse_only`

Chạy lại đánh giá:

```bash
python -m group_project.evaluation.eval_pipeline --configs hybrid_rerank dense_only
```

Thêm `--limit 5` để chạy nhanh trên tập con khi thử nghiệm (đỡ tốn lượt gọi LLM).

> Nhóm dùng **DeepEval** chứ không phải RAGAS: `ragas==0.1.21` kéo `numpy<2`, mà bản
> numpy đó không có wheel cho Python 3.14 nên pip phải build từ nguồn và cần Visual C++
> Build Tools. Bản ragas mới hơn thì kéo `scikit-network`, cũng phải build bằng C.
> Bài lab cho phép cả ba framework.

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
                      ┌─────────────────────────────────────────┐
  NGUỒN DỮ LIỆU       │  rmit.edu.vn (trang công khai)          │
                      └───────────────┬─────────────────────────┘
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         │                                                          │
   Task 1: tải PDF chính sách                    Task 2: crawl bài viết/thông báo
   → data/landing/legal/  (3 file)               → data/landing/news/  (5 file JSON)
         │                                                          │
         └────────────────────────────┬────────────────────────────┘
                                      ▼
                      Task 3: MarkItDown → Markdown chuẩn hoá
                      • bóc nội dung khỏi HTML thô (crawler fallback lưu cả trang)
                      • bỏ dòng chỉ chứa link điều hướng
                      • khử bài trùng nội dung theo hash
                      → data/standardized/   (240KB/bài → ~8KB/bài)
                                      ▼
                      Task 4: Chunking & Indexing
                      • RecursiveCharacterTextSplitter, size=800, overlap=100
                      • embedding BAAI/bge-m3 (1024 chiều, đa ngôn ngữ)
                      → chroma_db/  (ChromaDB, cosine)
                                      │
  ════════════════════════════════════╪════════════════════════════════════
  TRUY VẤN                            ▼
                              ┌───────────────┐
                              │  Câu hỏi      │
                              └───────┬───────┘
                                      │  (chạy song song — ThreadPoolExecutor)
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        Task 5: Semantic Search              Task 6: Lexical Search
        dense, cosine trên ChromaDB          BM25 trên corpus markdown
        → score ∈ [0,1]  ◄── giữ lại         → score BM25 (thang khác)
                    │        làm căn cứ                 │
                    │        fallback                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                      Task 7: RRF Rerank   RRF(d) = Σ 1/(60 + rank)
                      • chỉ dùng THỨ HẠNG — hai thang điểm trên
                        không cộng trực tiếp được với nhau
                      • sau khi fuse, điểm chỉ còn ~0.016
                                      ▼
                      Task 9: Retrieval Pipeline
                      • nếu cosine GỐC của top-1 dense < 0.48
                        → chuyển sang Task 8
                      • KHÔNG so ngưỡng với điểm RRF (xem ghi chú dưới)
                                      │
                          ┌───────────┴───────────┐
                     đủ tốt │                     │ không đủ tốt
                            ▼                     ▼
                    source = "hybrid"     Task 8: PageIndex Vectorless
                                          truy vấn theo cấu trúc tài liệu
                                          source = "pageindex"
                            └───────────┬───────────┘
                                        ▼
                      Task 10: Generation có Citation
                      • reorder chống "lost in the middle": front + back[::-1]
                      • nhãn nguồn lấy từ metadata["source"] (tên file thật)
                      • temperature=0.3, top_p=0.9
                      • thiếu bằng chứng → "Tôi không thể xác minh..."
                                        ▼
                      app.py — Streamlit Chatbot
                      • lịch sử hội thoại, câu hỏi gợi ý
                      • bật/tắt Semantic · BM25 · Rerank  ──┐
                      • hiển thị nguồn + điểm số            │
                                                            ▼
                                          group_project/evaluation/
                                          DeepEval — 4 metric, so sánh A/B
                                          cùng 3 cờ trên → số đo khớp với UI
```

### Hai quyết định thiết kế đáng chú ý

**Ngưỡng fallback so với điểm cosine gốc, không phải điểm RRF.** Sau khi fuse, điểm
RRF chỉ phản ánh thứ hạng: top-1 luôn xấp xỉ `1/(60+1) ≈ 0.016` kể cả với câu hỏi hoàn
toàn lạc đề. Nếu lấy điểm đó so ngưỡng thì fallback không bao giờ kích hoạt. Task 9 giữ
riêng `dense_results[0]["score"]` (cosine, thang `[0,1]`) làm căn cứ quyết định.

**Tắt rerank thì bỏ luôn bước RRF.** Với `RERANK_METHOD="rrf"`, bước fuse chính là bước
rerank — nếu chỉ bỏ lần rerank thứ hai thì cờ `use_reranking` là no-op và phép so sánh
A/B cho hai kết quả giống hệt nhau.

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Lê Quang Huy | 2A202601821 | Role 1 — Team Leader & RAG Architect: Task 9 (hybrid pipeline + fallback), gộp code vào `main`, README kiến trúc | ⬜ Chưa bắt đầu |
| Nguyễn Chí Hướng | 2A202601203 | Role 2 — Data & Retrieval Specialist: Task 1–4 (thu thập, crawl, convert markdown, chunking + ChromaDB) | ⬜ Chưa bắt đầu |
| Phạm Thị Liên | 2A202601795 | Role 3 — Frontend & Chatbot Dev: Task 5, 8, 10 + Streamlit `app.py` (memory, hiển thị source) | ⬜ Chưa bắt đầu |
| Nguyễn Tiến Đạt | 2A202601387 | Role 4 — Evaluation & QA: Task 6, 7 + `golden_dataset.json`, RAGAS `eval_pipeline.py`, `results.md` | ⬜ Chưa bắt đầu |

Chi tiết vai trò, branch, quy tắc sở hữu file và lịch checkpoint: [TEAMMATES.md](../TEAMMATES.md)

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
