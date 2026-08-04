# DANH SÁCH THÀNH VIÊN & PHÂN VAI

> Day 08 — RAG Pipeline v2 (University Services) · Đại học VinUni
> Repo nhóm: `huylq-at-work/K3-Day08-D305-A1` · Nhóm 4 người → **Phương Án A** trong [LAB_GUIDE.md](LAB_GUIDE.md) mục 2

## 1. Thành viên

| STT | Họ và tên | Mã sinh viên |
| :-: | :-- | :-- |
| 1 | Nguyễn Chí Hướng | 2A202601203 |
| 2 | Nguyễn Tiến Đạt | 2A202601387 |
| 3 | Phạm Thị Liên | 2A202601795 |
| 4 | Lê Quang Huy | 2A202601821 |

## 2. Phân công vai trò & Branch

Vai trò lấy đúng **Phương Án A (nhóm 4 người)** của `LAB_GUIDE.md`, và giữ tính liên tục với
Day 04: ai làm eval ở Day 04 thì tiếp tục làm eval, ai làm prompt thì tiếp tục cầm phần
prompt/generation.

| Thành viên | Vai trò Day 08 | Branch | File sở hữu (chỉ người này được sửa) | Vai Day 04 (để đối chiếu) |
| :-- | :-- | :-- | :-- | :-- |
| Lê Quang Huy | **Role 1 — Team Leader & RAG Architect** | `role1-architect` | `src/task9_retrieval_pipeline.py`, `src/supervisor.py`, `requirements.txt`, `.env.example`, `README.md` | UI, Deploy & Integrator |
| Nguyễn Chí Hướng | **Role 2 — Data & Retrieval Specialist** | `role2-data-retrieval` | `src/task1_collect_legal_docs.py`, `src/task2_crawl_news.py`, `src/task3_convert_markdown.py`, `src/task4_chunking_indexing.py`, `data/**` | Tool Engineer |
| Phạm Thị Liên | **Role 3 — Frontend & Chatbot Dev** | `role3-frontend-gen` | `app.py`, `src/task5_semantic_search.py`, `src/task8_pageindex_vectorless.py`, `src/task10_generation.py` | Prompt & Tool-Declaration Optimizer |
| Nguyễn Tiến Đạt | **Role 4 — Evaluation & QA Engineer** | `role4-eval-qa` | `src/task6_lexical_search.py`, `src/task7_reranking.py`, `group_project/evaluation/**` | Eval Designer & Observability |

### Ai sở hữu Task nào

| Task | Nội dung | Điểm | Người làm |
| :-: | :-- | :-: | :-- |
| 1 | Thu thập ≥3 PDF/DOCX chính sách → `data/landing/legal/` | 3 | Hướng (R2) |
| 2 | Crawl ≥5 bài viết → `data/landing/news/` | 3 | Hướng (R2) |
| 3 | Convert Markdown → `data/standardized/` | 4 | Hướng (R2) |
| 4 | Chunking (800/100) + index ChromaDB (`BAAI/bge-m3`) | 7 | Hướng (R2) |
| 5 | `semantic_search()` — dense retrieval | 6 | Liên (R3) |
| 6 | `lexical_search()` — BM25 | 6 | Đạt (R4) |
| 7 | `rerank_rrf()` / MMR / cross-encoder | 6 | Đạt (R4) |
| 8 | PageIndex vectorless | 4 | Liên (R3) |
| 9 | `retrieve()` — hybrid + fallback (cosine < 0.48) | 7 | Huy (R1) |
| 10 | `generate_with_citation()` + reorder | 4 | Liên (R3) |

> Task 2 và 4 là hai việc nặng nhất về thời gian. Nếu Hướng bị kẹt ở CP1, Huy (R1) gánh
> Task 3 để không trễ mốc CP2 — R1 không có task riêng ở CP1–CP2.

### Quy tắc sở hữu file — đọc kỹ

**Không ai sửa `tests/test_individual.py`.** Đây là bộ chấm; sửa nó là mất điểm.

**`chroma_db/` không commit** — mỗi người tự chạy `python -m src.task4_chunking_indexing` để
sinh lại. Nếu đổi bộ dữ liệu đầu vào thì phải xoá `chroma_db/` rồi index lại, nếu không kết
quả search sẽ lẫn giữa hai lần chạy.

**Task 9 phụ thuộc Task 5, 6, 7, 8.** R1 không viết `retrieve()` cho tới khi bốn hàm kia
merge xong vào `main` — viết trước sẽ phải sửa lại vì signature đổi.

**`app.py` chỉ R3 sửa.** R1 và R2 muốn đổi gì trong UI thì nói với R3, đừng sửa trực tiếp —
đây là file cả nhóm hay đụng nhất ở CP5.

## 3. Lịch theo Checkpoint (tổng 180 phút)

| CP | Thời gian | R1 — Huy | R2 — Hướng | R3 — Liên | R4 — Đạt |
| :-: | :-- | :-- | :-- | :-- | :-- |
| CP0 | 0:00–0:10 | Tạo repo nhóm, chia `.env` (`OPENROUTER_API_KEY`) | `venv` + `requirements.txt`, test import `chromadb` | `streamlit run app.py` chạy được | Cài `ragas`, `datasets` |
| CP1 | 0:10–0:35 | Duyệt nguồn dữ liệu, tránh trùng | **Task 1 + 2** | **Task 3** (hỗ trợ) | Chuẩn bị 15 câu hỏi nháp cho golden dataset |
| CP2 | 0:35–1:00 | Chốt `CHUNK_SIZE=800`, `OVERLAP=100`, `bge-m3` | **Task 4** | **Task 5** | **Task 6** |
| CP3 | 1:00–1:20 | Review công thức RRF (k=60) | Hỗ trợ R4 | **Task 8** | **Task 7** |
| CP4 | 1:20–1:45 | **Task 9** + chạy `pytest tests/ -v` | Debug retrieval | **Task 10** | Soát format citation, thử query lạc đề để test fallback |
| CP5 | 1:45–2:15 | Gộp code vào `main`, viết kiến trúc trong README | Nối `generate_with_citation()` vào `app.py` | **Chatbot UI** (memory, source, top_k) | **golden_dataset.json + RAGAS + results.md** |
| CP6 | 2:15–3:00 | Thuyết trình kiến trúc | Trả lời câu hỏi hybrid/RRF/fallback | Live demo Streamlit | Báo cáo điểm RAGAS + A/B |

## 4. Quy trình Git

**Không ai commit thẳng vào `main`** — mọi thay đổi vào `main` đi qua Pull Request, R1 review.

**Lấy branch của mình về (lần đầu):**

```bash
git clone git@github.com:huylq-at-work/K3-Day08-D305-A1.git && cd K3-Day08-D305-A1 && git checkout -b role2-data-retrieval
```

*(đổi tên branch theo bảng mục 2)*

**Trong lúc làm — lấy code mới nhất từ main:**

```bash
git pull origin main
```

**Làm xong — đẩy lên branch của mình:**

```bash
git add . && git commit -m "Role X: mo ta ngan" && git push origin HEAD
```

**Mở Pull Request:**

```bash
gh pr create --base main --head role2-data-retrieval --title "Role 2: Task 1-4 data + indexing" --body "Mo ta ngan"
```

### Thứ tự merge bắt buộc

1. **R2** merge trước (Task 1–4) — không có `data/standardized/` và `chroma_db/` thì Task 5, 6
   không chạy được, cả nhóm ngồi chờ.
2. **R3** (Task 5, 8) và **R4** (Task 6, 7) merge tiếp, hai PR này độc lập nhau.
3. **R1** `git pull origin main` rồi mới viết Task 9.
4. **R3** Task 10 + `app.py` merge sau cùng ở CP5.

### Xử lý conflict

Bảng sở hữu file ở mục 2 được thiết kế để **không có conflict**. Nếu vẫn conflict nghĩa là ai
đó sửa file không thuộc phần mình — dừng lại, hỏi trong nhóm, đừng tự resolve.

## 5. Không commit

`.env`, API key dưới mọi dạng, `.venv/`, `__pycache__/`, `chroma_db/`.
Kiểm tra bằng `git status` trước mỗi lần commit.
