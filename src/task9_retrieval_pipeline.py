"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from numbers import Real

import sys
import os
# Ensure src directory is in PYTHONPATH for script execution
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search
from task7_reranking import rerank, rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ngưỡng cosine GỐC (thang [0,1] của semantic_search), KHÔNG phải điểm RRF.
# Đo lại cho corpus của nhóm bằng: python -m src.task9_retrieval_pipeline --calibrate
SCORE_THRESHOLD = 0.48  # Ngưỡng cosine gốc đã được chốt cho corpus hiện tại
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def _future_result(name: str, future: Future) -> list[dict]:
    """
    Lấy kết quả một retriever; một nhánh chưa sẵn sàng không làm hỏng nhánh còn lại.

    Semantic và lexical là hai nguồn độc lập. Mọi lỗi runtime của một nguồn (thiếu
    model/thư viện, Chroma chưa index hoặc lỗi kết nối) được ghi log và chuyển thành
    danh sách rỗng để nguồn còn lại và PageIndex fallback vẫn có thể phục vụ query.
    """
    try:
        results = future.result()
        return results if isinstance(results, list) else []
    except Exception as exc:
        print(f"[WARN] {name} search unavailable: {exc}")
        return []


def _pageindex_fallback(query: str, top_k: int) -> list[dict]:
    """Gọi PageIndex an toàn vì Task 8 và API key đều là tùy chọn."""
    try:
        results = pageindex_search(query, top_k=top_k)
    except Exception as exc:
        print(f"[WARN] PageIndex fallback unavailable: {exc}")
        return []

    normalized = []
    for item in results or []:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        result = item.copy()
        result.setdefault("score", 0.0)
        result.setdefault("metadata", {})
        result["source"] = "pageindex"
        normalized.append(result)
    return normalized[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query không được để trống")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k phải là số nguyên không âm")
    if not isinstance(score_threshold, Real) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold phải nằm trong [0, 1]")
    if top_k == 0:
        return []

    retrieval_k = top_k * 2

    # Step 1: hai retriever độc lập được chạy đồng thời để giảm latency.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as executor:
        dense_future = executor.submit(semantic_search, query, retrieval_k)
        sparse_future = executor.submit(lexical_search, query, retrieval_k)
        dense_results = _future_result("Semantic", dense_future)
        sparse_results = _future_result("Lexical", sparse_future)

    # Step 2: RRF chỉ dùng thứ hạng để fuse; điểm của hai retriever không cùng
    # thang đo nên không cộng trực tiếp cosine với BM25.
    merged = rerank_rrf(
        [dense_results, sparse_results],
        top_k=retrieval_k,
    )
    for item in merged:
        item["source"] = "hybrid"

    # RRF đã là bước reranking. Chỉ chạy thêm reranker khi cấu hình một phương
    # pháp khác (ví dụ Jina cross-encoder), tránh ghi đè điểm fused bằng RRF lần 2.
    if use_reranking and merged and RERANK_METHOD != "rrf":
        final_results = rerank(
            query,
            merged,
            top_k=top_k,
            method=RERANK_METHOD,
        )
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Step 3/4: quyết định fallback bằng cosine GỐC của dense top-1. Tuyệt đối
    # không dùng merged[0]['score'], vì đó là RRF (~0.016 với k=60).
    best_dense_score = (
        float(dense_results[0].get("score", 0.0)) if dense_results else 0.0
    )
    if best_dense_score < score_threshold:
        print(
            f"[INFO] Dense cosine {best_dense_score:.3f} < "
            f"threshold {score_threshold:.3f}; trying PageIndex"
        )
        fallback_results = _pageindex_fallback(query, top_k)
        if fallback_results:
            return fallback_results

    return final_results[:top_k]


# =============================================================================
# Calibration helper — dùng để chọn SCORE_THRESHOLD, không thuộc phần chấm điểm
# =============================================================================

# Câu chắc chắn nằm trong corpus (dịch vụ/chính sách RMIT).
IN_DOMAIN_QUERIES = [
    "What is the tuition fee at RMIT Vietnam?",
    "How do I book a library study room?",
    "What scholarships are available for international students?",
    "Does the university provide on-campus accommodation?",
]

# Câu chắc chắn lạc đề hoặc rác — dùng để đo sàn điểm cosine.
OUT_OF_DOMAIN_QUERIES = [
    "xyzabc123nonsense",
    "How do I replace the timing belt on a Toyota Corolla?",
    "Cách nấu phở bò Nam Định",
    "asdkjh qwe zxc",
]


def calibrate_threshold() -> None:
    """
    In điểm cosine gốc của hai nhóm query để chọn SCORE_THRESHOLD nằm ở giữa.

    Chạy sau khi Task 4 đã sinh xong chroma_db/:
        python -m src.task9_retrieval_pipeline --calibrate
    """
    def _measure(queries: list[str]) -> list[float]:
        scores = []
        for q in queries:
            hits = semantic_search(q, top_k=1)
            score = float(hits[0]["score"]) if hits else 0.0
            scores.append(score)
            print(f"  [{score:.3f}] {q}")
        return scores

    print("\nIN-DOMAIN (điểm nên CAO):")
    in_scores = _measure(IN_DOMAIN_QUERIES)

    print("\nOUT-OF-DOMAIN (điểm nên THẤP):")
    out_scores = _measure(OUT_OF_DOMAIN_QUERIES)

    floor = min(in_scores)
    ceiling = max(out_scores)
    print(f"\nIn-domain thấp nhất : {floor:.3f}")
    print(f"Out-domain cao nhất  : {ceiling:.3f}")

    if floor > ceiling:
        print(f"→ Đề xuất SCORE_THRESHOLD = {(floor + ceiling) / 2:.2f}")
    else:
        print(
            "→ Hai nhóm CHỒNG LẤN, không ngưỡng nào tách sạch được. "
            "Xem lại chunking (Task 4) hoặc đổi embedding model."
        )


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        calibrate_threshold()
        sys.exit(0)

    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
