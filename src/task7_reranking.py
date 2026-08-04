"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
JINA_TIMEOUT_SECONDS = 30


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k phải là số nguyên không âm")


def _candidate_key(candidate: dict) -> str:
    """Lấy khóa ổn định để nhận diện cùng một chunk giữa các ranker."""
    metadata = candidate.get("metadata") or {}
    for key in ("id", "chunk_id"):
        value = candidate.get(key) or metadata.get(key)
        if value is not None:
            return f"{key}:{value}"

    source = metadata.get("source_path") or metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if source is not None and chunk_index is not None:
        return f"source:{source}:chunk:{chunk_index}"

    content = candidate.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("Mỗi candidate phải có content không rỗng hoặc chunk ID")
    return f"content:{content}"


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    _validate_top_k(top_k)
    if top_k == 0 or not candidates:
        return []

    # Không có key là trạng thái hợp lệ: dùng RRF trên thứ hạng đầu vào.
    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key:
        return rerank_rrf([candidates], top_k=top_k)

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": [candidate["content"] for candidate in candidates],
                "top_n": min(top_k, len(candidates)),
                "return_documents": False,
            },
            timeout=JINA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        api_results = response.json()["results"]

        reranked = []
        for result in api_results:
            index = int(result["index"])
            if not 0 <= index < len(candidates):
                raise ValueError(f"Jina trả index không hợp lệ: {index}")
            score = round(float(result["relevance_score"]), 6)
            item = candidates[index].copy()
            item["score"] = score
            item["rerank_score"] = score
            reranked.append(item)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]
    except (
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
    ) as exc:
        print(f"[WARN] Jina reranker unavailable; fallback to RRF: {exc}")
        return rerank_rrf([candidates], top_k=top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    _validate_top_k(top_k)
    if not isinstance(k, int) or k < 0:
        raise ValueError("k phải là số nguyên không âm")
    if top_k == 0 or not ranked_lists:
        return []

    scores: dict[str, float] = {}
    candidates_by_key: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    seen_order = 0

    for ranked_list in ranked_lists:
        # Một ranker chỉ được đóng góp một lần cho mỗi chunk, kể cả input lỗi có
        # duplicate; rank bắt đầu từ 1 theo công thức RRF chuẩn.
        seen_in_list: set[str] = set()
        for rank, candidate in enumerate(ranked_list, start=1):
            key = _candidate_key(candidate)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in candidates_by_key:
                candidates_by_key[key] = candidate
                first_seen[key] = seen_order
                seen_order += 1

    ranked_keys = sorted(
        scores,
        key=lambda key: (-scores[key], first_seen[key]),
    )

    results = []
    for key in ranked_keys[:top_k]:
        score = round(scores[key], 6)
        item = candidates_by_key[key].copy()
        item["score"] = score
        item["rrf_score"] = score
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    _validate_top_k(top_k)
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # Interface chung nhận một list; pipeline có nhiều list nên gọi trực
        # tiếp rerank_rrf([semantic_results, bm25_results]).
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
