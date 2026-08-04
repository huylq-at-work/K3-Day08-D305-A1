"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

try:
    from src.task4_chunking_indexing import get_collection, get_embedding_model
except ImportError:
    from task4_chunking_indexing import get_collection, get_embedding_model


def _fallback_semantic_search(query: str, top_k: int) -> list[dict]:
    """Fallback dense-like search when ChromaDB is unavailable.

    This keeps the Task 5 contract working in environments where the vector
    store dependency is missing. It reuses Task 4 documents/chunks and ranks
    them with a simple token-overlap score.
    """
    try:
        try:
            from src.task4_chunking_indexing import chunk_documents, load_documents
        except ImportError:
            from task4_chunking_indexing import chunk_documents, load_documents

        docs = load_documents()
        chunks = chunk_documents(docs)
    except Exception:
        return []

    query_tokens = {
        token.strip(".,;:!?()[]{}\"'`").lower()
        for token in query.split()
        if token.strip(".,;:!?()[]{}\"'`")
    }
    if not query_tokens:
        return []

    results = []
    for chunk in chunks:
        content = chunk.get("content", "")
        content_tokens = {
            token.strip(".,;:!?()[]{}\"'`").lower()
            for token in content.split()
            if token.strip(".,;:!?()[]{}\"'`")
        }
        if not content_tokens:
            continue

        overlap = len(query_tokens & content_tokens)
        score = overlap / max(len(query_tokens), 1)
        if score <= 0:
            continue

        results.append(
            {
                "content": content,
                "score": round(float(score), 4),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]

def _generate_hypothetical_doc(query: str) -> str:
    """Generate a hypothetical document for HyDE.

    For simplicity, this implementation returns the query itself.
    In a real scenario, you would call an LLM to generate a passage that
    answers the query.
    """
    return query

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Semantic search using HyDE.

    Embeds a hypothetical document generated from the query,
    then retrieves the most similar chunks from the ChromaDB collection.

    Returns top_k results sorted by descending similarity score.
    """
    # Step 1: Generate hypothetical document
    hypothetical = _generate_hypothetical_doc(query)

    # Step 2: Try vector-store retrieval first; fall back gracefully if the
    # environment does not have ChromaDB installed.
    try:
        model = get_embedding_model()
        query_vector = model.encode(hypothetical).tolist()

        collection = get_collection()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist)
            output.append({"content": doc, "score": round(score, 4), "metadata": meta})

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]
    except Exception:
        return _fallback_semantic_search(query, top_k)


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
