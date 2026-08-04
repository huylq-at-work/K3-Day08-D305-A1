"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
from .task4_chunking_indexing import get_collection, get_embedding_model

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

    # Step 2: Embed the hypothetical document using the same model as Task 4
    model = get_embedding_model()
    query_vector = model.encode(hypothetical).tolist()

    # Step 3: Query the vector store (ChromaDB) for similar chunks
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Step 4: Build output list with cosine similarity scores
    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, 1.0 - dist)  # cosine distance -> similarity
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    # Step 5: Sort and return top_k results
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
