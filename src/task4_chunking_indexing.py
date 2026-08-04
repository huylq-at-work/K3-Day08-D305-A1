"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import hashlib
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# RecursiveCharacterTextSplitter giữ đoạn/văn/câu nguyên vẹn khi có thể và vẫn
# bảo đảm giới hạn kích thước khi Markdown không có cấu trúc heading đồng đều.
# 800 ký tự đủ context cho chính sách song ngữ; overlap 100 (12.5%) giúp hạn chế
# mất ý ở ranh giới chunk mà không tạo quá nhiều dữ liệu trùng lặp.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# BGE-M3 tạo vector 1024 chiều và hỗ trợ đa ngôn ngữ, phù hợp corpus RMIT có cả
# tiếng Việt lẫn tiếng Anh. Embedding được chuẩn hóa trước khi lưu để truy vấn
# cosine ổn định.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 16

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"
UPSERT_BATCH_SIZE = 128


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path,
                    "type": relative_path.split("/", 1)[0],
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        def split_text(text: str) -> list[str]:
            return splitter.split_text(text)
    except Exception:
        def split_text(text: str) -> list[str]:
            """Fallback splitter with fixed-size windows and overlap."""
            cleaned = text.strip()
            if not cleaned:
                return []

            chunks: list[str] = []
            start = 0
            text_length = len(cleaned)
            while start < text_length:
                end = min(start + CHUNK_SIZE, text_length)
                if end < text_length:
                    split_point = cleaned.rfind("\n\n", start, end)
                    if split_point == -1:
                        split_point = cleaned.rfind("\n", start, end)
                    if split_point == -1:
                        split_point = cleaned.rfind(". ", start, end)
                    if split_point == -1:
                        split_point = cleaned.rfind(" ", start, end)
                    if split_point != -1 and split_point > start:
                        end = split_point + 1

                chunk = cleaned[start:end].strip()
                if chunk:
                    chunks.append(chunk)

                if end >= text_length:
                    break

                start = max(end - CHUNK_OVERLAP, start + 1)

            return chunks

    chunks = []
    for doc in documents:
        content = doc.get("content", "").strip()
        if not content:
            continue
        for chunk_index, chunk_text in enumerate(split_text(content)):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [chunk["content"] for chunk in chunks],
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension không hợp lệ: {embeddings.shape}; "
            f"cần (*, {EMBEDDING_DIM})"
        )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def _chunk_id(chunk: dict) -> str:
    """Tạo ID ngắn, xác định và ổn định cho mỗi chunk."""
    metadata = chunk["metadata"]
    identity = f"{metadata['source_path']}:{metadata['chunk_index']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _collection_metadata() -> dict[str, Any]:
    return {
        "hnsw:space": "cosine",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "distance_to_score": "score = 1.0 - distance",
    }


def get_collection() -> Any:
    """Retrieve the ChromaDB collection used for indexing and searching.

    Returns:
        chromadb.api.models.Collection.Collection: The initialized collection.
    """
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Ensure collection exists (will create if missing)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=_collection_metadata(),
    )
    return collection

def get_embedding_model():
    """Instantiate and return the SentenceTransformer embedding model.

    Returns:
        SentenceTransformer: The embedding model defined by EMBEDDING_MODEL.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def index_to_vectorstore(chunks: list[dict]) -> int:
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        raise ValueError("Không có chunk để index")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Tất cả chunks phải được embed trước khi index")

    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Collection cũ dùng model/cấu hình khác không thể trộn chung embedding.
    try:
        existing = client.get_collection(name=COLLECTION_NAME)
        metadata = existing.metadata or {}
        incompatible = (
            metadata.get("embedding_model") != EMBEDDING_MODEL
            or metadata.get("embedding_dimension") != EMBEDDING_DIM
            or metadata.get("chunk_size") != CHUNK_SIZE
            or metadata.get("chunk_overlap") != CHUNK_OVERLAP
        )
        if incompatible:
            client.delete_collection(name=COLLECTION_NAME)
    except Exception as exc:
        # Chroma dùng các exception khác nhau giữa các phiên bản khi collection
        # chưa tồn tại; chỉ bỏ qua trường hợp đó, không che lỗi DB khác.
        if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
            raise

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=_collection_metadata(),
    )

    ids = [_chunk_id(chunk) for chunk in chunks]
    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        batch_ids = ids[start : start + UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=batch_ids,
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )

    # Upsert không tự xóa chunks của tài liệu đã bị đổi/xóa khỏi corpus.
    stored_ids = set(collection.get(include=[])["ids"])
    stale_ids = list(stored_ids - set(ids))
    for start in range(0, len(stale_ids), UPSERT_BATCH_SIZE):
        collection.delete(ids=stale_ids[start : start + UPSERT_BATCH_SIZE])

    return collection.count()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    indexed_count = index_to_vectorstore(chunks)
    print(f"[OK] Indexed {indexed_count} chunks to {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
