"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import sys
# Removed duplicate import and old format_context implementation
# Ensure src directory is in PYTHONPATH for direct script execution
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Model mặc định khi dùng OpenRouter: bản ":free" không tính phí.
LLM_MODEL = "inclusionai/ling-3.0-flash:free"

# Model thay thế khi bị rate limit (429). Đây là model ID CỦA OPENROUTER, không
# phải fallback sang provider khác — việc chọn provider nằm ở generate_with_citation().
FALLBACK_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-pro",
]



# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into a context string for the prompt.

    Tên tài liệu nằm ở metadata['source'] (Task 4 ghi md_file.name vào đó). Field
    'source' ở cấp ngoài là KÊNH truy xuất do Task 9 gắn ('hybrid' | 'pageindex'),
    không phải tên nguồn — dùng nó làm nhãn citation thì LLM chỉ thấy "hybrid" ở
    mọi đoạn, không phân biệt được tài liệu nào, và sẽ từ chối khẳng định.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float, 'source': str}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # `or {}` thay vì default {} — Task 9 có thể trả metadata=None từ PageIndex.
        metadata = chunk.get('metadata') or {}
        source = metadata.get('source') or f"Source {i}"
        doc_type = metadata.get('type', "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)




# =============================================================================
# GENERATION
# =============================================================================

# Số lần thử lại khi lỗi mạng thoáng qua.
LLM_MAX_ATTEMPTS = 3
_TRANSIENT_MARKERS = ("connection", "timeout", "timed out", "temporarily", "502", "503", "504")


def _call_llm(api_key: str, base_url: str | None, model_id: str, user_message: str) -> str:
    """
    Gọi LLM, thử lại khi gặp lỗi mạng thoáng qua.

    Không có retry thì một lần rớt mạng sẽ biến thành NỘI DUNG câu trả lời
    ("LLM generation error: Connection error."). Quan sát thực tế: eval chấm
    Answer Relevancy = 0.000 cho những câu đó, dù retrieval hoàn toàn đúng —
    tức là một sự cố mạng thoáng qua làm hỏng cả bảng điểm.
    """
    import time

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    last_exc: Exception | None = None

    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()

            # Rate limit để vòng ngoài đổi sang model khác, không retry ở đây.
            if "429" in message or "rate limit" in message:
                raise

            if not any(m in message for m in _TRANSIENT_MARKERS):
                raise

            if attempt < LLM_MAX_ATTEMPTS:
                wait = 2 ** (attempt - 1)
                print(f"[WARN] LLM lỗi tạm thời ({exc}); thử lại sau {wait}s")
                time.sleep(wait)

    raise last_exc if last_exc else RuntimeError("LLM call failed")

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    use_semantic: bool = True,
    use_lexical: bool = True,
    use_reranking: bool = True,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    try:
        chunks = retrieve(
            query,
            top_k=top_k,
            use_semantic=use_semantic,
            use_lexical=use_lexical,
            use_reranking=use_reranking,
        )
    except ValueError:
        # Cấu hình sai từ UI (tắt cả hai retriever) — để lộ ra thay vì nuốt.
        raise
    except Exception:
        chunks = []

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered) if reordered else ""

    if not reordered:
        return {
            "answer": (
                "Tôi chưa có đủ dữ liệu truy xuất để trả lời chắc chắn. "
                "Hãy hoàn thiện các bước retrieval trước, rồi thử lại câu hỏi này."
            ),
            "sources": [],
            "retrieval_source": "none",
        }

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Chọn provider theo key thực sự có. Trước đây base_url luôn trỏ OpenRouter nên
    # key OpenAI gửi vào đó sẽ bị từ chối xác thực, và danh sách FALLBACK_MODELS
    # ("openai/...", "google/...") chỉ là model ID của OpenRouter chứ không phải
    # fallback sang provider khác.
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

    if openrouter_key:
        api_key = openrouter_key
        base_url = "https://openrouter.ai/api/v1"
        models_to_try = [LLM_MODEL] + FALLBACK_MODELS
    elif openai_key:
        api_key = openai_key
        base_url = None  # endpoint mặc định của OpenAI
        # Model ID trên OpenAI không có tiền tố "openai/" như trên OpenRouter.
        models_to_try = ["gpt-4o-mini"]
    else:
        api_key = None
        base_url = None
        models_to_try = []

    if api_key:
        # Attempt primary model first, then fallbacks on rate limit (429)
        answer = ""
        for model_id in models_to_try:
            try:
                answer = _call_llm(api_key, base_url, model_id, user_message)
                # Successful generation, break out of fallback loop
                break
            except Exception as exc:
                # If rate limited, try next model; otherwise keep error message
                if "429" in str(exc) or "rate limit" in str(exc).lower():
                    # Continue to next fallback model
                    continue
                else:
                    answer = f"LLM generation error: {exc}"
                    break
        if not answer:
            answer = "LLM generation failed due to rate limits and no fallback keys available."
    else:
        answer = (
            "Chế độ generation hiện đang ở dạng skeleton. "
            "Khi có API key, hệ thống sẽ trả lời dựa trên context và gắn citation."
        )

    retrieval_source = reordered[0].get("source", "hybrid") if reordered else "none"
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
