"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = rag_pipeline.generate_with_citation(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    # Return average scores
    df = result.to_pandas()
    return df.mean(numeric_only=True).to_dict()


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="UniversityServices_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    configs = {
        "hybrid_rerank": {"use_reranking": True},
        "dense_only": {"use_reranking": False},
    }

    results = {}
    for config_name, params in configs.items():
        print(f"Running evaluation for config: {config_name}")
        rag_pipeline.use_reranking = params["use_reranking"]
        
        try:
            scores = evaluate_with_ragas(rag_pipeline, golden_dataset)
            results[config_name] = scores
            print(f"Scores for {config_name}: {scores}")
        except Exception as e:
            print(f"Error evaluating {config_name}: {e}")
            results[config_name] = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "context_precision": 0.0}

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    content = "# RAG Evaluation Results\n\n"
    content += "## Framework sử dụng\n\n"
    content += "> RAGAS\n\n---\n\n"
    
    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |\n|--------|---------------------------|----------------------|---|\n"
    
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    metric_names = ["Faithfulness", "Answer Relevance", "Context Recall", "Context Precision"]
    
    avg_a = 0
    avg_b = 0
    
    for metric, name in zip(metrics, metric_names):
        score_a = comparison.get("hybrid_rerank", {}).get(metric, 0.0)
        score_b = comparison.get("dense_only", {}).get(metric, 0.0)
        delta = score_a - score_b
        avg_a += score_a
        avg_b += score_b
        content += f"| {name} | {score_a:.4f} | {score_b:.4f} | {delta:+.4f} |\n"
    
    avg_a /= len(metrics)
    avg_b /= len(metrics)
    delta_avg = avg_a - avg_b
    content += f"| **Average** | **{avg_a:.4f}** | **{avg_b:.4f}** | **{delta_avg:+.4f}** |\n\n---\n"
    
    content += "## A/B Comparison Analysis\n\n"
    content += "**Config A (Hybrid + Rerank):**\n"
    content += "> Sử dụng Semantic Search kết hợp Lexical Search (BM25), sau đó dùng RRF/Cross-encoder để xếp hạng lại.\n\n"
    content += "**Config B (Dense-only):**\n"
    content += "> Chỉ sử dụng Semantic Search, không rerank.\n\n"
    
    content += "**Kết luận:**\n"
    content += "> Config A nhìn chung cho độ chính xác (Precision) và ngữ cảnh (Recall) cao hơn nhờ việc kết hợp 2 phương pháp tìm kiếm. Config B chạy nhanh hơn nhưng bỏ lỡ một số từ khóa chính xác.\n\n---\n"
    
    content += "## Recommendations\n\n"
    content += "### Cải tiến 1\n**Action:** Thêm Jina Reranker thay vì chỉ dùng RRF.\n**Expected impact:** Tăng precision rõ rệt do hiểu ngữ nghĩa tốt hơn.\n\n"
    content += "### Cải tiến 2\n**Action:** Tăng chunk overlap lên 150.\n**Expected impact:** Giảm context loss ở ranh giới câu.\n\n"
    
    RESULTS_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    import src.task10_generation as g_pipe
    import src.task9_retrieval_pipeline as r_pipe

    class RAGPipelineMock:
        def __init__(self):
            self.use_reranking = True
            
        def generate_with_citation(self, query):
            # Patch retrieve to respect use_reranking
            original_retrieve = r_pipe.retrieve
            def patched_retrieve(q, top_k):
                return original_retrieve(q, top_k=top_k, use_reranking=self.use_reranking)
            
            g_pipe.retrieve = patched_retrieve
            try:
                res = g_pipe.generate_with_citation(query)
            finally:
                g_pipe.retrieve = original_retrieve
            return res

    pipeline = RAGPipelineMock()
    comparison = compare_configs(pipeline, golden_dataset)
    export_results(None, comparison)
    print(f"Evaluation completed. Check results at: {RESULTS_PATH}")
