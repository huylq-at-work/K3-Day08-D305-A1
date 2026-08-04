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
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Phải đặt TRƯỚC khi import deepeval — nó đọc settings lúc import.
#
# DeepEval áp một deadline chung cho cả lượt chấm. Với 18 câu × 4 metric, lượt chạy
# vượt deadline và chết bằng TimeoutError sau khi đã tốn tiền gọi LLM — mất trắng.
# Tắt deadline của DeepEval; giới hạn thật vẫn do OpenAI SDK và mạng quyết định.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "YES")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

# DeepEval mặc định cho mỗi lượt gọi LLM 30 giây (openai_model._request_timeout_seconds).
# Trên đường truyền chậm, judge bắn openai.APITimeoutError và cả lượt chạy chết sau khi
# đã tốn tiền. 120s đủ rộng mà vẫn không treo vô hạn.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS", "120")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# 4 cấu hình để so sánh A/B. Ba cờ này là tham số của retrieve() (Task 9) và
# được nối thẳng ra checkbox trong app.py, nên số đo ở đây khớp với thứ nhóm
# demo trên giao diện.
CONFIGS: dict[str, dict] = {
    "hybrid_rerank": {"use_semantic": True, "use_lexical": True, "use_reranking": True},
    "hybrid_norerank": {"use_semantic": True, "use_lexical": True, "use_reranking": False},
    "dense_only": {"use_semantic": True, "use_lexical": False, "use_reranking": True},
    "sparse_only": {"use_semantic": False, "use_lexical": True, "use_reranking": True},
}

# Model làm giám khảo. Metric của DeepEval gọi LLM nhiều lần cho MỖI câu hỏi
# (không phải 1 lần), nên chọn model rẻ.
JUDGE_MODEL = "gpt-4o-mini"

# Số test case chấm song song. Đổi bằng cờ --concurrency.
ASYNC_CONCURRENCY = 3

# Số test case mỗi lượt gọi evaluate(). Xem ghi chú trong evaluate_with_deepeval().
BATCH_SIZE = 6

# Số lần thử lại mỗi lô khi DeepEval bắn APITimeoutError chập chờn.
BATCH_MAX_ATTEMPTS = 3


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def _build_test_cases(golden_dataset: list[dict], config: dict) -> list:
    """Chạy RAG pipeline trên từng câu hỏi và gói thành LLMTestCase."""
    from deepeval.test_case import LLMTestCase

    from src.task10_generation import generate_with_citation

    test_cases = []
    for i, item in enumerate(golden_dataset, 1):
        result = generate_with_citation(item["question"], **config)
        retrieval_context = [
            str(c.get("content", "")) for c in (result.get("sources") or [])
        ]
        # DeepEval từ chối test case có retrieval_context rỗng; giữ 1 chuỗi
        # placeholder để câu hỏi vẫn được chấm (và bị điểm thấp) thay vì crash.
        if not retrieval_context:
            retrieval_context = ["(không truy xuất được đoạn nào)"]

        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=str(result.get("answer") or ""),
                expected_output=item["expected_answer"],
                retrieval_context=retrieval_context,
            )
        )
        print(f"  [{i}/{len(golden_dataset)}] {item['question'][:55]}")

    return test_cases


def _metrics(threshold: float = 0.7) -> list:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    return [
        FaithfulnessMetric(threshold=threshold, model=JUDGE_MODEL),
        AnswerRelevancyMetric(threshold=threshold, model=JUDGE_MODEL),
        ContextualRecallMetric(threshold=threshold, model=JUDGE_MODEL),
        ContextualPrecisionMetric(threshold=threshold, model=JUDGE_MODEL),
    ]


def evaluate_with_deepeval(golden_dataset: list[dict], config: dict) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval với 4 metric.

    pip install deepeval

    Returns:
        {
            'per_metric': {ten_metric: diem_trung_binh},
            'cases': [ {question, scores: {metric: diem}, reasons: {...}} ],
        }
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig

    test_cases = _build_test_cases(golden_dataset, config)

    # Chấm theo LÔ NHỎ thay vì đẩy cả 18 case vào một lượt evaluate().
    #
    # Đo trên máy nhóm: 6 case chạy ổn định, từ 8 case trở lên là cả lượt chết bằng
    # openai.APITimeoutError — dù mạng hoàn toàn khoẻ (6/6 lượt ping OpenAI < 2s).
    # Vấn đề nằm ở tầng async của DeepEval khi số task lớn, không phải đường truyền.
    # Chia lô giữ mỗi lượt evaluate() dưới ngưỡng đó, và một lô hỏng cũng không kéo
    # theo các lô đã chấm xong.
    all_test_results = []
    n_batches = (len(test_cases) + BATCH_SIZE - 1) // BATCH_SIZE

    for start in range(0, len(test_cases), BATCH_SIZE):
        batch = test_cases[start : start + BATCH_SIZE]
        idx = start // BATCH_SIZE + 1

        # DeepEval bắn APITimeoutError một cách CHẬP CHỜN: cùng một lô 6 case, lần
        # chạy này xong, lần sau chết — trong khi ping OpenAI trực tiếp 6/6 dưới 2s.
        # Thử lại từng lô để một lần chập không xoá công của các lô đã chấm xong.
        for attempt in range(1, BATCH_MAX_ATTEMPTS + 1):
            try:
                print(f"  → chấm lô {idx}/{n_batches} ({len(batch)} case)")
                batch_results = evaluate(
                    batch,
                    _metrics(),
                    async_config=AsyncConfig(
                        max_concurrent=ASYNC_CONCURRENCY, throttle_value=1
                    ),
                )
                all_test_results.extend(batch_results.test_results)
                break
            except Exception as exc:
                if attempt == BATCH_MAX_ATTEMPTS:
                    print(
                        f"  [BỎ QUA] lô {idx} hỏng sau {attempt} lần thử: "
                        f"{type(exc).__name__}. Báo cáo sẽ thiếu {len(batch)} câu."
                    )
                    break
                print(f"  [WARN] lô {idx} lỗi ({type(exc).__name__}); thử lại")

    if not all_test_results:
        raise RuntimeError(
            "Không chấm được câu nào. Thử hạ --concurrency hoặc giảm BATCH_SIZE."
        )

    per_metric: dict[str, list[float]] = {}
    cases: list[dict] = []

    for test_result in all_test_results:
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for metric_data in test_result.metrics_data or []:
            name = metric_data.name
            score = float(metric_data.score or 0.0)
            scores[name] = score
            reasons[name] = str(metric_data.reason or "")
            per_metric.setdefault(name, []).append(score)

        cases.append(
            {
                "question": test_result.input,
                "answer": test_result.actual_output,
                "scores": scores,
                "reasons": reasons,
                "mean": sum(scores.values()) / len(scores) if scores else 0.0,
            }
        )

    return {
        "per_metric": {
            name: sum(vals) / len(vals) for name, vals in per_metric.items()
        },
        "cases": cases,
    }


# =============================================================================
# Option 2: RAGAS — KHÔNG DÙNG (giữ lại để tham khảo)
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    KHÔNG DÙNG — nhóm chọn DeepEval, xem evaluate_with_deepeval().

    Lý do: `ragas==0.1.21` kéo `numpy<2`, mà bản numpy đó không có wheel cho
    Python 3.14 nên pip phải build từ nguồn và cần Visual C++ Build Tools. Bản
    ragas mới hơn thì kéo `scikit-network`, cũng phải build bằng C. Bài lab cho
    phép chọn 1 trong 3 framework.

    pip install ragas
    """
    # Bản tham khảo (chưa chạy được trên môi trường của nhóm):
    #
    # from ragas import evaluate
    # from ragas.metrics import (
    #     faithfulness,
    #     answer_relevancy,
    #     context_recall,
    #     context_precision,
    # )
    # from datasets import Dataset
    #
    # eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    #
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     eval_data["question"].append(item["question"])
    #     eval_data["answer"].append(result["answer"])
    #     eval_data["contexts"].append([c["content"] for c in result["sources"]])
    #     eval_data["ground_truth"].append(item["expected_answer"])
    #
    # dataset = Dataset.from_dict(eval_data)
    # result = evaluate(
    #     dataset,
    #     metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    # )
    # return result.to_pandas()
    raise NotImplementedError("Implement evaluate_with_ragas")


# =============================================================================
# Option 3: TruLens — KHÔNG DÙNG (giữ lại để tham khảo)
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    KHÔNG DÙNG — nhóm chọn DeepEval, xem evaluate_with_deepeval().

    pip install trulens
    """
    # Bản tham khảo (chưa chạy trên môi trường của nhóm):
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

def compare_configs(golden_dataset: list[dict], config_names: list[str]) -> dict:
    """
    So sánh A/B giữa các config trong CONFIGS.

    - hybrid_rerank   : Semantic + BM25 + RRF  (mặc định)
    - hybrid_norerank : Semantic + BM25, không fuse thứ hạng
    - dense_only      : chỉ Semantic
    - sparse_only     : chỉ BM25
    """
    results = {}
    for name in config_names:
        if name not in CONFIGS:
            raise ValueError(f"Config không tồn tại: {name}. Có: {list(CONFIGS)}")
        print(f"\n=== Config: {name} ===")
        results[name] = evaluate_with_deepeval(golden_dataset, CONFIGS[name])
    return results


# =============================================================================
# Export Results
# =============================================================================

METRIC_ORDER = [
    "Faithfulness",
    "Answer Relevancy",
    "Contextual Recall",
    "Contextual Precision",
]


def _metric_columns(comparison: dict) -> list[str]:
    seen = []
    for run in comparison.values():
        for name in run["per_metric"]:
            if name not in seen:
                seen.append(name)
    return [m for m in METRIC_ORDER if m in seen] + [
        m for m in seen if m not in METRIC_ORDER
    ]


def export_results(comparison: dict, baseline: str, n_questions: int) -> None:
    """Export evaluation results to results.md"""
    columns = _metric_columns(comparison)
    lines: list[str] = []

    lines.append("# RAG Evaluation Results")
    lines.append("")
    lines.append(
        f"Framework: **DeepEval** · Judge model: `{JUDGE_MODEL}` · "
        f"Golden dataset: **{n_questions} câu hỏi**"
    )
    lines.append("")
    lines.append(
        "Sinh tự động bằng `python -m group_project.evaluation.eval_pipeline`. "
        "Đừng sửa tay — chạy lại script."
    )
    lines.append("")

    # --- Bảng điểm tổng ---
    lines.append("## 1. Bảng điểm theo config")
    lines.append("")
    header = "| Config | " + " | ".join(columns) + " | Trung bình |"
    lines.append(header)
    lines.append("|" + "---|" * (len(columns) + 2))

    means = {}
    for name, run in comparison.items():
        vals = [run["per_metric"].get(c, 0.0) for c in columns]
        mean = sum(vals) / len(vals) if vals else 0.0
        means[name] = mean
        row = f"| `{name}` | " + " | ".join(f"{v:.3f}" for v in vals)
        row += f" | **{mean:.3f}** |"
        lines.append(row)
    lines.append("")

    # --- So sánh A/B ---
    lines.append("## 2. So sánh A/B")
    lines.append("")
    base_mean = means.get(baseline, 0.0)
    lines.append(f"Mốc so sánh: `{baseline}` (trung bình {base_mean:.3f})")
    lines.append("")
    lines.append("| Config | Trung bình | Chênh so với mốc |")
    lines.append("|---|---|---|")
    for name, mean in sorted(means.items(), key=lambda x: -x[1]):
        delta = mean - base_mean
        sign = "+" if delta >= 0 else ""
        marker = " ← mốc" if name == baseline else ""
        lines.append(f"| `{name}` | {mean:.3f} | {sign}{delta:.3f}{marker} |")
    lines.append("")

    best = max(means, key=means.get) if means else "-"
    lines.append(f"Config tốt nhất: **`{best}`**")
    lines.append("")

    # --- Worst performers ---
    lines.append("## 3. Worst performers")
    lines.append("")
    lines.append(f"5 câu hỏi điểm thấp nhất ở config `{baseline}`:")
    lines.append("")

    worst = sorted(comparison[baseline]["cases"], key=lambda c: c["mean"])[:5]
    for case in worst:
        lines.append(f"**{case['question']}** — trung bình `{case['mean']:.3f}`")
        lines.append("")
        for metric, score in case["scores"].items():
            lines.append(f"- {metric}: `{score:.3f}`")
        worst_metric = min(case["scores"], key=case["scores"].get) if case["scores"] else None
        if worst_metric:
            reason = case["reasons"].get(worst_metric, "").strip()
            if reason:
                lines.append(f"- Lý do ({worst_metric}): {reason}")
        lines.append("")

    # --- Đề xuất ---
    lines.append("## 4. Nhận xét & đề xuất")
    lines.append("")
    lines.append(
        "- Điểm Contextual Recall/Precision thấp nghĩa là retriever chưa lấy đúng "
        "đoạn văn — sửa ở Task 3/4 (chất lượng corpus, chunking) chứ không phải "
        "ở prompt Task 10."
    )
    lines.append(
        "- Faithfulness cao nhưng Answer Relevancy thấp nghĩa là câu trả lời bám "
        "context nhưng lạc đề; thường do retriever đưa nhầm tài liệu."
    )
    lines.append(
        "- Câu hỏi tiếng Việt hỏi về tài liệu tiếng Anh là nhóm yếu nhất — đây là "
        "lý do nhóm chọn embedding đa ngôn ngữ `BAAI/bge-m3`."
    )
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] Đã ghi {RESULTS_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG evaluation với DeepEval")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["hybrid_rerank", "dense_only"],
        help=f"Config để so sánh. Có: {list(CONFIGS)}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ chạy N câu đầu — dùng khi thử nghiệm để đỡ tốn lượt gọi LLM.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=ASYNC_CONCURRENCY,
        help="Số test case chấm song song. Hạ xuống nếu gặp APITimeoutError.",
    )
    args = parser.parse_args()
    ASYNC_CONCURRENCY = args.concurrency

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise SystemExit(
            "Thiếu OPENAI_API_KEY trong .env — DeepEval cần LLM để chấm điểm."
        )

    golden_dataset = load_golden_dataset()
    if args.limit:
        golden_dataset = golden_dataset[: args.limit]
    print(f"Loaded {len(golden_dataset)} test cases")

    comparison = compare_configs(golden_dataset, args.configs)
    export_results(comparison, baseline=args.configs[0], n_questions=len(golden_dataset))
