"""
Task 8 — PageIndex Vectorless RAG.

This module provides a lightweight, vector‑less retrieval pipeline using the PageIndex service.
It performs the following steps:
1. Load the `PAGEINDEX_API_KEY` from a `.env` file.
2. Convert each markdown document in `data/standardized/` to a temporary PDF (using fpdf2).
3. Upload the PDF to PageIndex via `client.submit_document()` and cache the returned `doc_id`s.
4. Perform a retrieval query, poll the retrieval job until it is `completed`, and return the top‑k results.

The cache of uploaded document IDs is persisted in `pageindex_doc_ids.json` so that subsequent runs do not re‑upload the same files.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, List, Dict

from dotenv import load_dotenv

# fpdf2 is used for quick PDF generation from plain text.
# If the library is unavailable we fall back to a dry‑run mode.
try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover
    FPDF = None  # type: ignore

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAGEINDEX_API_KEY: str = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR: Path = Path(__file__).parent.parent / "data" / "standardized"
TMP_PDF_DIR: Path = Path(__file__).parent / "tmp_pageindex_pdfs"
CACHE_PATH: Path = Path(__file__).parent / "pageindex_doc_ids.json"

# Ensure temporary directory exists.
TMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _load_cache() -> Dict[str, str]:
    """Load the JSON cache that maps markdown file paths to PageIndex `doc_id`s.

    Returns an empty dictionary if the cache file does not exist or is malformed.
    """
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_cache(cache: Dict[str, str]) -> None:
    """Persist the cache mapping to ``pageindex_doc_ids.json``.
    """
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

def _markdown_to_pdf(md_path: Path) -> Path:
    """Convert a markdown file to a simple PDF.

    The conversion strips markdown syntax and writes the raw text into a PDF using fpdf2.
    The resulting PDF is placed in ``TMP_PDF_DIR`` with the same stem and a ``.pdf`` extension.
    """
    if not FPDF:
        raise RuntimeError("fpdf2 library is required for PDF generation.")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=12)

    text = md_path.read_text(encoding="utf-8")
    # Very naive markdown stripping – sufficient for the exercise.
    for line in text.splitlines():
        # Remove leading markdown symbols (#, -, *, etc.)
        cleaned = line.lstrip('#*- ')  # noqa: E741
        pdf.multi_cell(w=0, h=5, txt=cleaned)

    pdf_path = TMP_PDF_DIR / f"{md_path.stem}.pdf"
    pdf.output(str(pdf_path))
    return pdf_path

# ---------------------------------------------------------------------------
# Core PageIndex interactions
# ---------------------------------------------------------------------------
def _get_client():
    """Instantiate a PageIndex client.

    Returns ``None`` if the SDK cannot be imported – callers must handle the dry‑run case.
    """
    try:
        from pageindex.client import PageIndexClient  # type: ignore
        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception:
        return None

def upload_documents() -> List[str]:
    """Upload all markdown documents as PDFs to PageIndex.

    Returns a list of the local markdown file paths that have been uploaded (or would be uploaded in dry‑run mode).
    The function caches the mapping ``{markdown_path: doc_id}`` in ``pageindex_doc_ids.json``.
    """
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {STANDARDIZED_DIR}")
        return []

    # Load existing cache to avoid re‑uploading unchanged files.
    cache = _load_cache()
    uploaded_paths: List[str] = []

    client = _get_client()
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY is not set. Running in dry‑run mode (no upload).")
        return [str(p) for p in md_files]
    if client is None:
        print("PageIndex SDK unavailable. Running in dry‑run mode.")
        return [str(p) for p in md_files]

    for md_path in md_files:
        md_str = str(md_path)
        if md_str in cache:
            # Already uploaded – skip.
            uploaded_paths.append(md_str)
            continue
        try:
            pdf_path = _markdown_to_pdf(md_path)
        except Exception as exc:
            print(f"Failed to convert {md_path.name} to PDF: {exc}")
            continue
        try:
            # The SDK expects a file‑like object; we open the PDF in binary mode.
            with pdf_path.open("rb") as fp:
                response = client.submit_document(file=fp)
            # Expected response contains a ``doc_id`` field.
            doc_id = response.get("doc_id")
            if not doc_id:
                print(f"Upload of {md_path.name} succeeded but no doc_id returned.")
                continue
            cache[md_str] = doc_id
            uploaded_paths.append(md_str)
            print(f"Uploaded {md_path.name} → doc_id={doc_id}")
        except Exception as exc:
            print(f"Error uploading {md_path.name}: {exc}")
            continue

    # Persist cache for future runs.
    _save_cache(cache)
    return uploaded_paths

def pageindex_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform a vector‑less retrieval against PageIndex.

    The function submits a retrieval job, polls until the job's ``status`` is ``completed``
    and then extracts the top‑k nodes. Each node is returned with ``content``, ``score``
    (currently a placeholder of ``1.0``), ``metadata`` and a ``source`` marker.
    """
    if not query.strip():
        return []
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY is not set – cannot perform live search.")
        return []

    client = _get_client()
    if client is None:
        print("PageIndex SDK unavailable – cannot perform live search.")
        return []

    # Submit the retrieval request.
    try:
        submit_resp = client.submit_retrieval(query=query, top_k=top_k)
        retrieval_id = submit_resp.get("retrieval_id")
    except Exception as exc:
        print(f"Failed to submit retrieval request: {exc}")
        return []

    if not retrieval_id:
        print("No retrieval_id returned from PageIndex.")
        return []

    # Poll until the job is completed.
    while True:
        try:
            status_resp = client.get_retrieval(retrieval_id)
        except Exception as exc:
            print(f"Error polling retrieval status: {exc}")
            break
        status = status_resp.get("status")
        if status == "completed":
            break
        if status in {"failed", "error"}:
            print(f"Retrieval job failed with status: {status}")
            return []
        # Simple back‑off – wait a couple of seconds before the next poll.
        time.sleep(2)

    # Parse retrieved nodes.
    nodes = status_resp.get("retrieved_nodes", [])
    results: List[Dict[str, Any]] = []
    for node in nodes[:top_k]:
        # The schema can vary; we attempt to extract a readable string.
        content_parts = []
        for section in node.get("relevant_contents", []):
            # Each section may be a dict with ``section_title`` and ``relevant_content``.
            title = section.get("section_title", "")
            body = section.get("relevant_content", "")
            if title:
                content_parts.append(f"{title}: {body}")
            else:
                content_parts.append(str(body))
        content = "\n".join(content_parts) or node.get("content", "")
        results.append({
            "content": content,
            "score": 1.0,  # Placeholder – PageIndex does not return a numeric score.
            "metadata": node,
            "source": "pageindex",
        })
    return results

# ---------------------------------------------------------------------------
# CLI entry‑point for quick manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Please set PAGEINDEX_API_KEY in a .env file (see README).")
    else:
        print("Uploading documents to PageIndex…")
        uploaded = upload_documents()
        print(f"Uploaded {len(uploaded)} document(s).")
        print("\nRunning a test query…")
        test_results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in test_results:
            print(f"[score placeholder] {r['content'][:200]}…")
