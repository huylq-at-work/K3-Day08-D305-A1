"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import hashlib
import io
import json
import re
from pathlib import Path

from markitdown import MarkItDown, StreamInfo

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _looks_like_html(text: str) -> bool:
    """Crawler fallback (requests) lưu nguyên trang HTML thay vì markdown."""
    head = text.lstrip()[:2000].lower()
    return "<!doctype html" in head or "<html" in head or "<div" in head


def _html_to_markdown(html: str, md: MarkItDown) -> str:
    """
    Bóc nội dung đọc được ra khỏi trang HTML thô.

    Task 2 chạy fallback bằng `requests` (crawl4ai không cài được), nên
    content_markdown thực chất là toàn bộ trang: script tracking, menu, SVG path.
    Nếu đưa thẳng vào index thì chunk chứa `<div id=...>` và toạ độ đường vẽ, và
    retrieval sẽ trả về đúng mớ đó thay vì nội dung bài viết.
    """
    # Bỏ script/style/svg/noscript trước — MarkItDown giữ lại nội dung text bên trong.
    for tag in ("script", "style", "svg", "noscript", "iframe"):
        html = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>", " ", html, flags=re.DOTALL | re.IGNORECASE
        )

    result = md.convert_stream(
        io.BytesIO(html.encode("utf-8")),
        stream_info=StreamInfo(extension=".html", mimetype="text/html"),
    )
    text = result.text_content
    text = _drop_navigation_lines(text)

    # Gom dòng trống thừa do menu/nav rỗng để chunk không toàn khoảng trắng.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Một dòng menu điển hình: "* [Hội nghị ABC](/vi/su-kien/tat-ca-cac-su-kien/2026/...)"
# — gần như toàn bộ ký tự nằm trong URL, không mang thông tin trả lời được.
# Cho phép nhiều dấu đầu dòng lồng nhau ("* + [..](..)") — MarkItDown sinh ra
# dạng này khi menu là <ul> lồng <ul>.
_LINK_ONLY = re.compile(r"^[\s*+\-]*\[[^\]]*\]\([^)]*\)\s*$")
_LINK_MARKUP = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _drop_navigation_lines(text: str) -> str:
    """
    Bỏ các dòng chỉ chứa link điều hướng.

    Sau khi bóc HTML, menu và sidebar của trang còn lại dưới dạng danh sách link
    thuần. Chúng trông giống nội dung nên vẫn được index, rồi chiếm suất trong
    top_k và đẩy đoạn văn thật ra ngoài — quan sát thực tế: 3/5 kết quả cho câu
    hỏi giờ mở cửa thư viện là menu của trang sự kiện.

    Chỉ bỏ dòng CHỈ có link, hoặc dòng mà phần chữ hiển thị quá ngắn so với độ
    dài URL. Câu văn bình thường có chèn link vẫn được giữ.
    """
    kept = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            kept.append(line)
            continue

        if _LINK_ONLY.match(stripped):
            continue

        # Dòng nhiều link mà phần chữ ngoài link không đáng kể -> menu.
        links = _LINK_MARKUP.findall(stripped)
        if len(links) >= 2:
            text_outside = _LINK_MARKUP.sub("", stripped).strip(" *+-|·•\t")
            if len(text_outside) < 20:
                continue

        kept.append(line)

    return "\n".join(kept)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    seen_hashes: dict[str, str] = {}  # hash nội dung -> tên file đã ghi

    # Khi hai file trùng nội dung, giữ file có tên "sạch" — bản `-duplicate`/`-copy`
    # bị bỏ, để citation hiển thị đúng tên bài chứ không phải bản sao.
    def _priority(p: Path) -> tuple[int, str]:
        name = p.stem.lower()
        return (1 if ("duplicate" in name or "copy" in name) else 0, p.name)

    for filepath in sorted(news_dir.iterdir(), key=_priority):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            raw = data.get("content_markdown", "")

            if _looks_like_html(raw):
                body = _html_to_markdown(raw, md)
                print(f"  [HTML] {len(raw):,} ky tu -> {len(body):,} ky tu sau khi boc")
            else:
                body = raw

            # Bài trùng nội dung chiếm chỗ của bài khác trong top_k khi retrieval.
            fingerprint = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if fingerprint in seen_hashes:
                print(f"  [SKIP] Trung noi dung voi {seen_hashes[fingerprint]}")
                continue
            seen_hashes[fingerprint] = filepath.name

            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm metadata header
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            output_path.write_text(header + body, encoding="utf-8")
            print(f"  [OK] Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n[OK] Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
