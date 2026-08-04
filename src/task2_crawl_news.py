"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/vi/su-kien/hoi-thao-thong-tin/ug",
    "https://www.rmit.edu.vn/vi/su-kien/tat-ca-cac-su-kien/2026/rmit-tech-camp",
    "https://www.rmit.edu.vn/libraryvn/about-us/hours-and-locations",
    "https://www.rmit.edu.vn/vi/tin-tuc/tat-ca-tin-tuc/2025/oct/rmit-viet-nam-trao-hoc-bong-tri-gia-47-5-ti-dong-nam-2025",
    # Link thứ 5 trùng theo danh sách bạn cung cấp
    "https://www.rmit.edu.vn/vi/tin-tuc/tat-ca-tin-tuc/2025/oct/rmit-viet-nam-trao-hoc-bong-tri-gia-47-5-ti-dong-nam-2025",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

            metadata = getattr(result, "metadata", {}) or {}
            title = metadata.get("title") if isinstance(metadata, dict) else None
            content_markdown = (
                getattr(result, "markdown", None)
                or getattr(result, "cleaned_html", None)
                or getattr(result, "html", None)
                or ""
            )

            if not content_markdown:
                raise ValueError("Empty crawl content")

            return {
                "url": url,
                "title": title or "Unknown",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": content_markdown,
            }
    except Exception as crawl_err:
        # Fallback đơn giản để tránh dừng pipeline khi Crawl4AI/browser gặp lỗi.
        try:
            resp = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                },
            )
            resp.raise_for_status()
            html = resp.text

            title = "Unknown"
            lower_html = html.lower()
            start = lower_html.find("<title>")
            end = lower_html.find("</title>")
            if start != -1 and end != -1 and end > start:
                title = html[start + 7:end].strip()

            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": html,
                "fallback": "requests",
                "crawl_error": str(crawl_err),
            }
        except Exception as fallback_err:
            return {
                "url": url,
                "title": "Unknown",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": "",
                "crawl_error": f"crawl4ai_error={crawl_err}; fallback_error={fallback_err}",
            }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
