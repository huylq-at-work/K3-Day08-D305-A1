"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai RMIT Vietnam — rmit.edu.vn):
    - https://www.rmit.edu.vn/study-at-rmit/tuition-fees
    - https://www.rmit.edu.vn/study-at-rmit/scholarships/...
    - https://www.rmit.edu.vn/students/my-studies/fees-and-payments

Gợi ý văn bản (chủ đề dịch vụ đại học):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Chính sách học bổng (Scholarship eligibility)
    - Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
    - Hướng dẫn đăng ký học phần qua cổng thông tin sinh viên (Course Registration)

Lưu ý: một số trang trường (vd VinUni, Fulbright) chặn bot crawler mặc định (HTTP 403) —
không phải lỗi của bạn, đó là cấu hình WAF/Cloudflare phía server. Đổi sang trang khác
thay vì cố vượt qua, và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/"
        "pdfs/study-at-rmit/tuition-fees/student-fees-and-charges-guide-06-2026.pdf",
        "student-fees-and-charges-guide-2026-rmit.pdf",
    ),
    (
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/study-at-rmit/scholarships/english-pdf/"
        "rmit-university-vietnam-scholarship-terms-and-conditions.pdf",
        "scholarship-terms-and-conditions-rmit.pdf",
    ),
    (
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/students/enrolment/Enrolment-Variation-Form%20%28f%29%201.pdf",
        "enrolment-variation-form-rmit.pdf",
    ),
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Tải và xác thực một tài liệu PDF/DOCX gốc vào ``DATA_DIR``."""
    if Path(filename).name != filename:
        raise ValueError("filename chỉ được chứa tên file, không chứa đường dẫn")

    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ValueError("Chỉ hỗ trợ file PDF hoặc DOCX")

    setup_directory()
    response = requests.get(
        url,
        headers={"User-Agent": "UniversityServicesRAG/1.0"},
        timeout=60,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    content = response.content
    signatures = {".pdf": b"%PDF-", ".docx": b"PK\x03\x04"}
    if not content.startswith(signatures[suffix]):
        raise ValueError(
            f"Nội dung tải về không phải {suffix[1:].upper()} "
            f"(Content-Type: {content_type or 'không có'})"
        )

    filepath = DATA_DIR / filename
    filepath.write_bytes(content)
    print(f"[OK] Da tai: {filepath} ({len(content):,} bytes)")
    return filepath


def download_legal_documents() -> list[Path]:
    """Tải bộ tài liệu chính sách công khai dùng cho Task 1."""
    return [download_file(url, filename) for url, filename in LEGAL_DOCUMENTS]


if __name__ == "__main__":
    download_legal_documents()
