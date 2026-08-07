# algorithm/pdfAna/engine.py
import time
import re
import os

try:
    import pymupdf
except ImportError:  # pragma: no cover - compatibility fallback
    import fitz as pymupdf


TEXT_EXTRACT_KWARGS = {
    "x_tolerance": 4,
    "y_tolerance": 3,
    "layout": False,
}


def _norm_meta(meta: dict) -> dict:
    if not meta:
        return {}
    out = {}
    for k, v in meta.items():
        key = str(k).strip().lower()
        out[key] = v.strip() if isinstance(v, str) else v
    return out


def _get_meta_field(m: dict, *names: str) -> str:
    for n in names:
        v = m.get(n.lower())
        if v:
            return str(v).strip()
    return ""

# 从文件名推断作者
def infer_author_from_filename(filename: str) -> str:
    base = os.path.basename(filename or "").strip()
    base = base.strip('“”"\' ').strip()
    base = re.sub(r"\.pdf$", "", base, flags=re.I).strip()
    if not base:
        return ""

    parts = re.split(r"[-—–－_]+", base)
    if len(parts) < 2:
        return ""

    candidate = parts[-1].strip()
    candidate = re.sub(r"^(作者|Author)\s*[:：]?\s*", "", candidate, flags=re.I).strip()

    if not candidate:
        return ""
    if re.fullmatch(r"\d{2,}", candidate):
        return ""
    if len(candidate) > 50:
        return ""

    if re.search(r"[\u4e00-\u9fff]{2,}", candidate) or re.search(r"[A-Za-z]{3,}", candidate):
        return candidate

    return ""


def infer_author_from_first_page(page):
    author = ""
    rect = page.rect
    top_rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * 0.55)
    top_text = page.get_textbox(top_rect) or ""
    if not top_text:
        top_text = _extract_text_with_kwargs(page)

    m = re.search(r"(作者|Author|编著|著|作者简介)\s*[:：]?\s*([^\n]{2,80})", top_text, re.I)
    if m:
        author = m.group(2).strip()
        return {"author": author}

    return {"author": author}


def _extract_text_with_kwargs(page_obj) -> str:
    """Try tuned extraction first; fall back for older PyMuPDF versions."""
    try:
        return page_obj.get_text("text", sort=True) or ""
    except TypeError:
        return page_obj.get_text("text") or ""


def _compact_spaced_digit_groups(text: str) -> str:
    pattern = re.compile(r"(?<!\d)(?:\d\s+){3,}\d(?!\d)")

    def repl(match: re.Match) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return pattern.sub(repl, text)


def _clean_text_for_output(text: str) -> str:
    if not text:
        return ""

    text = (
        text.replace("\u00a0", " ")
        .replace("\u2002", " ")
        .replace("\u2003", " ")
        .replace("\u2009", " ")
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[A-Za-z])", "", text)

    # Keep CJK reading natural while preserving mixed-language boundaries.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[A-Za-z0-9])", " ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\n(?=[\u4e00-\u9fff])", " ", text)
    text = re.sub(r"(?<=[A-Za-z0-9,.;:!?])\n(?=[A-Za-z0-9])", " ", text)

    # Remove artificial spacing introduced by PDF text positioning.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。！？；：、）】》〉」』])", "", text)
    text = re.sub(r"(?<=[（【《〈「『])\s+(?=[\u4e00-\u9fff])", "", text)
    text = _compact_spaced_digit_groups(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# 解析PDF文件
def parse_pdf_bytes(file_bytes: bytes, filename: str) -> dict:
    """核心解析入口：algorithm 层只认 bytes + filename"""
    file_start = time.time() 
    # 初始化结果
    result = {
        "filename": filename,
        "author": "",
        "pages": 0,
        "page_texts": [],
        "error": None,
    }

    if not (filename or "").lower().endswith(".pdf"):
        result["error"] = f"Invalid file format: {filename}"
        return result

    try:
        # 0) 优先从文件名推断作者
        author = infer_author_from_filename(filename)

        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            # 1) 文件名拿不到，再尝试 metadata
            if not author:
                meta = _norm_meta(pdf.metadata or {})
                meta_author = _get_meta_field(meta, "author")
                author = meta_author or author

            # 2) metadata 还没有，再用第一页推断
            if not author and pdf.page_count > 0:
                guess = infer_author_from_first_page(pdf.load_page(0))
                author = guess.get("author", "") or ""

            result["author"] = author

            for page_index in range(pdf.page_count):
                page = pdf.load_page(page_index)
                words = page.get_text("words", sort=True) or []
                rect = page.rect
                x_mid = rect.x0 + (rect.width / 2.0)
                left_words = [w for w in words if float(w[0]) < x_mid]
                right_words = [w for w in words if float(w[0]) >= x_mid]

                column_count = 1
                if len(words) > 0 and len(left_words) > len(words) * 0.3 and len(right_words) > len(words) * 0.3:
                    column_count = 2

                if column_count == 1:
                    # 提取文本
                    raw_text = _extract_text_with_kwargs(page)
                else:
                    # 提取左右文本
                    left_bbox = pymupdf.Rect(rect.x0, rect.y0, x_mid, rect.y1)
                    right_bbox = pymupdf.Rect(x_mid, rect.y0, rect.x1, rect.y1)
                    try:
                        left_text = page.get_textbox(left_bbox) or ""
                        right_text = page.get_textbox(right_bbox) or ""
                        raw_text = left_text + "\n" + right_text
                    except Exception:
                        raw_text = ""

                # 清洗文本
                clean_text = _clean_text_for_output(raw_text)
                result["page_texts"].append(
                    {
                        "page": page_index + 1,
                        "raw_text": raw_text,
                        "clean_text": clean_text,
                        "text": clean_text,
                    }
                )

            result["pages"] = len(result["page_texts"])

    except Exception as e:
        result["error"] = str(e)

    result["cost"] = round(time.time() - file_start, 2)
    return result

# 检测PDF类型
def detect_pdf_type(file_bytes: bytes, sample_pages: int = 2, min_text_len: int = 30) -> str:
    """
    返回:
    - "text"   : 文字版 PDF
    - "scan"   : 扫描版 / 图片型 PDF
    """
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            total_text_len = 0
            max_pages = min(sample_pages, pdf.page_count)

            for page_index in range(max_pages):
                page = pdf.load_page(page_index)
                text = _extract_text_with_kwargs(page)
                text = "".join(text.split())  # 去掉空白字符
                total_text_len += len(text)

            if total_text_len >= min_text_len:
                return "text"
            return "scan"
    except Exception:
        # 打不开或异常时，保守起见走扫描版
        return "scan"
