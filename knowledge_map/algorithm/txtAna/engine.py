import os
import re

import chardet

# 从文件名推断作者
def infer_author_from_filename(filename: str) -> str:
    base = os.path.basename(filename or "").strip()
    base = re.sub(r"\.(txt|log|md|data)$", "", base, flags=re.I).strip()
    if not base:
        return ""

    parts = re.split(r"[-—–－_]+", base)
    if len(parts) < 2:
        return ""

    candidate = parts[-1].strip()
    if len(candidate) > 50:
        return ""
    return candidate

# 解码文本
def _decode_text(file_bytes: bytes) -> str:
    # 检测编码
    detected = chardet.detect(file_bytes)
    # 猜测编码
    guessed = detected.get("encoding") or "utf-8"

    try_encodings = [guessed, "utf-8", "gbk", "gb18030", "big5", "latin1"]
    for enc in try_encodings:
        if not enc:
            continue
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue

    return file_bytes.decode("utf-8", errors="ignore")

# 解析TXT文件
def parse_txt_bytes(file_bytes: bytes, filename: str) -> dict:
    result = {
        "filename": filename,
        "author": "",
        "pages": 1,
        "page_texts": [],
        "error": None,
    }

    try:
        content = _decode_text(file_bytes)
        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()

        result["author"] = infer_author_from_filename(filename)
        result["page_texts"] = [{"page": 1, "text": content}]
    except Exception as e:
        result["error"] = f"TXT parse failed: {e}"

    return result
