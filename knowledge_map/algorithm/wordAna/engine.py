import io
import os
import re
import shutil
import subprocess
import tempfile

from docx import Document

# 从文件名推断作者
def infer_author_from_filename(filename: str) -> str:
    base = os.path.basename(filename or "").strip()
    base = re.sub(r"\.(docx|doc)$", "", base, flags=re.I).strip()
    if not base:
        return ""

    parts = re.split(r"[-—–－_]+", base)
    if len(parts) < 2:
        return ""

    candidate = parts[-1].strip()
    if len(candidate) > 50:
        return ""
    return candidate

# 解析DOCX文件
def parse_docx_bytes(file_bytes: bytes, filename: str) -> dict:
    result = {
        "filename": filename,
        "author": "",
        "pages": 1,
        "page_texts": [],
        "error": None,
    }

    try:
        doc = Document(io.BytesIO(file_bytes))

        author = ""
        if doc.core_properties and doc.core_properties.author:
            author = str(doc.core_properties.author).strip()
        if not author:
            author = infer_author_from_filename(filename)

        chunks = []
        for para in doc.paragraphs:
            text = (para.text or "").strip()
            if text:
                chunks.append(text)

        full_text = "\n\n".join(chunks).strip()

        result["author"] = author
        result["page_texts"] = [{"page": 1, "text": full_text}]
    except Exception as e:
        result["error"] = f"DOCX parse failed: {e}"

    return result

# 运行DOC工具
def _run_doc_tool(file_path: str) -> tuple[str, str]:
    antiword = shutil.which("antiword")
    if antiword:
        cmd = [antiword, "-m", "UTF-8", "-w", "0", file_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.stdout, proc.stderr

    catdoc = shutil.which("catdoc")
    if catdoc:
        cmd = [catdoc, "-d", "utf-8", file_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.stdout, proc.stderr

    raise FileNotFoundError("antiword/catdoc not found")

# 解析DOC文件
def parse_doc_bytes(file_bytes: bytes, filename: str) -> dict:
    result = {
        "filename": filename,
        "author": infer_author_from_filename(filename),
        "pages": 1,
        "page_texts": [],
        "error": None,
    }
"""
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        stdout, stderr = _run_doc_tool(tmp_path)
        text = (stdout or "").replace("\r\n", "\n").replace("\r", "\n").strip()

        if not text:
            err = (stderr or "").strip()
            if not err:
                err = "empty output from antiword/catdoc"
            result["error"] = f"DOC parse failed: {err}"
            return result

        result["page_texts"] = [{"page": 1, "text": text}]
    except FileNotFoundError:
        result["error"] = "DOC parse failed: antiword/catdoc not installed on server"
    except Exception as e:
        result["error"] = f"DOC parse failed: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return result
"""