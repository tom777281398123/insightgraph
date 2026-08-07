import json
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

_DOLPHIN_ROOT = Path(__file__).resolve().parents[1] / "Dolphin"
sys.path.insert(0, str(_DOLPHIN_ROOT))

from algorithm.Dolphin.demo_page import DOLPHIN, process_document

_MODEL = None
_MODEL_LOCK = threading.Lock()


def init_dolphin_model(model_path: str) -> None:
    global _MODEL
    if _MODEL is not None:
        return

    with _MODEL_LOCK:
        if _MODEL is not None:
            return
        _MODEL = DOLPHIN(model_path)


def _ensure_model():
    if _MODEL is None:
        raise RuntimeError("Dolphin model not initialized. Call init_dolphin_model() on startup.")
    return _MODEL


def _clean_text_for_output(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[A-Za-z])", "", text)

    # Keep CJK reading natural while preserving mixed-language boundaries.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[A-Za-z0-9])", " ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\n(?=[\u4e00-\u9fff])", " ", text)
    text = re.sub(r"(?<=[A-Za-z0-9,.;:!?])\n(?=[A-Za-z0-9])", " ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 解析PDF文件
def parse_pdf_bytes_with_dolphin(
    pdf_bytes: bytes,
    filename: str,
    model_path: str,
    max_batch_size: int = 4,
    enable_postprocess: bool = True,
) -> Dict[str, Any]:
    """Parse PDF bytes via Dolphin and return normalized API output."""
    global _MODEL
    if _MODEL is None:
        init_dolphin_model(model_path)

    model = _ensure_model()

    with tempfile.TemporaryDirectory(prefix="dolphin_job_") as td:
        td = Path(td)
        inp = td / "input.pdf"
        out_dir = td / "out"

        inp.write_bytes(pdf_bytes)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "markdown" / "figures").mkdir(parents=True, exist_ok=True)
        (out_dir / "output_json").mkdir(parents=True, exist_ok=True)
        (out_dir / "layout_visualization").mkdir(parents=True, exist_ok=True)

        combined_json_path, _all_results = process_document(
            document_path=str(inp),
            model=model,
            save_dir=str(out_dir),
            max_batch_size=max_batch_size,
            enable_postprocess=enable_postprocess,
        )

        combined_obj = json.loads(Path(combined_json_path).read_text(encoding="utf-8"))

        page_texts = []
        for page in combined_obj.get("pages", []):
            page_no = page.get("page_number", 0)
            elements = page.get("elements", []) or []

            text_chunks = []
            for e in elements:
                label = (e.get("label") or "").strip().lower()
                raw_text = (e.get("text") or "").strip()

                if label == "fig":
                    continue
                if raw_text.startswith("![Figure]("):
                    continue
                if raw_text:
                    text_chunks.append(raw_text)

            page_text = _clean_text_for_output("\n".join(text_chunks))
            page_texts.append({"page": page_no, "text": page_text})

        full_text = _clean_text_for_output(
            "\n\n".join(p["text"] for p in page_texts if p.get("text"))
        )

        return {
            "engine": "dolphin",
            "filename": filename,
            "author": "",
            "pages": len(page_texts),
            "page_texts": page_texts,
            "full_text": full_text,
            "raw": combined_obj,
            "error": None,
        }
