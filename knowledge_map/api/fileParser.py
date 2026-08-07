from fastapi import APIRouter, Request, BackgroundTasks
import asyncio
import json
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field

from algorithm.pdfAna.engine import parse_pdf_bytes
from algorithm.dolphinAdapter.engine import parse_pdf_bytes_with_dolphin
from algorithm.pdfAna.engine import detect_pdf_type
from algorithm.txtAna.engine import parse_txt_bytes
from algorithm.wordAna.engine import parse_docx_bytes, parse_doc_bytes
from values.global_config_variable import DOLPHIN_MODEL_PATH, OCR_PARSE_CALLBACK_URL, ASYNC_MINDMAP_GENERATE_RESULT_URL
router_file_parser = APIRouter()

MAX_CONCURRENCY = 5 # 最大并发数
DOWNLOAD_TIMEOUT = 120 # 下载超时时间
CALLBACK_TIMEOUT = 30 # 回调超时时间

# 结果文件保存目录
RESULT_DIR = Path("results/recognition_json")
RESULT_DIR.mkdir(parents=True, exist_ok=True) # 创建结果文件保存目录

# 文件项
class FileItem(BaseModel):
    fileId: str = Field(..., description="文件唯一标识")
    fileUrl: str = Field(..., description="文件下载地址")

# 解析请求
class ParseRequest(BaseModel):
    files: list[FileItem] = Field(..., description="待解析的文件列表")
    taskId: Any = Field(..., description="任务唯一标识，支持数字或字符串")

# 构建结果文件URL
def _build_result_url(request: Request, result_filename: str) -> str:
    return f"{request.base_url}results/recognition_json/{result_filename}"

# 下载文件
def _download_file_bytes(file_url: str) -> tuple[bytes, str]:
    resp = requests.get(file_url, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    # 获取文件名
    filename = Path(file_url.split("?")[0]).name
    if not filename:
        filename = "unknown_file"
    
    if "." not in filename:
        content_type = str(resp.headers.get("Content-Type") or "").lower()
        if "pdf" in content_type:
            filename = f"{filename}.pdf"
        elif "word" in content_type or "docx" in content_type:
            filename = f"{filename}.docx"
        elif "text" in content_type:
            filename = f"{filename}.txt"

    return resp.content, filename


# 发送解析结果回调
def _post_parse_callback(task_id: Any, results: list[dict[str, Any]]) -> None:
    if not OCR_PARSE_CALLBACK_URL:
        return
    # 设置请求体
    payload = {
        "code": 200,
        "msg": "success",
        "data": {
            "taskId": task_id,
            "results": results,
        },
    }
    # 设置请求头
    headers = {
        "Content-Type": "application/json",
    }

    # 发送请求
    resp = requests.post(
        OCR_PARSE_CALLBACK_URL,
        json=payload,
        headers=headers,
        timeout=CALLBACK_TIMEOUT,
    )
    resp.raise_for_status()

# 提取页面文本
def _extract_page_text(page_item: dict[str, Any]) -> str:
    # 提取清洗后的文本
    clean_text = str(page_item.get("clean_text") or "").strip()
    if clean_text:
        return clean_text
    # 提取文本
    text = str(page_item.get("text") or "").strip()
    if text:
        return text
    # 提取原始文本
    return str(page_item.get("raw_text") or "").strip()

# 构建全文
def _build_full_text(result: dict[str, Any]) -> str:
    # 提取页面文本
    page_texts = result.get("page_texts") or []
    # 如果页面文本不是列表，则返回空字符串
    if not isinstance(page_texts, list):
        return ""
    # 初始化文本块列表
    chunks: list[str] = []
    for page_item in page_texts:
        if not isinstance(page_item, dict):
            continue
        page_text = _extract_page_text(page_item)
        if page_text:
            chunks.append(page_text)

    return "\n\n".join(chunks).strip()

# 构建警告
def _build_warnings(result: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    if result.get("error"):
        warnings.append("parse_error")

    if not str(result.get("author") or "").strip():
        warnings.append("author_missing")

    page_texts = result.get("page_texts") or []
    if not isinstance(page_texts, list) or not page_texts:
        warnings.append("page_texts_empty")
        return warnings

    empty_pages: list[str] = []
    for idx, page_item in enumerate(page_texts, start=1):
        if not isinstance(page_item, dict):
            empty_pages.append(str(idx))
            continue

        if not _extract_page_text(page_item):
            page_no = page_item.get("page", idx)
            empty_pages.append(str(page_no))

    if empty_pages:
        warnings.append(f"empty_pages:{','.join(empty_pages)}")

    return warnings


# 后台任务状态存储字典
TASK_STORE = {}

# 后台任务处理
async def background_parse_worker(task_id, files: list[FileItem], request_base_url: str):
    TASK_STORE[task_id] = {"status": "processing"}
    # 创建信号量
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    # 处理单个文件
    async def handle_one(item: FileItem) -> dict[str, Any]:
        try:
            async with sem:
                # 1. 下载文件
                file_bytes, filename = await asyncio.to_thread(_download_file_bytes, item.fileUrl)
                # 2. 按后缀分流
                ext = Path(filename).suffix.lower()
                if ext == ".txt":
                    result = await asyncio.to_thread(parse_txt_bytes, file_bytes, filename)
                    result["engine"] = "txt"
                    result["pdf_type"] = "text"

                elif ext == ".docx":
                    result = await asyncio.to_thread(parse_docx_bytes, file_bytes, filename)
                    result["engine"] = "docx"
                    result["pdf_type"] = "text"

                elif ext == ".pdf":
                    pdf_type = await asyncio.to_thread(detect_pdf_type, file_bytes)
                    if pdf_type == "text":
                        result = await asyncio.to_thread(parse_pdf_bytes, file_bytes, filename)
                        result["engine"] = "pdf_text"
                        result["pdf_type"] = "text"
                    else:
                        result = await asyncio.to_thread(
                            parse_pdf_bytes_with_dolphin,
                            file_bytes,
                            filename,
                            DOLPHIN_MODEL_PATH,
                            enable_postprocess=True,
                        )
                        result["engine"] = "dolphin"
                        result["pdf_type"] = "scan"

                else:
                    result = {
                        "filename": filename,
                        "author": "",
                        "pages": 0,
                        "page_texts": [],
                        "error": f"不支持的文件后缀: {ext or 'unknown'}",
                    }

                # 3. 统一后处理
                if "pdf_type" not in result:
                    result["pdf_type"] = "text"

                result["full_text"] = _build_full_text(result)
                result["warnings"] = _build_warnings(result)

                # 4. 落盘结果
                result_filename = f"parse_{item.fileId}.json"
                result_path = RESULT_DIR / result_filename
                with open(result_path, "w", encoding="utf-8") as fw:
                    json.dump(result, fw, ensure_ascii=False, indent=2)

                return {
                    "taskId": task_id,
                    "fileId": item.fileId,
                    "pages": result.get("pages", 0),
                    "resultUrl": f"{request_base_url}results/recognition_json/{result_filename}",
                    "success": result.get("error") is None,
                }
        
        except Exception as e:
            import traceback
            print(f"[ERROR] 文件解析失败: fileId={item.fileId}, fileUrl={item.fileUrl}, err={e}")
            traceback.print_exc()
            return {
                "taskId": task_id,
                "fileId": item.fileId,
                "pages": 0,
                "resultUrl": "",
                "success": False,
            }

    results = await asyncio.gather(*[handle_one(item) for item in files])

    TASK_STORE[task_id] = {
        "status": "completed",
        "results": results
    }

    try:
        await asyncio.to_thread(_post_parse_callback, task_id, results)
        print(f"[INFO] callback sent: taskId={task_id}, url={OCR_PARSE_CALLBACK_URL}")
        # print(f"[INFO] callback sent: taskId={task_id}, url={ASYNC_MINDMAP_GENERATE_RESULT_URL}")
    except Exception as e:
        print(f"[ERROR] callback failed: taskId={task_id}, url={OCR_PARSE_CALLBACK_URL}, err={e}")
        # print(f"[ERROR] callback failed: taskId={task_id}, url={ASYNC_MINDMAP_GENERATE_RESULT_URL}, err={e}")



# --- 核心同步处理逻辑 ---
def process_file_sync(file_item: FileItem, request: Request) -> dict[str, Any]:
    """
    同步处理单个文件。
    """
    try:
        # 1. 下载文件 (放在线程池避免阻塞主协程，虽然这里是同步接口)
        # 注意：在 FastAPI 同步接口中，长时间运行的阻塞操作最好放在线程池中，或者降低并发
        file_bytes, filename = _download_file_bytes(file_item.fileUrl)
        
        ext = Path(filename).suffix.lower()
        result = {}

        if ext == ".txt":
            result = parse_txt_bytes(file_bytes, filename)
            result["engine"] = "txt"
            result["pdf_type"] = "text"
        elif ext == ".docx":
            result = parse_docx_bytes(file_bytes, filename)
            result["engine"] = "docx"
            result["pdf_type"] = "text"
        elif ext == ".pdf":
            # 这里可能需要耗时，建议使用 asyncio.to_thread 如果是在 async def 中
            pdf_type = detect_pdf_type(file_bytes)
            if pdf_type == "text":
                result = parse_pdf_bytes(file_bytes, filename)
                result["engine"] = "pdf_text"
                result["pdf_type"] = "text"
            else:
                result = parse_pdf_bytes_with_dolphin(
                    file_bytes,
                    filename,
                    DOLPHIN_MODEL_PATH,
                    enable_postprocess=True,
                )
                result["engine"] = "dolphin"
                result["pdf_type"] = "scan"
        else:
            result = {
                "filename": filename,
                "author": "",
                "pages": 0,
                "page_texts": [],
                "error": f"不支持的文件后缀：{ext or 'unknown'}",
            }
            result["engine"] = "unknown"
            result["pdf_type"] = "unknown"

        if "pdf_type" not in result: result["pdf_type"] = "text"
        
        # 2. 后处理
        result["full_text"] = _build_full_text(result)
        result["warnings"] = _build_warnings(result)
        
        # 3. 落盘
        result_filename = f"parse_{file_item.fileId}.json"
        result_path = RESULT_DIR / result_filename
        with open(result_path, "w", encoding="utf-8") as fw:
            json.dump(result, fw, ensure_ascii=False, indent=2)
            
        return {
            "taskId": file_item.fileId,
            "fileId": file_item.fileId,
            "pages": result.get("pages", 0),
            "resultUrl": _build_result_url(request, result_filename),
            "success": result.get("error") is None,
            "error_detail": result.get("error"), # 增加错误详情方便调试
        }
    
    except Exception as e:
        print(f"[ERROR] 文件解析失败：fileId={file_item.fileId}, err={e}")
        traceback.print_exc()
        return {
            "taskId": file_item.fileId,
            "fileId": file_item.fileId,
        }
        
# 文件解析接口
@router_file_parser.post("/file/parse")
async def file_parse(request: Request, body: ParseRequest, background_tasks: BackgroundTasks):
    task_id = body.taskId
    # 如果任务不存在，则创建任务并启动后台处理
    if task_id not in TASK_STORE:
        TASK_STORE[task_id] = {"status": "pending"}
        # 添加后台任务
        background_tasks.add_task(
            background_parse_worker, # 异步后台处理任务
            task_id=task_id,
            files=body.files,
            request_base_url=str(request.base_url)
        )

    status = TASK_STORE.get(task_id, {}).get("status","pending")

    data = {
        "taskId": task_id,
        "status": status,
    }

    if status == "completed":
        data["results"] = TASK_STORE.get(task_id, {}).get("results", []) 

    return {
        "code": 200,
        "msg": "success",
        "data": data,
    }


@router_file_parser.get("/file/parse/result")
async def get_parse_result(taskId: str):
    for k, v in TASK_STORE.items():
        if str(k) == str(taskId):
            if v["status"] == "completed":
                return {
                    "code": 200,
                    "msg": "success",
                    "data": {
                        "results": v["results"],
                        "taskId": k
                    }
                }
            else:
                return {
                    "code": 200,
                    "msg": "success",
                    "data": {
                        "taskId": k,
                        "status": v["status"]
                    }
                }

    return {"code": 404, "msg": "Task not found"}


