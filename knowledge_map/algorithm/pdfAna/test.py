from flask import Flask, jsonify, request
import time

try:
    import pymupdf
except ImportError:  # pragma: no cover - compatibility fallback
    import fitz as pymupdf


app = Flask(__name__)
app.json.ensure_ascii = False


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/pdf/parse")
def parse_pdf():
    start_time = time.time()  # 记录请求开始时间

    # no file
    if "file" not in request.files:
        return jsonify({"code": 1, "msg": "No file part"}), 400

    files = request.files.getlist("file")
    
    if not files:
        return jsonify({"code": 1, "msg": "No file uploaded"}), 400
    
    page_texts = []
    
    for f in files:
        print(f"接收文件: {f.filename}, 文本类型: {f.content_type}")
        print(f"文件类型: {type(f)}")  

        # not pdf
        if not f.filename.lower().endswith(('.pdf')):

            return jsonify({"code": 2, "msg": f"Invalid file format{f.filename}"}), 400
        try:
            file_bytes = f.read()
            with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
                for i in range(1, pdf.page_count + 1):
                    page = pdf.load_page(i - 1)
                    print(f"正在处理文件 {f.filename} 的 {i} 页")
                    
                    words = page.get_text("words", sort=True) or []
                 
                    column_count = 1
                    left_words = []
                    right_words = []
                
                    rect = page.rect
                    x_mid = rect.x0 + (rect.width / 2.0)
                    for word in words:
                        if float(word[0]) < x_mid:
                            left_words.append(word)
                        else:
                            right_words.append(word)
                    print(f"left_words: {len(left_words)}, right_words: {len(right_words)}, words: {len(words)}")
                    if len(left_words) > len(words) * 0.3 and len(right_words) > len(words) * 0.3:
                        column_count = 2
                    
                    print(f"页面 {i}的列数: {column_count}")
                    
                    if column_count == 1:
                        page_text = page.get_text("text", sort=True) or ""
                    else:
                        left_column_bbox = pymupdf.Rect(rect.x0, rect.y0, x_mid, rect.y1)
                        right_column_bbox = pymupdf.Rect(x_mid, rect.y0, rect.x1, rect.y1)
                        print(f"left_column_bbox:{left_column_bbox},right_column_bbox:{right_column_bbox}")
                        
                        try:
                            left_text = page.get_textbox(left_column_bbox) or ""
                            right_text = page.get_textbox(right_column_bbox) or ""
                            page_text = left_text + "\n" + right_text
                        except Exception as e:
                            print(f"从页面{i}提取文本时出错：{e}")
                            page_text = ""

                    page_texts.append({"file": f.filename, "page": i, "text": page_text})
            print(f"完成解析文件：{f.filename}")
        except Exception as e:
            return jsonify({"code": 3, "msg": "parse failed", "error": str(e)}), 500

    full_text = "\n".join(p.get("text", "") for p in page_texts).strip()

    end_time = time.time()  # 记录请求结束时间
    processing_time = end_time - start_time  # 计算总时间

    return jsonify({
        "code": 0,
        "msg": "success",
        "data": {
            "total_files": len(files),
            "pages": len(page_texts),
            "full_text": full_text,
            "page_texts": page_texts,
            "processing_time": round(processing_time, 2)  # 显示处理时间，保留两位小数
        }
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=True)
