"""
若则分析接口
"""
# from flask import Blueprint, request, jsonify, g
# from log.log import logging
# from tools.if_then.if_then import if_then

# # 注册蓝图
# blueprint_if_then = Blueprint('blueprint_if_then', __name__)


# # 普通接口--用蓝图注册路由
# @blueprint_if_then.route("/ekg/ifThen", methods=['POST'])
# def main_train():
#     if request.method == 'POST':
#         try:
#             data = request.json
#             logging.info(f"Request data: {str(data)}")

#             patterns = data.get("patterns", "")
#             files_url = data.get("files", "")

#             # 获取已加载的大模型
#             tokenizer = g.tokenizer
#             model = g.model

#             # 调用主要功能代码
#             prediction_data = if_then(patterns, files_url, model, tokenizer, chunk_size=10)

#             result = prediction_data["data"]

#             # 返回调用成功信息
#             return jsonify({
#                 "code": 200,
#                 "msg": "success",
#                 "data": result
#             })
#         except Exception as e:
#             logging.error(f"Error in if_then: {str(e)}")
#             return jsonify({
#                 "code": 500,
#                 "msg": f"Server Error: {str(e)}",
#                 "data": {}
#             })



from flask import Blueprint, request, jsonify, g
from log.log import logging
from tools.if_then.if_then import if_then
import requests
from value.global_value import ASYNC_THENIF_THEN_RESULT_URL, CHUNK_SIZE_IF_THEN

# 注册蓝图
blueprint_if_then = Blueprint('blueprint_if_then', __name__)


@blueprint_if_then.route("/ekg/ifThen", methods=['POST'])
def start_if_then_task():
    data = request.json
    logging.info(f"Request data: {data}")
    patterns = data.get("patterns", "")
    files_url = data.get("files", "")
    task_id = data.get("taskId", "")

    # 回调 URL（请根据实际情况修改）
    callback_url = ASYNC_THENIF_THEN_RESULT_URL

    # 提交任务到后台线程池执行耗时任务
    g.executor.submit(async_if_then_task, patterns, files_url, task_id, callback_url)

    return jsonify({
        "code": 200,
        "msg": "Event prediction task submitted, processing in background!"
    })


def async_if_then_task(patterns, files_url, task_id, callback_url):
    # 在后台任务中使用 Flask 应用上下文以获取全局变量
    try:
        # 获取已加载的 tokenizer 和 model（确保在 g 中已初始化）
        tokenizer = g.tokenizer
        model = g.model

        # 调用主要功能代码
        prediction_data = if_then(patterns, files_url, model, tokenizer, chunk_size=CHUNK_SIZE_IF_THEN)

        result = prediction_data["data"]

        callback_payload = {
            "code": 200,
            "msg": "success",
            "taskId": task_id,
            "data": result
        }
    except Exception as e:
        logging.error(f"Error in async if_then task: {str(e)}")
        callback_payload = {
            "code": 500,
            "msg": f"Server Error: {str(e)}",
            "data": {}
        }

    # 通过 HTTP POST 请求将任务结果回调到指定 URL
    try:
        requests.post(callback_url, json=callback_payload)
    except Exception as ex:
        logging.error(f"Error calling callback URL: {ex}")


# 模拟接收异步任务回调结果的接口
@blueprint_if_then.route("/ekg/async_if_then_result", methods=['POST'])
def if_then_result():
    result = request.json.get('data')
    logging.info(f"Received if_then result: {result}")
    return jsonify({"message": f"Received if_then result: {result}"})
