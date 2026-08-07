"""
错误矫正API
"""
### 同步接口
# from flask import Blueprint, request, jsonify, g
# from log.log import logging
# from tools.error_correct.error_correct import error_correct

# # 注册蓝图
# blueprint_error_correct = Blueprint('blueprint_error_correct', __name__)


# # 普通接口--用蓝图注册路由
# @blueprint_error_correct.route("/ekg/errorCorrect", methods=['POST'])
# def main_train():
#     if request.method == 'POST':
#         try:
#             data = request.json
#             logging.info(f"Request data: {str(data)}")

#             entities = data.get("entities", "")
#             relations = data.get("relations", "")

#             # 获取已加载的大模型
#             tokenizer = g.tokenizer
#             model = g.model

#             # 调用主要功能代码
#             prediction_data = error_correct(entities, relations, tokenizer, model, group_size=5)

#             result = prediction_data["data"]

#             # 返回调用成功信息
#             return jsonify({
#                 "code": 200,
#                 "msg": "success",
#                 "data": result
#             })
#         except Exception as e:
#             logging.error(f"Error in hello: {str(e)}")
#             return jsonify({
#                 "code": 500,
#                 "msg": f"Server Error: {str(e)}",
#                 "data": {}
#             })


### 异步接口
from flask import Blueprint, request, jsonify, g
from log.log import logging
from tools.error_correct.error_correct import error_correct
import requests
from value.global_value import ASYNC_ERROR_CORRECT_RESULT_URL     # 回调后端地址
from value.global_value import CHUNK_SIZE_ERROR_CORRECT
# 注册蓝图
blueprint_error_correct = Blueprint('blueprint_error_correct', __name__)


@blueprint_error_correct.route("/ekg/errorCorrect", methods=['POST'])
def start_error_correct_task():
    data = request.json
    logging.info(f"Request data: {data}")
    entities = data.get("entities", "")
    relations = data.get("relations", "")
    ontRelList = data.get("ontRelList", "")
    task_id = data.get("taskId", "")
    graph_type = data.get("graphType", "")

    # 回调 URL（请根据实际情况修改）
    callback_url = ASYNC_ERROR_CORRECT_RESULT_URL

    # 提交任务到后台线程池执行耗时任务
    g.executor.submit(async_error_correct_task, entities, relations, ontRelList, task_id, graph_type, callback_url)

    return jsonify({
        "code": 200,
        "msg": "Event prediction task submitted, processing in background!"
    })


def async_error_correct_task(entities, relations, ontRelList, task_id, graph_type, callback_url):
    # 在后台任务中使用 Flask 应用上下文以获取全局变量
    try:
        # 获取已加载的 tokenizer 和 model（确保在 g 中已初始化）
        tokenizer = g.tokenizer
        model = g.model

        # 调用主要功能代码
        prediction_data = error_correct(entities, relations, ontRelList, tokenizer, model, group_size=CHUNK_SIZE_ERROR_CORRECT)

        result = prediction_data["data"]

        result["taskId"] = task_id
        result["graphType"] = graph_type

        callback_payload = {
            "code": 200,
            "msg": "success",
            "data": result
        }
    except Exception as e:
        logging.error(f"Error in async error correct task: {str(e)}")
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
@blueprint_error_correct.route("/ekg/async_error_correct_result", methods=['POST'])
def error_correct_result():
    result = request.json.get('data')
    logging.info(f"Received error correct result: {result}")
    return jsonify({"message": f"Received error correct result: {result}"})


