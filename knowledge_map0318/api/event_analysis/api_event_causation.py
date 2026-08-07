"""
事件溯因接口
"""
### 同步接口
# from flask import Blueprint, request, jsonify, g
# from log.log import logging
# from tools.event_analysis.event_causation import event_causation

# # 注册蓝图
# blueprint_event_causation = Blueprint('blueprint_event_causation', __name__)


# # 普通接口--用蓝图注册路由
# @blueprint_event_causation.route("/ekg/eventCausation", methods=['POST'])
# def main_train():
#     if request.method == 'POST':
#         try:
#             data = request.json
#             logging.info(f"Request data: {str(data)}")

#             eventGraph = data.get("eventGraph", "")

#             # 获取已加载的大模型
#             tokenizer = g.tokenizer
#             model = g.model

#             # 调用主要功能代码
#             prediction_data = event_causation(eventGraph, tokenizer, model)

#             # 如果prediction_data中的code字段为500，则抛出异常
#             if prediction_data["code"] == 500:
#                 raise Exception(prediction_data["msg"])
#             else:
#                 result = prediction_data["data"]

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
from flask import Blueprint, request, jsonify, g, current_app
from log.log import logging
from tools.event_analysis.event_causation import event_causation
import requests
from value.global_value import ASYNC_EVENT_CASATION_RESULT_URL

# 注册蓝图
blueprint_event_causation = Blueprint('blueprint_event_causation', __name__)


@blueprint_event_causation.route("/ekg/eventCausation", methods=['POST'])
def start_event_causation_task():
    data = request.json
    logging.info(f"Request data: {data}")
    eventGraph = data.get("eventGraph", "")
    # 回调 URL（请根据实际情况修改）
    callback_url = ASYNC_EVENT_CASATION_RESULT_URL

    # 提交任务到后台线程池执行耗时任务
    g.executor.submit(async_event_causation_task, eventGraph, callback_url)

    return jsonify({
        "code": 200,
        "msg": "Event causation task submitted, processing in background!"
    })


def async_event_causation_task(eventGraph, callback_url):
    # 在后台任务中使用 Flask 应用上下文以获取全局变量
        try:
            # 获取全局加载的 tokenizer 和 model（确保在 g 中已初始化）
            tokenizer = g.tokenizer
            model = g.model
            prediction_data = event_causation(eventGraph, tokenizer, model)

            # 若内部调用返回错误，则抛出异常
            if prediction_data["code"] == 500:
                raise Exception(prediction_data["msg"])
            result = prediction_data["data"]
            callback_payload = {
                "code": 200,
                "msg": "success",
                "data": result
            }
        except Exception as e:
            logging.error(f"Error in async event causation task: {str(e)}")
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
@blueprint_event_causation.route("/ekg/async_event_causation_result", methods=['POST'])
def event_causation_result():
    result = request.json.get('data')
    logging.info(f"Received event causation task result: {result}")
    return jsonify({"message": f"Received event causation result: {result}"})