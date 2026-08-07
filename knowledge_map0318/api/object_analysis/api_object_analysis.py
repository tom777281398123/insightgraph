# from flask import Blueprint, request, jsonify, g
# from log.log import logging
# from tools.object_analysis.object_analysis import object_analysis

# # 注册蓝图
# blueprint_object_analysis = Blueprint('blueprint_object_analysis', __name__)


# # 普通接口--用蓝图注册路由
# @blueprint_object_analysis.route("/ekg/objectAnalysis", methods=['POST'])
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
#             prediction_data = object_analysis(entities, relations, tokenizer, model)

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


from flask import Blueprint, request, jsonify, g
from log.log import logging
from tools.object_analysis.object_analysis import object_analysis
import requests
from value.global_value import BACKEND_URL, CHUNK_SIZE

# 注册蓝图
blueprint_object_analysis = Blueprint('blueprint_object_analysis', __name__)


@blueprint_object_analysis.route("/ekg/objectAnalysis", methods=['POST'])
def start_event_prediction_task():
    data = request.json
    logging.info(f"Request data: {data}")
    entities = data.get("entities", "")
    relations = data.get("relations", "")

    # 回调 URL（请根据实际情况修改）
    callback_url = BACKEND_URL + "/ekg/async_object_analysis_result"

    # 提交任务到后台线程池执行耗时任务
    g.executor.submit(async_object_analysis_task, entities, relations, callback_url)

    return jsonify({
        "code": 200,
        "msg": "Event prediction task submitted, processing in background!"
    })


def async_object_analysis_task(entities, relations, callback_url):
    # 在后台任务中使用 Flask 应用上下文以获取全局变量
    try:
        # 获取已加载的 tokenizer 和 model（确保在 g 中已初始化）
        tokenizer = g.tokenizer
        model = g.model

        # 调用主要功能代码
        prediction_data = object_analysis(entities, relations, tokenizer, model, chunk_size=CHUNK_SIZE)

        # 如果 prediction_data 中的 code 字段为 500，则抛出异常
        if prediction_data["code"] == 500:
            raise Exception(prediction_data["msg"])
        result = prediction_data["data"]

        callback_payload = {
            "code": 200,
            "msg": "success",
            "data": result
        }
    except Exception as e:
        logging.error(f"Error in async event prediction task: {str(e)}")
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
@blueprint_object_analysis.route("/ekg/async_object_analysis_result", methods=['POST'])
def object_analysis_result():
    result = request.json.get('data')
    logging.info(f"Received object analysis result: {result}")
    return jsonify({"message": f"Received object analysis result: {result}"})
