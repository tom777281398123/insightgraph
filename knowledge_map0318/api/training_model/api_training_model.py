'''
模型训练接口
'''
### 同步接口
# from flask import Blueprint, request, jsonify, g
# from log.log import logging
# from tools.trainning_model.main import run_train
#
# # 注册蓝图
# blueprint_training_model = Blueprint('blueprint_training_model', __name__)
#
#
# # 普通接口--用蓝图注册路由
# @blueprint_training_model.route("/ekg/train", methods=['POST'])
# def main_train():
#     if request.method == 'POST':
#         try:
#             data = request.json
#             logging.info(f"Request data: {str(data)}")
#
#             files = data.get("files", "")
#             train_param = data.get("train_param", "")
#
#             # 调用主要功能代码
#             prediction_data = run_train(files, num_epochs=train_param["num_train_epochs"], learning_rate=0.002,
#                                         batch_size=train_param["train_batch_size"])
#
#             # 如果prediction_data中的code字段为500，则抛出异常
#             if prediction_data["code"] == 500:
#                 raise Exception(prediction_data["msg"])
#             else:
#                 result = prediction_data["data"]
#
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
import os
from datetime import datetime
from urllib.parse import urljoin

from flask import Blueprint, request, jsonify, g
from log.log import logging
from tools.trainning_model.main import train_ner, train_rel, train
import requests
from value.global_value import LOCAL_TEST, BACKEND_URL, HTTP_IP_PORT, FILE_SERVIXE_URL, ASYNV_MODEL_TRAIN_RESULT_URL, DATASET_FLODER_PATH
from tools.utils import download_file
from urllib.parse import unquote
# 注册蓝图
blueprint_training_model = Blueprint('blueprint_training_model', __name__)


@blueprint_training_model.route("/ekg/train", methods=['POST'])
def start_event_prediction_task():
    data = request.json
    logging.info(f"Request data: {data}")
    ### 模型训练中共有五类参数
    # model_name: 模型名称
    # files: 一个文件，然后我们自己划分训练集·测试集
    # train_param: 训练参数
    # hyper_param: 超参数
    # model_param: 模型参数
    model_name = data.get("model_name", "")   # 模型名称
    model_id = data.get("model_id", "")     # 模型id
    files = data.get("files", "")             # 训练集·测试集文件
    train_param = data.get("train_param", "") # 训练参数
    hyper_param = data.get("hyper_param", "") # 超参数
    model_param = data.get("model_param", "") # 模型参数

    # 检查模型名称和文件内容是否为空
    if not model_name:
        return jsonify({"code": 500, "msg": "Model name cannot be empty!!!"})
    if not files:
        return jsonify({"code": 500, "msg": "Files content cannot be empty!!!"})


    # 回调 URL（请根据实际情况修改）
    callback_url = ASYNV_MODEL_TRAIN_RESULT_URL

    # 提交任务到后台线程池执行耗时任务
    g.executor.submit(async_training_model_task, model_name, model_id, files, train_param, hyper_param, model_param, callback_url)

    return jsonify({
        "code": 200,
        "msg": "Train model task submitted, processing in background!"
    })


def async_training_model_task(model_name, model_id, files, train_param, hyper_param, model_param, callback_url):
    # 在后台任务中使用 Flask 应用上下文以获取全局变量
    try:
        ### 1、下载训练文件
        if LOCAL_TEST==1:   # 本地调试
            timestamp = datetime.now().strftime("%Y-%m-%d")   # 获取当前时间戳（文件暂时保存路径）------本地调试用
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")   # 获取当前时间戳（文件暂时保存路径）
        dataset_folder = os.path.join(DATASET_FLODER_PATH, timestamp)    # 训练数据集文件下载路径
        # 如果文件路径不存在则新建路径
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        # 下载训练用文件
        if LOCAL_TEST!=1:    
            download_file(files, dataset_folder)
        files = unquote(files)
        print(files)
        # 获取value链接中的文件名
        file_name = os.path.basename(files)
        dataset_file_path = os.path.join(dataset_folder, file_name)    # 训练数据集文件路径
        print(dataset_file_path)

        ### 2、训练模型
        prediction_data = train(dataset_file_path, train_param, hyper_param, model_param, timestamp, model_id)

        # 如果 prediction_data 中的 code 字段为 500，则抛出异常
        if prediction_data["code"] == 500:
            raise Exception(prediction_data["msg"])
        result = prediction_data["data"]

        ner_log_file_name = result["ner_logFile"]
        rel_log_file_name = result["rel_logFile"]
        result_file_name = result["resultJsonFile"]

        precision = result["precision"]
        recall = result["recall"]
        f1_score = result["f1-score"]
        complete_time = result["timestamp"]
        # model_id = result["model_id"]    # id从自己生成改为从后端接收
        model_path = result["model_path"]

        ### 文件下载地址拼接  HTTP_IP_PORT + FILE_SERVIXE_URL + FILE_SERVIXE_PATH（"data/downloads"）后面的路径
        base_url = HTTP_IP_PORT+FILE_SERVIXE_URL
        # print(base_url)
        result_path = f"training_model/result_file/{timestamp}/"

        base_result_url = os.path.join(base_url, result_path)

        result_download_link = os.path.join(base_result_url, result_file_name)
        ner_log_download_link = os.path.join(base_result_url, ner_log_file_name)
        rel_log_download_link = os.path.join(base_result_url, rel_log_file_name)

        print(f"result_download_link: {result_download_link}")
        print(f"ner_log_download_link: {ner_log_download_link}")
        print(f"rel_log_download_link: {rel_log_download_link}")

        result = {
            "output": "You can check the log file for detailed output!",
            "logFile": ner_log_download_link,
            "resultJsonFile": result_download_link,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "complete_time": complete_time,
            "model_id": model_id,
            "model_path": model_path
        }
        print(f"result: {result}")

        callback_payload = {
            "code": 200,
            "msg": "success",
            "data": result
        }
    except Exception as e:
        logging.error(f"Error in async training model task: {str(e)}")
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
@blueprint_training_model.route("/ekg/async_training_model_result", methods=['POST'])
def object_analysis_result():
    result = request.json.get('data')
    logging.info(f"Received training model result: {result}")
    return jsonify({"message": f"Received training model result: {result}"})


