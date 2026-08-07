'''
模型训练接口
'''
import os
import shutil
import sys
import uuid

import torch

from tools.train_model.get_system_memory import get_system_memory

sys.path.append("./")

import json
from flask import Blueprint, request, jsonify
import subprocess
from datetime import datetime

from api.train_model.celery_config import celery
from tools.utils import download_file
from celery.result import AsyncResult
from log.log import logging
import requests

from value.global_value import HTTP_IP_PORT, LOG_RESULT_PATH, PYTHON_WHICH, URL_DOWNLOAD_FILE_TRAIN, TRAIN_DATA_PATH, ORIGIN_TRAIN_DATA_PATH

MODEL_PATH = 'src/example/ner/standard/'
CHECKPOINTS_PATH = MODEL_PATH + 'checkpoints/'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 注册蓝图
blueprint_main_train = Blueprint('blueprint_main_train', __name__)

# 读入参数并校验参数
def validate_params(data):
    # 训练参数
    train_batch_size = data.get("train_param", {}).get("train_batch_size", 46)
    eval_batch_size = data.get("train_param", {}).get("eval_batch_size", 58)
    num_train_epochs = data.get("train_param", {}).get("num_train_epochs", 66)

    # 超参数
    lr_step = data.get("hyper_param", {}).get("lr_step", 5)
    lr_gamma = data.get("hyper_param", {}).get("lr_gamma", 0.8)
    beta1 = data.get("hyper_param", {}).get("beta1", 0.9)

    # 模型参数
    num_layers = data.get("model_param", {}).get("num_layers", 2)
    drop_out = data.get("model_param", {}).get("drop_out", 0.5)

    # 参数逻辑检查
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be a positive integer.")
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be a positive integer.")
    
    # 限制训练批大小
    max_train_batch_size = 64  # 考虑服务器显存限制
    if train_batch_size > max_train_batch_size:
        raise ValueError(f"train_batch_size must not exceed {max_train_batch_size}.")
    
    # 限制训练轮次
    max_num_train_epochs = 10  # 设置最大训练轮次
    if num_train_epochs < 1 or num_train_epochs > max_num_train_epochs:
        raise ValueError(f"num_train_epochs must be between 1 and {max_num_train_epochs}.")
    
    if num_train_epochs < 1:
        raise ValueError("num_train_epochs must be at least 1.")
    if lr_step < 1:
        raise ValueError("lr_step must be at least 1.")
    if not (0 < lr_gamma <= 1):
        raise ValueError("lr_gamma must be between 0 and 1.")
    if not (0 < beta1 < 1):
        raise ValueError("beta1 must be between 0 and 1.")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if not (0 <= drop_out < 1):
        raise ValueError("drop_out must be between 0 and 1.")

    return {
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "lr_step": lr_step,
        "lr_gamma": lr_gamma,
        "beta1": beta1,
        "num_layers": num_layers,
        "drop_out": drop_out
    }


# 异步任务
@celery.task
def async_main_train_lstm(data):
    try:
        unique_id = uuid.uuid4().hex[:8]  # 使用UUID的前8个字符作为唯一标识符
        data_dir = os.path.join(TRAIN_DATA_PATH, f"lstm_{unique_id}")

        # 下载传入的训练文件(可能有多个),将传入的文件存放到指定的训练目录中
        files = data.get("files", [])
        for file in files:
            download_file(file, data_dir)

        # 如果文件夹下不具备训练所需的文件，就使用原数据进行训练
        required_files = ['train.txt', 'test.txt', 'valid.txt']
        for f in required_files:
            f_path = os.path.join(data_dir, f)
            if not os.path.isfile(f_path):
                data_dir = ORIGIN_TRAIN_DATA_PATH

        # 构建脚本的完整路径
        script_path = os.path.join(os.getcwd() , MODEL_PATH, 'run_lstmcrf_main.py')
        logging.info(f"训练脚本路径：{script_path}")

        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), f"{CHECKPOINTS_PATH}", f"lstmcrf_{unique_id}_{time_now}")

        # 验证参数
        params = validate_params(data)

        hydra_args = [
            f'train_batch_size={params["train_batch_size"]}',
            f'eval_batch_size={params["eval_batch_size"]}',
            f'num_train_epochs={params["num_train_epochs"]}',
            f'lr_step={params["lr_step"]}',
            f'lr_gamma={params["lr_gamma"]}',
            f'beta1={params["beta1"]}',
            f'num_layers={params["num_layers"]}',
            f'drop_out={params["drop_out"]}',
            f'data_dir={data_dir}',
            f'output_dir={output_dir}'
        ]
        
        # 构建完整的命令
        command = [PYTHON_WHICH, script_path] + hydra_args

        # 执行 Python 脚本
        result = subprocess.run(command, 
                                capture_output=True, 
                                text=True, 
                                check=True)

        # 记录训练结果
        logging.info(f"Training completed. Output: {result.stdout}")
        try:
            result_ = os.path.join(output_dir, "classification_report.json")
            log_ = os.path.join(output_dir, "run_lstmcrf_main.log")

            # todo: 和后端对接，使用什么符号来标识训练好的模型
            shutil.copyfile(result_, os.path.join(LOG_RESULT_PATH, f'classification_report_{unique_id}.json'))
            shutil.copyfile(log_, os.path.join(LOG_RESULT_PATH, f'run_lstmcrf_main_{unique_id}.log'))

            # 读取log文件结果，补充精确率、召回率、F1值、完成时间
            with open(result_, 'r', encoding='utf-8') as f:
                log_dict = json.loads(f.read())
            result_dict = log_dict.get("result_dict", {}).get("weighted avg", {})

            result_json = {
                "data":{
                    "output": "You can check the log file for detailed output!",
                    "resultJsonFile": os.path.join(HTTP_IP_PORT, URL_DOWNLOAD_FILE_TRAIN, f'classification_report_{unique_id}.json'),
                    "logFile": os.path.join(HTTP_IP_PORT, URL_DOWNLOAD_FILE_TRAIN, f'run_lstmcrf_main_{unique_id}.log'),
                    "precision": result_dict.get("precision", 0.0),
                    "recall": result_dict.get("recall", 0.0),
                    "f1-score": result_dict.get("f1-score", 0.0),
                    # "complete_time": result_dict.get("complete_time", datetime.now().strftime("%Y%m%d_%H%M%S"))
                    "complete_time": result_dict.get("complete_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                }
            }
            logging.info(f"Files ( result | log ) copy successful. ")
            return result_json
        except FileNotFoundError:
            logging.error(f"File { result_ } and { log_ } not found, copy failed.")
            return {"msg": "error", "message": str(e)}
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed. Error: {e.stderr}")
        return {"msg": "error", "message": e.stderr}
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return {"msg": "error", "message": str(e)}

# 用蓝图注册路由
@blueprint_main_train.route("/train/main_train", methods=['POST'])
def main_train():
    if request.method == 'POST':
        try:
            data = request.json
            logging.info(f"Request data: {str(data)}")
            
            # Check if a GPU is available
            if not torch.cuda.is_available():
                logging.warning("No GPU found. Training will be performed on CPU.")
                return jsonify({
                    "code": 400,
                    "msg": "No GPU available. Please ensure a GPU is available for training.",
                    "data": {}
                })

            # 检查显存大小
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                free_memory_gb = free_memory / (1024 ** 3)  # 转换为GB
                if free_memory_gb < 8:
                    logging.warning("Insufficient GPU memory. At least 8GB of free GPU memory is required.")
                    return jsonify({
                        "code": 400,
                        "msg": "Insufficient GPU memory. At least 8GB of free GPU memory is required.",
                        "data": {}
                    })
                
            # 检查内存信息
            memory_info = get_system_memory()
            logging.info(f"Free system memory: {memory_info['available_memory_gb']} GB")
            if memory_info["available_memory_gb"] < 10:
                logging.warning("Insufficient system memory. At least 10GB of free memory is required.")
                return jsonify({
                    "code": 400,
                    "msg": "Insufficient system memory. At least 10GB of free memory is required.",
                    "data": {}
                })

            if data["model_name"] == "lstm":
                # 异步执行训练任务
                task = async_main_train_lstm.delay(data)
                return jsonify({
                    "code": 202, 
                    "msg": "Task accepted", 
                    "data": {"task_id": task.id}
                })
            else:
                return jsonify({
                    "code": 500, 
                    "msg": f"Training for {data['model_name']} model is not supported!", 
                    "data": {}
                })
        except Exception as e:
            logging.error(f"Error in main_train: {str(e)}")
            return jsonify({
                "code": 500, 
                "msg": f"Server Error: {str(e)}",
                "data": {}
            })

# 添加一个新路由来检查任务状态
@blueprint_main_train.route("/train/status", methods=['GET'])
def check_task_status():
    task_id = request.args.get('task_id')  # 从查询参数获取 task_id
    model_name = request.args.get('model_name')  # 从查询参数获取 model_name，默认为 'lstm'

    if model_name=='lstm':
        task = async_main_train_lstm.AsyncResult(task_id)
    
        if task.state == 'PENDING':
            response = {
                'code': 300,
                'state': task.state,
                'msg': {
                    "data": {
                        "output": 'Task is waiting for execution!',
                        "logFile": "",
                        "resultJsonFile": ""
                    }
                }
            }
        elif task.state != 'FAILURE':
            response = {
                'code': 200,
                'state': task.state,
                'msg': task.info
            }
        else:
            response = {
                'code': 500,
                'state': task.state,
                'msg': {
                    "data": {
                        "output": 'Task failed',
                        "logFile": "",
                        "resultJsonFile": ""
                    }
                }
            }
        return jsonify(response)
    else:
        response = {
            'code': 500,
            'state': "ERROR",
            'msg': {
                "data": {
                    "output": 'This model is not currently supported!',
                    "logFile": "",
                    "resultJsonFile": ""
                }
            }
        }
        return jsonify(response)
    

# 停止任务
@blueprint_main_train.route('/stop/<task_id>', methods=['POST'])
def stop_task(task_id):
    task_result = AsyncResult(task_id)
    if task_result:
        task_result.revoke(terminate=True)  # 停止并终止任务
        return {"message": "Task is being stopped."}
    return {"message": "Task not found."}




