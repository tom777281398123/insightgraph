'''
Author: LXY
Date: 2024-10-30 09:22:20
Description: 
FilePath: /kg_algorithm/api/test/api_hello.py
'''
from flask import Blueprint, request, jsonify, g
from log.log import logging
from tools.test.hello import hello
from tools.event_extraction.event_extraction import call_local_llm
import time
import requests
import os

# 注册蓝图
blueprint_hello = Blueprint('blueprint_hello', __name__)

# 普通接口--用蓝图注册路由
@blueprint_hello.route("/hello", methods=['POST'])
def main_train():
    if request.method == 'POST':
        try:
            data = request.json
            logging.info(f"Request data: {str(data)}")

            name = data.get("name", "")
            
            # 调用主要功能代码
            welcome_msg = hello(name)

            # 获取已加载的大模型
            tokenizer = g.tokenizer
            model = g.model

            messages = [{"role": "user", "content": "请告诉我你的名字"}]

            # 测试使用大模型获得返回结果
            call_local_llm(messages, tokenizer, model)

            # 返回调用成功信息
            return jsonify({
                "code": 200, 
                "msg": "success", 
                "data": {"welcome_msg": welcome_msg}
            })
        except Exception as e:
            logging.error(f"Error in hello: {str(e)}")
            return jsonify({
                "code": 500, 
                "msg": f"Server Error: {str(e)}",
                "data": {}
            })

def long_running_task(data, callback_url):
    time.sleep(5)  # 模拟长时间运行的任务
    result = "任务完成！"
    
    # 调用回调 URL 将结果发送回前端
    try:
        requests.post(callback_url, json={'result': result})
    except requests.exceptions.RequestException as e:
        print(f"回调失败: {e}")

# 异步接口--处理耗时较长的任务
@blueprint_hello.route('/async_task', methods=['POST'])
def start_task():
    data = request.json
    print("handle_event_clustering_async接收到的请求参数如下：\n", data)
    # 将处理结果回调到这个 URL
    callback_url = "http://localhost:6004/async_task_result"  # 需要根据实际情况修改

    # 使用executor异步处理耗时较长的任务
    g.executor.submit(long_running_task, data, callback_url)

    response = {
        "code": 200,
        "msg": "XX任务已提交，正在后台处理中！"
    }
    return jsonify(response)

# 模拟异步接口完成后回调(模拟后端)
@blueprint_hello.route('/async_task_result', methods=['POST'])
def task_result():
    result = request.json.get('result')
    print("收到任务结果: {result}")
    return jsonify({"message": f"收到任务结果: {result}"})
