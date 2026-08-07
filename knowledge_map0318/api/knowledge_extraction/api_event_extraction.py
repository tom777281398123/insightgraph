"""
事件抽取接口
"""
from flask import Blueprint, jsonify, request, g
from log.log import logging
from tools.event_extraction.event_extraction import start_events_extraction
from tools.utils import download_file, extract_text, process_next_task, add_task_to_queue
from value.global_value import LOCAL_TEST, EVENT_DOWNLOAD_PATH, BACKEND_ONTOLOGY_URL, ASYNC_EE_RESULT_URL, DELETE_TEMP_FILE
import datetime
import requests
import json
import os
import shutil

# 注册蓝图
blueprint_event_extraction = Blueprint('blueprint_event_extraction', __name__)

# 事件抽取接口函数
def event_extraction(files, ontologyAllId, extractionDataRecordId, callback_url):
    """
    files          :抽取文件下载地址列表
    ontologyAllId  :本体表ID
    extractionId   :抽取任务ID
    callback_url   :任务完成后传给后端的地址
    """
    try:
        # 1 下载文件 & 获取文件中的文本
        file_texts = {}
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(EVENT_DOWNLOAD_PATH, current_time)   # 下载地址 + 当前时间
        for file in files:
            print(file)           
            file_path = download_file(file, save_path)    # 将文件下载到save_path目录下，返回单一文件的路径
            file_texts[file] = extract_text(file_path)    # 抽取文件的文本内容；file是http链接，file_path是本地文件路径
                
        if not file_texts:   # 获取的文本内容为空
            result = {"code": 400, "message": "未找到任何文本内容。", "data": {"extractionId": extractionDataRecordId}}
            # 本次任务执行失败，检查任务进程是否有其他任务
            # process_next_task()
            requests.post(callback_url, json=result)
            return
            
        # 2 查询事件本体列表 转换成需要的格式
        if LOCAL_TEST==1:
            response = requests.get(f"{BACKEND_ONTOLOGY_URL}/{ontologyAllId}")     # 本地模拟后端调用
        else:
            response = requests.get(f"{BACKEND_ONTOLOGY_URL}?ontologyAllId={ontologyAllId}")   # 前后端联调用

        if response.status_code == 200:
            # 获取返回的 JSON 数据
            data = response.json()  # 将响应内容转换为 JSON 格式
            logging.info(f"获取到的本体数据为{data}")
            # print(data)  # 打印获取到的数据
        else:
            result = {"code": 500, "message": "请求本体失败。", "data": {"extractionId": extractionDataRecordId}}
            # 本次任务执行失败，检查任务进程是否有其他任务
            # process_next_task()
            requests.post(callback_url, json=result)
            return

        ontologyClList = data['ontologyClList']
        ontologyReList = data['ontologyReList']
        ontologyDict = {}  # 事件本体字典，key为本体名称，value为本体id
        for item in ontologyClList:   
            ontologyDict[item["ontologyName"]] = item["ontologyId"]
        relationSet = set()   # 事件关系集合
        for item in ontologyReList:
            relationSet.add(item["relationName"])
        
        # 3 调用事件抽取函数
        result = start_events_extraction(file_texts=file_texts,     # 待抽取文本
                                                        model=g.model,            # 大模型参数
                                                        tokenizer=g.tokenizer,      # 大模型参数
                                                        extractionDataRecordId=extractionDataRecordId,   # 任务id
                                                        ontologyDict=ontologyDict,     # 事件本体字典
                                                        ontologyReList=(list(relationSet))      # 事件关系集合
                                                        )
        logging.info(f"事件抽取接口的返回数据为{result}")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"tools/event_extraction/event_extraction_result_{current_time}.json", 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        # 执行完成后检查队列中是否有待执行任务
        # process_next_task()
        requests.post(callback_url, json=result)

        ### 删除暂存文件夹
        if DELETE_TEMP_FILE == 1:
            shutil.rmtree(save_path)

    except Exception as e:
        result = {"code": 500, "message": f"请求失败，错误信息：{e}", "data": {"extractionId": extractionDataRecordId}}
        # 本次任务执行失败，检查任务进程是否有其他任务
        # process_next_task()
        requests.post(callback_url, json=result)

        ### 删除暂存文件夹
        if DELETE_TEMP_FILE == 1:
            shutil.rmtree(save_path)

# 异步接口--处理耗时较长的任务
@blueprint_event_extraction.route('/ekg/eventExtraction', methods=['POST'])
def start_task():
    # 获取请求参数
    data = request.json
    files = data['files']    # 文件下载列表
    ontologyAllId = data['ontologyAllId']   # 本体表ID
    extractionDataRecordId = data['extractionDataRecordId']   # 抽取任务ID
    logging.info(f"接收到请求参数：{files}, {ontologyAllId}, {extractionDataRecordId}")

    # 使用executor异步处理耗时较长的任务
    g.executor.submit(event_extraction, files, ontologyAllId, extractionDataRecordId, ASYNC_EE_RESULT_URL)

    # # 添加任务到队列
    # add_task_to_queue(lambda task_id: event_extraction(task_id, files, ontologyAllId, extractionDataRecordId, ASYNC_EE_RESULT_URL), extractionDataRecordId)
    # process_next_task()
    
    response = {
        "code": 200,
        "msg": f"事件抽取任务(ID:{extractionDataRecordId})已提交，正在后台处理中！"
    }
    return jsonify(response)
    
# 模拟异步接口完成后回调(模拟后端)
@blueprint_event_extraction.route('/async_event_extraction_task_result', methods=['POST'])
def task_result():
    result = request.json.get('data')
    print(f"收到任务结果: {result}")
    return jsonify({"message": f"收到任务结果: {result}"})

# 模拟后端返回本体(模拟后端)
@blueprint_event_extraction.route('/ontology/getOntologyData/<ontologyAllId>', methods=['GET'])
def getOntologyData(ontologyAllId):
    print("模拟后端获取到本体Id，查询本体的过程")
    # 读取 JSON 文件
    with open(os.path.join(os.getcwd(), "./tools/event_extraction/ontology.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)

    return jsonify(data)

