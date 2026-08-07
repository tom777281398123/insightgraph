'''
知识融合接口
'''
import os
import sys
import copy 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
import zipfile
import shutil
import pandas as pd

from flask import Blueprint, request, jsonify, g
import threading
from log.log import logging
from tools.graph_fusion.knowledge_fusion import knowledgeFusion
from value.global_value import URL_KNOWLEDGE_FUSION_RAW, URL_KNOWLEDGE_FUSION_TEMP, URL_DOWNLOAD_ZIP, HTTP_IP_PORT, URL_UPLOAD_ZIP, ASYNC_FUSION_RESULT_URL, DELETE_TEMP_FILE
import requests
from zipfile import ZipFile, ZipInfo
from tools.graph_fusion.fusion_utils import merge_event_bk

dir_path = os.path.dirname(os.path.abspath(__file__))
# 注册一个
blueprint_KnowledgeFusion = Blueprint('fusion', __name__)

# 解压并读取excel表
def extract_and_read_excel(zip_file_path, extract_to='uploads'):
    '''
    功能：解压图谱zip文件
    输入： zip_file_path  --zip压缩包路径
           extract_to  --解压后文件存放目录
    输出：excel表存储的图谱内容
    '''
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    logging.info(extract_to)
    # 解压缩文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(extract_to)
    # 压缩包第一级路径
    last_dir_no_ext = os.path.basename(copy.copy(zip_file_path))
    while '.' in last_dir_no_ext:
        last_dir_no_ext = os.path.splitext(last_dir_no_ext)[0]
    logging.info(last_dir_no_ext)
    zip_first_path = os.path.join(extract_to, last_dir_no_ext)
    logging.info(zip_first_path)
    logging.info(os.listdir(zip_first_path))  # 注意其中有两层文件夹
    # 查找解压后的Excel文件
    excel_excel_file = [f for f in os.listdir(zip_first_path) if f.endswith('.xlsx') or f.endswith('.xls')]
    # 确保读取的excel文件只有一个
    if len(excel_excel_file) != 1:
        # shutil.rmtree(extract_to)  # 清理提取的目录
        raise ValueError(f"发现图谱压缩包中包含 {len(excel_excel_file)} 个Excel文件，期望只有一个。请检查并重新输入！")
    # 读取Excel文件
    excel_path = os.path.join(zip_first_path, excel_excel_file[0])

    excels = pd.ExcelFile(excel_path)
    # logging.info(excels.sheet_names)
    if '实体表' not in excels.sheet_names or '实体关系表' not in excels.sheet_names:
        raise ValueError(f"zip压缩包中的图谱excel表中不包含“实体表”或“实体关系表”，请检查名称是否与模板严格一致！！！")
    df_entities = pd.read_excel(excel_path, sheet_name='实体表')  # 使用pandas读取Excel文件
    df_relation = pd.read_excel(excel_path, sheet_name='实体关系表')
    logging.info(f"内容来自 {excel_excel_file}：")
    logging.info(df_entities)  # 打印DataFrame内容
    logging.info(df_relation)  
    
    # # 删除提取的目录及其内容
    # shutil.rmtree(extract_to)
    
    return df_entities, df_relation

# 直接读取excel文件
def read_excel(excel_path):
    excels = pd.ExcelFile(excel_path)
    logging.info(excels.sheet_names)
    if '实体表' not in excels.sheet_names or '实体关系表' not in excels.sheet_names:
        raise ValueError(f"zip压缩包中的图谱excel表中不包含“实体表”或“实体关系表”，请检查名称是否与模板严格一致！！！")
    df_entities = pd.read_excel(excel_path, sheet_name='实体表')  # 使用pandas读取Excel文件
    df_relation = pd.read_excel(excel_path, sheet_name='实体关系表')
    logging.info(f"内容来自 {excel_path}：")
    logging.info(df_entities)  # 打印DataFrame内容
    logging.info(df_relation)  
    
    # # 删除提取的目录及其内容
    # shutil.rmtree(extract_to)
    return df_entities, df_relation


def split_knowledge_and_event(extraction_data):
    knowledge_id = []
    knowledge_entities = []
    event_entities = []
    knowledge_relations = []
    event_relations = []

    for entity in extraction_data.get("entities"):
        entityType = entity.get("entityType")
        if entityType == "知识":
            knowledge_entities.append(entity)
            knowledge_id.append(entity.get("entityId"))
        elif entityType == "事件":
            event_entities.append(entity)
    
    for relation in extraction_data.get("relations"):
        headEntityId = relation.get("headEntityId")
        # 在知识实体列表中，则为知识
        if headEntityId in knowledge_id:
            knowledge_relations.append(relation)
        else:
            event_relations.append(relation)

    return {"entities": knowledge_entities, "relations": knowledge_relations}, {"entities": event_entities, "relations": event_relations}

# 异步执行知识融合
def async_knowledgeFusion(extraction_data, df_entities, df_relation, fusion_id, callback_url):
    # extraction_data:抽取的内容，包含实体和关系，需要融合
    # df_entities:实体表
    # df_relation:关系表
    # fusion_id:知识融合任务id
    # callback_url:回调地址
    logging.info("异步知识融合任务开始执行......")

    try:

        # 新增1：将数据表extraction_data拆分为两个
        has_knowledge = False
        has_event = False
        entities = extraction_data.get("entities")
        for e in entities:
            entityType = e.get("entityType")
            if entityType=="知识":
                has_knowledge = True
            if entityType == "事件":
                has_event = True

        # 同时存在事件与知识，调用新的逻辑
        if has_knowledge and has_event:
            extraction_knowledge_data, extraction_event_data = split_knowledge_and_event(extraction_data)

            # 融合知识
            k_three_list_data, k_df_fusion_entities, k_df_fusion_relations = knowledgeFusion(extraction_knowledge_data, df_entities, df_relation)
            print("知识融合完毕")
            # 融合事件
            e_three_list_data, e_df_fusion_entities, e_df_fusion_relations = knowledgeFusion(extraction_event_data, df_entities, df_relation)
            print("事件融合完毕")
            # 融合两者
            df_fusion_entities, df_fusion_relations = merge_event_bk(e_df_fusion_entities, k_df_fusion_entities, e_df_fusion_relations, k_df_fusion_relations)
            print("混合融合完毕")
            # 融合冲突结果
            three_list_data = {
                "directSave": k_three_list_data.get("directSave") + e_three_list_data.get("directSave"),
                "fusionData": k_three_list_data.get("fusionData") + e_three_list_data.get("fusionData"),
                "sameData": k_three_list_data.get("sameData") + e_three_list_data.get("sameData")
            }

        else:
            # 知识融合过程
            # three_list_data: 融合后传给前端的三个列表
            # df_fusion_entities: 融合后的实体表
            # df_fusion_relations: 融合后的关系表
            three_list_data, df_fusion_entities, df_fusion_relations = knowledgeFusion(extraction_data, df_entities, df_relation)
    except Exception as e:
        result = {"code": 500, "message": f"请求失败，错误信息：{e}", "data": ""}
        print(result)

    ### 将融合后的实体表和关系表导入压缩包
    # 定义文件和文件夹名称
    new_folder = URL_KNOWLEDGE_FUSION_TEMP   # 知识融合后，生成图谱的暂存文件夹
    new_folder_1 = os.path.join(new_folder, "entity_img")
    new_folder_2 = os.path.join(new_folder, "entity_property_file")
    new_folder_3 = os.path.join(new_folder, "relation_img")
    excel_file_name = os.path.join(new_folder,"combined.xlsx")
    
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间为字符串，用 '_' 连接
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    zip_file_name = f"{URL_KNOWLEDGE_FUSION_TEMP}_{formatted_time}.zip"

    # 创建一个新文件夹
    if not os.path.exists(new_folder_1):
        os.makedirs(new_folder_1)
    if not os.path.exists(new_folder_2):
        os.makedirs(new_folder_2)
    if not os.path.exists(new_folder_3):
        os.makedirs(new_folder_3)
    
    # 使用 ExcelWriter 将它们写入同一个 Excel 文件
    with pd.ExcelWriter(excel_file_name) as writer:
        df_fusion_entities.to_excel(writer, sheet_name="实体表", index=False)  # 将第一个 DataFrame 写入子表 "Sheet1"
        df_fusion_relations.to_excel(writer, sheet_name="实体关系表", index=False)  # 将第二个 DataFrame 写入子表 "Sheet2"

    logging.info("融合后的图谱数据已成功写入 combined.xlsx 文件中！")

    # 压缩新文件夹为 ZIP 压缩包
    with ZipFile(zip_file_name, 'w') as zipf:
        # 遍历文件夹
        for root, dirs, files in os.walk(new_folder):
            # 添加空文件夹
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                rel_path = os.path.relpath(dir_path, new_folder) + '/'
                if not os.listdir(dir_path):  # 检查文件夹是否为空
                    zip_info = ZipInfo(rel_path)
                    zipf.writestr(zip_info, '')  # 写入一个空的条目代表空文件夹
            # 添加文件
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, new_folder))
    logging.info(f"新文件夹和 Excel 表已压缩为 ZIP 包:{zip_file_name}")

    three_list_data["knowledgeGraph"] = f"{HTTP_IP_PORT}/download/file/knowledge_fusion/KG_Fusion_{formatted_time}.zip"
    three_list_data["fusionId"] = fusion_id

    logging.info(f"异步回传的响应结果如下：\n{three_list_data}")

    logging.info("异步知识融合任务执行完毕,回传结果......")
    # 调用回调 URL 将结果发送回前端
    try:
        result = {
            "code": 200, 
            "msg": "success",
            "data": three_list_data,
        }

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_json_path = f"tools/graph_fusion/fusion_result_{current_time}.json"

        with open(result_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        requests.post(callback_url, json=result)


    except requests.exceptions.RequestException as e:
        logging.info(f"回调失败: {e}")

    ### 删除暂存文件夹
    if DELETE_TEMP_FILE == 1:
        shutil.rmtree(new_folder)


# 用蓝图注册路由
@blueprint_KnowledgeFusion.route("/ekg/fusion", methods=['post'])
def fusion():
    # 检查 HTTP 请求是否为 POST
    if request.method == 'POST':
        # 解析输入 JSON 数据
        try:
            input_data = json.loads(request.data)
            logging.info(f"知识融合接收到的请求参数如下：{input_data}\n")
            logging.info(f"Request data: {str(input_data)}")
        except json.JSONDecodeError:
            return jsonify({"code": 500, "msg": "输入的 JSON 数据格式错误！"})

        # 提取 extractionData 部分,包含实体和关系
        extraction_data = input_data.get("extractionData")
        # 提取的 fusionId 部分，为融合任务ID
        fusion_id = input_data.get("fusionId")
        # 提取 knowledgeGraph 部分
        knowledge_graph = input_data.get("knowledgeGraph")

        # 读取知识图谱
        if knowledge_graph:   # 图谱不为空
            graph_name = knowledge_graph.get("graphName")  # 图谱名称
            graph_type = knowledge_graph.get("graphType")  # 图谱类型
            zip_url = knowledge_graph.get("knowledgeGraph")  # 图谱URL
            # 检查图谱 URL 是否为 ZIP 文件链接
            if not zip_url.endswith('.zip'):
                return jsonify({"code": 500, "msg": "知识图谱 URL 不是 ZIP 文件链接！", "data": {"fusionId": fusion_id}})

            # 下载 ZIP 文件
            zip_url_new = zip_url
            zip_url_new = zip_url_new.replace("\\", "/")
            zip_filename = os.path.basename(zip_url_new)
            # local_zip_path = os.path.join(os.getcwd(), URL_DOWNLOAD_ZIP, zip_filename)   # 将zip文件下载到对应的api文件夹下api/knowledgeExtraction/uploads/<filename
            local_zip_path = os.path.join(URL_KNOWLEDGE_FUSION_RAW, zip_filename)   # 将zip文件下载到data/downloads/knowledge_fusion/<filename
            logging.info(f"local_zip_path:{local_zip_path}")
            try:
                response = requests.get(zip_url, stream=True, timeout=10)
                # logging.info(f"zip_url的地址为{zip_url}")
                response.raise_for_status()  # 检查 HTTP 状态码
                with open(local_zip_path, 'wb') as zip_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        zip_file.write(chunk)
            except Exception as e:
                logging.info(f"下载图谱 {graph_name} 的 ZIP 文件失败：{str(e)}")
                return jsonify({"code": 500, "msg": f"下载图谱 {graph_name} 的 ZIP 文件失败：{str(e)}", "data": {"fusionId": fusion_id}})

            # 解压 ZIP 文件并检查是否有 Excel 文件
            try:
                excel_path = None
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    last_dir_no_ext = copy.copy(zip_filename)
                    while '.' in last_dir_no_ext:
                        last_dir_no_ext = os.path.splitext(last_dir_no_ext)[0]
                    # extracted_dir = os.path.join(os.getcwd(), URL_DOWNLOAD_ZIP, last_dir_no_ext)   # zip文件解压目录
                    extracted_dir = os.path.join(URL_KNOWLEDGE_FUSION_RAW, last_dir_no_ext)   # zip文件解压目录
                    # logging.info(extracted_dir)
                    zip_ref.extractall(extracted_dir)

                    # 查找解压目录中的 Excel 文件
                    for root, _, files in os.walk(extracted_dir):
                        for file in files:
                            if file.endswith('.xls') or file.endswith('.xlsx'):
                                excel_path = os.path.join(root, file)
                                break

                    if not excel_path:
                        return jsonify({"code": 500, "msg": f"图谱 {graph_name} 的压缩包中没有找到 Excel 文件！", "data": {"fusionId": fusion_id}})

            except zipfile.BadZipFile:
                return jsonify({"code": 500, "msg": f"图谱 {graph_name} 的 ZIP 文件无效或损坏！", "data": {"fusionId": fusion_id}})
        else:
            return jsonify({"code": 500, "msg": "知识图谱 URL 为空！", "data": {"fusionId": fusion_id}})
        
        # 读取图谱zip文件中的excel内容
        df_entities, df_relation = read_excel(excel_path)

        # 使用executor异步处理耗时较长的任务
        g.executor.submit(async_knowledgeFusion, extraction_data, df_entities, df_relation, fusion_id, ASYNC_FUSION_RESULT_URL)

        ### 删除暂存文件夹
        if DELETE_TEMP_FILE == 1:
            shutil.rmtree(extracted_dir)
        #   shutil.rmtree(local_zip_path)
        # 返回成功信息
        return jsonify({"code": 200, "msg": f"知识融合任务ID:{fusion_id}已启动，请稍后查看结果。"})

# 模拟异步接口完成后回调(模拟后端)
@blueprint_KnowledgeFusion.route('/async_fusion_task_result', methods=['POST'])
def task_result():
    result = request.json.get('data')
    print(f"收到任务结果: {result}")
    return jsonify({"message": f"收到任务结果: {result}"})

if __name__ == "__main__":
    test_zip_path = 'graph_test.zip'
    excel_df = extract_and_read_excel(test_zip_path)