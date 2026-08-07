'''
多图融合接口
'''
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import zipfile
import shutil
import requests
import pandas as pd
from datetime import datetime
import copy
from zipfile import ZipFile, ZipInfo
from flask import Blueprint, request, jsonify, g

from log.log import logging
from value.global_value import URL_DOWNLOAD_MUITGRAPH_ZIP, URL_UPLOAD_MUITGRAPH_ZIP, HTTP_IP_PORT, ASYNC_MULTIF_RESULT_URL, LOCAL_TEST
from tools.graph_fusion.knowledge_fusion import knowledge_KG_Fusion
from api.knowledge_extraction.api_knowledge_fusion import extract_and_read_excel, read_excel

from tools.graph_fusion.fusion_utils import merge_event_bk

# 注册一个
blueprint_graphFusion = Blueprint('graph_fusion', __name__)


# # 解压并读取excel表
# def extract_and_read_excel(zip_file_path, extract_to='uploads'):
#     '''
#     功能：解压图谱zip文件
#     输入： zip_file_path  --zip压缩包路径
#            extract_to  --解压后文件存放目录
#     输出：excel表存储的图谱内容
#     '''
#     # 检查目录是否存在，如果不存在则创建
#     if not os.path.exists(extract_to):
#         os.makedirs(extract_to)
#     # 解压缩文件
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
    
#     # 压缩包第一级路径
#     # 压缩包第一级路径
#     last_dir_no_ext = os.path.basename(copy.copy(zip_file_path))
#     while '.' in last_dir_no_ext:
#         last_dir_no_ext = os.path.splitext(last_dir_no_ext)[0]
#     print(last_dir_no_ext)
#     zip_first_path = os.path.join(extract_to, last_dir_no_ext)
#     print(zip_first_path)
#     print(os.listdir(zip_first_path))  # 注意其中有两层文件夹
#     # 查找解压后的Excel文件
#     excel_excel_file = [f for f in os.listdir(zip_first_path) if f.endswith('.xlsx') or f.endswith('.xls')]
#     # 确保读取的excel文件只有一个
#     if len(excel_excel_file) != 2:
#         # shutil.rmtree(extract_to)  # 清理提取的目录
#         raise ValueError(f"发现图谱压缩包中包含 {len(excel_excel_file)} 个Excel文件，期望有2个。请检查并重新输入！")
#     # 读取Excel文件
#     excel_path1 = os.path.join(zip_first_path, excel_excel_file[0])
#     excel_path2 = os.path.join(zip_first_path, excel_excel_file[1])

#     excels1 = pd.ExcelFile(excel_path1)
#     excels2 = pd.ExcelFile(excel_path2)

#     # 一个基础知识表,一个事件表
#     print(excels1.sheet_names)
#     print(excels2.sheet_names)

#     if '实体表' not in excels1.sheet_names or '实体关系表' not in excels1.sheet_names:
#         raise ValueError(f"zip压缩包中的图谱excel表中不包含“实体表”或“实体关系表”，请检查名称是否与模板严格一致！！！")

#     if '实体表' not in excels2.sheet_names or '实体关系表' not in excels2.sheet_names:
#         raise ValueError(f"zip压缩包中的图谱excel表中不包含“实体表”或“实体关系表”，请检查名称是否与模板严格一致！！！")


#     df1_entities = pd.read_excel(excel_path1, sheet_name='实体表')  # 使用pandas读取Excel文件
#     df1_relations = pd.read_excel(excel_path1, sheet_name='实体关系表')

#     df2_entities = pd.read_excel(excel_path2, sheet_name='实体表')  # 使用pandas读取Excel文件
#     df2_relations = pd.read_excel(excel_path2, sheet_name='实体关系表')


#     # print(f"内容来自 {excel_excel_file}：")
#     # print(df_entities)  # 打印DataFrame内容
#     # print(df_relation)  
    
#     # # 删除提取的目录及其内容
#     # shutil.rmtree(extract_to)
#     return df1_entities, df1_relations, df2_entities, df2_relations 

def async_mutigraphFusion(new_graphs, fusion_id, callback_url):
    # new_graphs: 多个知识图谱的信息和下载地址
    # fusion_id: 融合任务ID
    # callback_url: 融合结果回调地址
    logging.info("异步多图融合任务启动......")

    ### 多图谱融合逻辑
    # 1、知识图谱与知识图谱融合，事件图谱与事件图谱融合
    # 2、知识图谱与事件图谱融合
    try:
        # 处理所有图谱
        knowledge_flag = 0  # 知识图谱标志
        event_flag = 0   # 事件图谱标志
        if new_graphs:
            # 事件图谱和事件图谱融合，知识图谱和知识图谱融合
            for graph in new_graphs:
                if graph["graphType"] == "知识图谱" and knowledge_flag == 0: 
                    # 4、知识图谱导入模板.xls 文件：图谱数据，excel表形式存储
                    knowledge_entities, knowledge_relation = read_excel(graph["knowledgeGraph"])
                    knowledge_flag = 1
                elif graph["graphType"] == "知识图谱" and knowledge_flag == 1:
                    knowledge_entities_1, knowledge_relation_1 = read_excel(graph["knowledgeGraph"])
                    knowledge_entities, knowledge_relation = knowledge_KG_Fusion(knowledge_entities_1, knowledge_relation_1, knowledge_entities, knowledge_relation)
                    print(knowledge_entities, knowledge_relation)
                elif graph["graphType"] == "事件图谱" and event_flag == 0:
                    # 5、事件图谱导入模板.xls 文件：事件数据，excel表形式存储
                    print(graph["knowledgeGraph"])
                    event_entities, event_relation = read_excel(graph["knowledgeGraph"])
                    print(knowledge_entities, knowledge_relation)
                    event_flag = 1
                elif graph["graphType"] == "事件图谱" and event_flag == 1:
                    event_entities_1, event_relation_1 = read_excel(graph["knowledgeGraph"])
                    event_entities, event_relation = knowledge_KG_Fusion(event_entities_1, event_relation_1, event_entities, event_relation)
                    print(event_entities, event_relation)
                else:
                    logging.error("读取的图谱的图谱类型，既不是知识图谱也不是事件图谱，请重新上传事件图谱或知识图谱！！！")
                    return jsonify({"code": 400, "msg": "读取的图谱的图谱类型，既不是知识图谱也不是事件图谱，请重新上传事件图谱或知识图谱！！！"})

            # 事件图谱和知识图谱融合
            if event_flag == 1 and knowledge_flag == 1:
                # 事件和知识图谱融合
                kg_entities, kg_relation = merge_event_bk(event_entities, knowledge_entities, event_relation, knowledge_relation)
                # logging.info(kg_entities)
                # logging.info(kg_relation)
            elif event_flag == 1 and knowledge_flag == 0:    # 输入图谱只有事件图谱
                kg_entities, kg_relation = event_entities, event_relation
            elif event_flag == 0 and knowledge_flag == 1:     # 输入图谱只有知识图谱
                kg_entities, kg_relation = knowledge_entities, knowledge_relation
            else:
                logging.error("读取的图谱既不是知识图谱也不是事件图谱，请重新上传事件图谱或知识图谱！！！")
                return jsonify({"code": 400, "msg": "读取的图谱既不是知识图谱也不是事件图谱，请重新上传事件图谱或知识图谱！！！"})
    except Exception as e:
        logging.error("Throw an exception: {}".format(e))
        return jsonify({"code": 500, "msg": "Server Error, throw an exception: {}".format(e)})
        
    # 将图谱包装为zip形式传回去
    ### 将融合后的实体表和关系表导入压缩包
    # 定义文件和文件夹名称
    new_folder = os.path.join(URL_UPLOAD_MUITGRAPH_ZIP, "graph_fusion/Muti_KG_Fusion")
    new_folder_1 = os.path.join(new_folder, "entity_img")
    new_folder_2 = os.path.join(new_folder, "entity_property_file")
    new_folder_3 = os.path.join(new_folder, "relation_img")
    excel_file_name = os.path.join(new_folder, "muti_graph_combined.xlsx")
    
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间为字符串，用 '_' 连接
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    zip_file_name = f"{URL_UPLOAD_MUITGRAPH_ZIP}/graph_fusion/Muti_KG_Fusion_{formatted_time}.zip"

    # 创建一个新文件夹
    if not os.path.exists(new_folder_1):
        os.makedirs(new_folder_1)
    if not os.path.exists(new_folder_2):
        os.makedirs(new_folder_2)
    if not os.path.exists(new_folder_3):
        os.makedirs(new_folder_3)
    
    # 使用 ExcelWriter 将它们写入同一个 Excel 文件
    with pd.ExcelWriter(excel_file_name) as writer:
        kg_entities.to_excel(writer, sheet_name="实体表", index=False)  # 将第一个 DataFrame 写入子表 "Sheet1"
        kg_relation.to_excel(writer, sheet_name="实体关系表", index=False)  # 将第二个 DataFrame 写入子表 "Sheet2"

    logging.info("数据已成功写入 muti_graph_combined.xlsx 文件中！")

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
    logging.info(zip_file_name)
    logging.info("新文件夹和 Excel 表已压缩为 ZIP 包:%s", zip_file_name)

    # 获取下载地址
    zip_file_path1 = zip_file_name.replace("data/downloads", "")
    fusion_ip = f"{HTTP_IP_PORT}/download/file{zip_file_path1}"
    print(fusion_ip)

    if LOCAL_TEST==1:
        # 删除新图谱所在文件夹
        shutil.rmtree(new_folder)
        # 删除解压后的文件夹
        shutil.rmtree(os.path.join(os.getcwd(), zip_file_name))
            
    logging.info("异步多图融合任务执行完毕,回传结果......")
    # 调用回调 URL 将结果发送回前端
    try:
        requests.post(callback_url, json={
                                            "code": 200,
                                            "msg": "success",
                                            "data": {
                                                "fusedKG": fusion_ip,
                                                "fusionId": fusion_id
                                            }       
                                            })
    except requests.exceptions.RequestException as e:
        logging.error(f"回调失败: {e}")



# 用蓝图注册路由
@blueprint_graphFusion.route("/ekg/multiFusion", methods=['post'])
def graph_fusion():
    # 检查请求方法是否为 POST
    if request.method == 'POST':
        try:
            input_data = json.loads(request.data)
            logging.info("多图融合接收到的请求参数如下：\n%s", input_data)
        except json.JSONDecodeError:
            return jsonify({"code": 500, "msg": "输入的 JSON 数据格式错误！"})

        # 获取 knowledgeGraphs 列表
        knowledge_graphs = input_data.get("knowledgeGraphs", [])
        logging.info(knowledge_graphs)
        # 获取多图融合任务ID
        fusion_id = input_data.get("fusionId", "")

        if not knowledge_graphs:
            return jsonify({"code": 500, "msg": "未提供 knowledgeGraphs 列表！"})

        # 本地存储路径
        os.makedirs(URL_DOWNLOAD_MUITGRAPH_ZIP, exist_ok=True)  # "api/graph_fusion/uploads"

        # 存储新的字典列表
        new_graphs = []
        for graph in knowledge_graphs:
            graph_name = graph.get("graphName")
            graph_type = graph.get("graphType")
            zip_url = graph.get("knowledgeGraph")
            print(zip_url)
            # 验证 ZIP 文件链接
            if not zip_url or not zip_url.endswith('.zip'):
                return jsonify({"code": 500, "msg": f"图谱 {graph_name} 的链接无效或不是 ZIP 文件！"})

            # 下载 ZIP 文件
            zip_filename = os.path.basename(zip_url)
            local_zip_path = os.path.join(os.getcwd(), URL_DOWNLOAD_MUITGRAPH_ZIP, zip_filename)
            print(zip_url)
            print(local_zip_path)
            logging.info("zip文件下载链接：%s", zip_url)
            logging.info("zip文件保存地址：%s",local_zip_path)
            try:
                response = requests.get(zip_url, stream=True, timeout=10)
                # print(response)
                response.raise_for_status()  # 检查 HTTP 状态码
                with open(local_zip_path, 'wb') as zip_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        zip_file.write(chunk)
                print("下载成功")
            except requests.RequestException as e:
                return jsonify({"code": 500, "msg": f"下载图谱 {graph_name} 的 ZIP 文件失败：{str(e)}"})

            # 解压 ZIP 文件并检查是否有 Excel 文件
            try:
                excel_path = None
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    last_dir_no_ext = copy.copy(zip_filename)
                    while '.' in last_dir_no_ext:
                        last_dir_no_ext = os.path.splitext(last_dir_no_ext)[0]
                    print(last_dir_no_ext)
                    extracted_dir = os.path.join(os.getcwd(), URL_DOWNLOAD_MUITGRAPH_ZIP, last_dir_no_ext)
                    # extracted_dir = os.path.join(os.getcwd(), URL_DOWNLOAD_MUITGRAPH_ZIP)
                    zip_ref.extractall(extracted_dir)
                    logging.info("解压目录：%s",extracted_dir)

                    # 查找解压目录中的 Excel 文件
                    for root, _, files in os.walk(extracted_dir):
                        for file in files:
                            if file.endswith('.xls') or file.endswith('.xlsx'):
                                excel_path = os.path.join(root, file)
                                break

                    if not excel_path:
                        return jsonify({"code": 500, "msg": f"图谱 {graph_name} 的压缩包中没有找到 Excel 文件！"})

            except zipfile.BadZipFile:
                return jsonify({"code": 500, "msg": f"图谱 {graph_name} 的 ZIP 文件无效或损坏！"})

            # 修改字典并加入新列表
            new_graphs.append({
                "graphName": graph_name,
                "graphType": graph_type,
                "knowledgeGraph": excel_path  # 修改为 Excel 文件路径
            })
        logging.info(new_graphs)

        # 使用executor异步处理耗时较长的任务
        g.executor.submit(async_mutigraphFusion, new_graphs, fusion_id, ASYNC_MULTIF_RESULT_URL)

        # 返回成功信息
        return jsonify({"code": 200, "msg": f"多图融合任务ID:{fusion_id}已启动，请稍后查看结果。"})

    # 非 POST 请求
    return jsonify({"code": 500, "msg": "仅支持 POST 请求！"})


# 模拟异步接口完成后回调(模拟后端)
@blueprint_graphFusion.route('/async_mutigraph_fusion_task_result', methods=['POST'])
def task_result():
    result = request.json.get('data')
    print(f"收到任务结果: {result}")
    return jsonify({"message": f"收到任务结果: {result}"})

if __name__ == "__main__":
    test_zip_path = 'graph_fusion.zip'
    df1_entities, df1_relations, df2_entities, df2_relations  = extract_and_read_excel(test_zip_path)
    merge_event_bk(df1_entities, df2_entities, df1_relations, df2_relations)