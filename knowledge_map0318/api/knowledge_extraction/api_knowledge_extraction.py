"""
知识抽取接口
"""
import os
from flask import Blueprint, jsonify, request, g
from log.log import logging
from tools.knowledge_extraction.knowledge_extraction import extract_knowledges
from tools.utils import download_file, extract_text, validate_keyword_position, process_next_task, add_task_to_queue
from value.global_value import LOCAL_TEST, BACKEND_ONTOLOGY_URL, ENTITY_DOWNLOAD_PATH, BACKEND_ENTITY_ONTOLOGY_URL, ASYNC_EN_RESULT_URL, DELETE_TEMP_FILE, BERT_MODEL_PATH
import datetime
import requests
import json
import shutil
import gc
import torch
from tools.trainning_model.main import extract_entities_and_relations, NERModelInterface, RelationPredictor
from tools.trainning_model.utils.config_ner import ConfigNer

# 注册蓝图
blueprint_entity_extraction = Blueprint('blueprint_entity_extraction', __name__)


def positions_overlap(pos1:list, pos2:list) -> bool:
    return not (pos1[1] <= pos2[0] or pos2[1] <= pos1[0])


# 知识抽取接口调用函数
def knowledge_extraction(files, ontologyAllId, extractionId, callback_url):
    """
    files          :抽取文件下载地址列表
    ontologyAllId  :实体表ID
    extractionId   :抽取任务ID
    callback_url   :任务完成后传给后端的地址
    """
    try:
        # 1 下载文件 & 获取文件中的文本
        file_texts = {}
        ### 本地测试：1  前后端联调：0
        if LOCAL_TEST==1:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(ENTITY_DOWNLOAD_PATH, current_time)   # 下载地址 + 当前时间
        for file in files:               
            file_path = download_file(file, save_path)    # 将文件下载到save_path目录下，返回单一文件的路径
            file_texts[file] = extract_text(file_path)    # 抽取文件的文本内容；file是http链接
                
        if not file_texts:   # 获取的文本内容为空
            result = {"code": 500, "message": "未找到任何文本内容。", "data": {"extractionId": extractionId}}
            requests.post(callback_url, json=result)
            return
            
        # 2 查询实体本体列表 转换成需要的格式
        if LOCAL_TEST==1:
            response = requests.get(f"{BACKEND_ENTITY_ONTOLOGY_URL}/{ontologyAllId}")     # 本地模拟后端调用
        else:
            response = requests.get(f"{BACKEND_ONTOLOGY_URL}?ontologyAllId={ontologyAllId}")
        print(response.status_code)
        if response.status_code == 200:
            # 获取返回的 JSON 数据
            data = response.json()  # 包含本体表和关系表
            logging.info(f"获取到的本体数据为{data}")
            # print(data)  # 打印获取到的数据
        else:
            result = {"code": 500, "message": "请求本体失败。", "data": {"extractionId": extractionId}}
            requests.post(callback_url, json=result)
            return

        ontologyClList = data['ontologyClList']  # 本体表
        ontologyReList = data['ontologyReList']  # 关系表
        # 本体名称list
        ontologySet = set()
        # 本体名称:id字典
        ontologyNaIDict = {}
        # 本体id:名称字典
        ontologyIdNDict = {}
        for item in ontologyClList:
            ontologySet.add(item["ontologyName"])
            ontologyNaIDict[item["ontologyName"]] = item["ontologyId"]
            ontologyIdNDict[item["ontologyId"]] = item["ontologyName"]
    
        relationDict = {}
        relationList = []
        # print(ontologyReList)
        # 遍历本体关系列表
        for item in ontologyReList:
            # 获取关系的头部和尾部实体名称
            headName = ontologyIdNDict.get(item["headId"])
            tailName = ontologyIdNDict.get(item["tailId"])
            # 获取本体中的关系名称
            relationList.append(item["relationName"])
            # 如果找到了头部和尾部的名称，就将关系加入字典
            if headName and tailName:
                # 生成键："headName-tailName"
                key = f"{headName}-{tailName}"
                
                # 如果键已经存在，追加关系名称；如果不存在，创建新的列表
                if key in relationDict:
                    relationDict[key].append(item["relationName"])
                else:
                    relationDict[key] = [item["relationName"]]
        print(relationList)
        print(relationDict)
        # 3 调用事件抽取函数
        result_dict = {}
        entity_all_dict = {}
        for file_name, file_text in file_texts.items():     # 一个文件一个文件地抽取
            print(file_name)
            # print(file_text)
            extract_flag = False
            extract_num = 0
            extract_result = None  # 防止报错
            while(not extract_flag and extract_num < 5):
                try:
                    extract_result = extract_knowledges(text_content=file_text,   # 待抽取的文本内容
                                                                llm_model=g.model,   # 大模型参数
                                                                tokenizer=g.tokenizer,    # 大模型参数
                                                                ontologyClList=list(ontologySet),    # 本体表
                                                                ontologyReList=relationDict       # 关系字典
                                                                )
                    extract_flag = True
                except Exception as e:
                    print(f"请求失败，错误信息：{e}")
                    print("大模型抽取出错，重新抽取中...")
                    extract_num += 1

            ## 抽取结果过滤
            new_extract_result = []   # 存放过滤后的三元组（过滤：三元组实体的本体是否为我们想要的，不符合的直接删掉该实体）
            ### 校验抽取三元组
            # 1、头尾实体都存在
            # 2、头尾实体的本体都在所给本体中
            # 3、关系在所给关系字典中
            for relation in extract_result:
                # 校验头尾实体内容
                if "headEntity" in relation and "tailEntity" in relation and "relationName" in relation and "score" in relation:    # 查看三元组中头尾实体是否都存在
                    headEntity = relation["headEntity"]
                    tailEntity = relation["tailEntity"]
                    # 校验本体
                    if "ontologyName" in headEntity and isinstance(headEntity["ontologyName"], str):
                        headId = ontologyNaIDict.get(headEntity["ontologyName"], "")
                        if not headId:    # 大模型给的本体名不在本体列表里面，直接删掉该实体
                            continue
                        headEntity["ontologyId"] = headId
                        headEntity.pop("ontologyName")   # 有本体ID后直接去掉本体名
                    else:
                        continue
                    if "ontologyName" in tailEntity and isinstance(tailEntity["ontologyName"], str):
                        tailId = ontologyNaIDict.get(tailEntity["ontologyName"], "")
                        if not tailId:    # 大模型给的本体名不在本体列表里面，直接删掉该实体
                            continue
                        tailEntity["ontologyId"] = tailId
                        tailEntity.pop("ontologyName")    # 有本体ID后直接去掉本体名
                    else:
                        continue
                    # print("本体检验完毕")
                    # 校验实体位置
                    # if "entityName" in headEntity and isinstance(headEntity["entityName"], str) and "location" in headEntity and isinstance(headEntity["location"], list):
                    #     entity_flag, entity_local = validate_keyword_position(file_text, headEntity["entityName"], headEntity["location"])
                    #     if not entity_flag:
                    #         continue
                    #     headEntity["location"] = entity_local
                    # else:
                    #     continue
                    # if "entityName" in tailEntity and isinstance(tailEntity["entityName"], str) and "location" in tailEntity and isinstance(tailEntity["location"], list):
                    #     entity_flag, entity_local = validate_keyword_position(file_text, tailEntity["entityName"], tailEntity["location"])
                    #     if not entity_flag:
                    #         continue
                    #     tailEntity["location"] = entity_local
                    # else:
                    #     continue
                    # print("实体检验完毕")
                    # 校验实体属性(属性三要素齐备)
                    # 校验属性存在
                    if "attributes" not in headEntity.keys() or not isinstance(headEntity["attributes"], list):
                        headEntity["attributes"] = []
                    if "attributes" not in tailEntity.keys() or not isinstance(tailEntity["attributes"], list):
                        tailEntity["attributes"] = []
                    # for attr_ in headEntity["attributes"]: 
                    #     if isinstance(attr_, dict) and attr_["key"] and isinstance(attr_["key"], str) and attr_["value"] and isinstance(attr_["value"], str) and attr_["location"] and isinstance(attr_["location"], list):
                    #         # 校验属性位置
                    #         attr_flag, attr_local = validate_keyword_position(file_text, attr_["value"], attr_["location"])
                    #         # 如果属性值找不到，从headEntity["attributes"]中删掉这条属性；如果能找到，位置修正
                    #         if not attr_flag:
                    #             headEntity["attributes"].remove(attr_)
                    #         else:
                    #             attr_["location"] = attr_local
                    #     else:
                    #         headEntity["attributes"].remove(attr_)

                    # for attr_ in tailEntity["attributes"]:
                    #     if isinstance(attr_, dict) and attr_["key"] and isinstance(attr_["key"], str) and attr_["value"] and isinstance(attr_["value"], str) and attr_["location"] and isinstance(attr_["location"], list):
                    #         # 校验属性位置
                    #         attr_flag, attr_local = validate_keyword_position(file_text, attr_["value"], attr_["location"])
                    #         # 如果属性值找不到，从tailEntity["attributes"]中删掉这条属性；如果能找到，位置修正
                    #         if not attr_flag:
                    #             tailEntity["attributes"].remove(attr_)
                    #         else:
                    #             attr_["location"] = attr_local
                    #     else:
                    #         tailEntity["attributes"].remove(attr_)
                    print("属性检验完毕")
                    print("---------------------------------------")
                    print(relation)
                    
                    ### 实体属性位置不重叠校验1（共两次校验）
                    # position_triplet = []
                    # position_triplet.append(headEntity["location"])
                    # position_triplet.append(tailEntity["location"])

                    # # 1、检查头尾实体是否有重叠
                    # if positions_overlap(headEntity["location"], tailEntity["location"]):
                    #     continue

                    # # 2、检查属性间位置是否有重叠
                    # for attr in headEntity["attributes"]:
                    #     # print(attr)
                    #     attr_intersect = False
                    #     for attr_pos in position_triplet:
                    #         # print(attr_pos)
                    #         if positions_overlap(attr_pos, attr["location"]):
                    #             headEntity["attributes"].remove(attr)
                    #             attr_intersect = True
                    #             break
                    #     if not attr_intersect:
                    #         position_triplet.append(attr["location"])
                    #     # print(positions_overlap(attr_pos, attr["location"]))
                    
                    # for attr in tailEntity["attributes"]:
                    #     # print(attr)
                    #     attr_intersect = False
                    #     for attr_pos in position_triplet:
                    #         # print(attr_pos)
                    #         if positions_overlap(attr_pos, attr["location"]):
                    #             tailEntity["attributes"].remove(attr)
                    #             attr_intersect = True
                    #             break
                    #     if not attr_intersect:
                    #         position_triplet.append(attr["location"])
                    #     print(positions_overlap(attr_pos, attr["location"]))
                    # print(relation)
                    # print("---------------------------------------")

                    # 校验关系               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!因验收修改
                    if relation["relationName"] in relationList:
                       new_extract_result.append(relation)
                    # new_extract_result.append(relation)
            # print(new_extract_result)

            ### 实体属性位置不重叠校验2
            # occupied_position = []
            # extract_result_final = []
            # # 实体位置校验
            # for relation in new_extract_result:
            #     headEntity = relation["headEntity"]
            #     tailEntity = relation["tailEntity"]
            #     entity_intersected = False
            #     for pos in occupied_position:
            #         if positions_overlap(pos, headEntity["location"]) or positions_overlap(pos, tailEntity["location"]):
            #             entity_intersected = True
            #             break
            #     if not entity_intersected:
            #         occupied_position.append(headEntity["location"])
            #         occupied_position.append(tailEntity["location"])
            #         extract_result_final.append(relation)
            # # 属性位置校验
            # for relation in extract_result_final:
            #     headEntity = relation["headEntity"]
            #     tailEntity = relation["tailEntity"]
            #     for attr in headEntity["attributes"]:
            #         for pos in occupied_position:
            #             attr_intersect = False
            #             if positions_overlap(pos, attr["location"]):
            #                 headEntity["attributes"].remove(attr)
            #                 attr_intersect = True
            #                 break
            #         if attr_intersect:
            #             occupied_position.append(attr["location"])
            #     for attr in tailEntity["attributes"]:
            #         for pos in occupied_position:
            #             attr_intersect = False
            #             if positions_overlap(pos, attr["location"]):
            #                 tailEntity["attributes"].remove(attr)
            #                 attr_intersect = True
            #                 break
            #         if attr_intersect:
            #             occupied_position.append(attr["location"])



            extract_result_final = new_extract_result    #位置校验注释打开，位置校验关闭则注释掉 
            ## 实体消歧统计
            for relation in extract_result_final:
                headEntity = relation["headEntity"]
                tailEntity = relation["tailEntity"]
                if headEntity["entityName"] not in entity_all_dict:   # 实体不在实体字典中，添加进字典，并把实体类型作为value
                    entity_all_dict[headEntity["entityName"]] = [ontologyIdNDict[headEntity["ontologyId"]]]
                else:     # 实体在实体字典中，找到该实体，看看该实体的实体类型列表包不包含本实体类型，包含直接跳过，不包含则添加
                    if ontologyIdNDict[headEntity["ontologyId"]] not in entity_all_dict[headEntity["entityName"]]:
                        entity_all_dict[headEntity["entityName"]].append(ontologyIdNDict[headEntity["ontologyId"]])

            result_dict[file_name] = extract_result_final
        
        # 4、实体消歧
        disam = {key: value for key, value in entity_all_dict.items() if len(value) != 1}
        
        result = {
            "code": 200,
            "message": "success",
            "data": {
                "extractionId": extractionId,
                "extractionIResult": result_dict,
                "disambiguation": disam
            }
        }
        print(result)
        logging.info(f"知识抽取接口的返回数据为:\n{result}")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"tools/knowledge_extraction/knowledge_extraction_result_{current_time}.json", 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        requests.post(callback_url, json=result)

        ### 删除暂存文件夹
        if DELETE_TEMP_FILE == 1:
            shutil.rmtree(save_path)

    except Exception as e:
        result = {"code": 500, "message": f"请求失败，错误信息：{e}", "data": {"extractionId": extractionId}}
        print(result)
        requests.post(callback_url, json=result)
        ### 删除暂存文件夹
        if DELETE_TEMP_FILE == 1:
            shutil.rmtree(save_path)

# 知识抽取接口调用函数----使用本地训练的模型
def knowledge_extraction_train_model(model_id, model_path, files, ontologyAllId, extractionId, callback_url):
    """
    model_id       :模型ID
    model_path     :模型路径
    files          :抽取文件下载地址列表
    ontologyAllId  :实体表ID
    extractionId   :抽取任务ID
    callback_url   :任务完成后传给后端的地址
    ---------------------------------------------
    注意：
    1、步骤包含：下载并读取文件、调用训练好的模型、抽取
    2、这里不用考虑本体ID，直接从模型库里面下载？？？？？？？？？？？？？？？？？？
    """
    try:
        # 1 下载文件 & 获取文件中的文本
        file_texts = {}
        for file in files:
            if LOCAL_TEST==1:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d")
                save_path = os.path.join(ENTITY_DOWNLOAD_PATH, current_time)   # 下载地址 + 当前时间
                file_path = os.path.join(save_path, os.path.basename(file))
                print(file_path)
                file_texts[file] = extract_text(file_path)    # 抽取文件的文本内容；file是http链接
            else:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = os.path.join(ENTITY_DOWNLOAD_PATH, current_time)   # 下载地址 + 当前时间
                file_path = download_file(file, save_path)    # 将文件下载到save_path目录下，返回单一文件的路径
                print(file_path)
                file_texts[file] = extract_text(file_path)    # 抽取文件的文本内容；file是http链接
        print(file_texts)
        if not file_texts:   # 获取的文本内容为空
            result = {"code": 400, "message": "未找到任何文本内容。", "data": {"extractionId": extractionId}}
            requests.post(callback_url, json=result)
            return
            
        # 2 查询实体本体列表 转换成需要的格式
        if LOCAL_TEST==1:
            response = requests.get(f"{BACKEND_ENTITY_ONTOLOGY_URL}/{ontologyAllId}")     # 本地模拟后端调用
        else:
            response = requests.get(f"{BACKEND_ONTOLOGY_URL}?ontologyAllId={ontologyAllId}")

        if response.status_code == 200:
            # 获取返回的 JSON 数据
            ontologyAll = response.json()  # 包含本体表和关系表
            logging.info(f"获取到的本体数据为{ontologyAll}")
            # print(data)  # 打印获取到的数据
        else:
            result = {"code": 500, "message": "请求本体失败。", "data": {"extractionId": extractionId}}
            print(result)
            requests.post(callback_url, json=result)
            return

        ontologyClList = ontologyAll['ontologyClList']  # 本体表
        ontologyReList = ontologyAll['ontologyReList']  # 关系表
        # 本体名称list
        ontologySet = set()
        # 本体名称:id字典
        ontologyNaIDict = {}
        # 本体id:名称字典
        ontologyIdNDict = {}
        for item in ontologyClList:
            ontologySet.add(item["ontologyName"])
            ontologyNaIDict[item["ontologyName"]] = item["ontologyId"]
            ontologyIdNDict[item["ontologyId"]] = item["ontologyName"]
    
        relationDict = {}
        relationList = ["N"]          # !!!!!!!!!!!！！！！！！！！！！！！！！！！!!!!!!!!!!!!!!!!!(需修改) 后期有训练好的模型，这里的初始N要删掉
        # 遍历本体关系列表
        for item in ontologyReList:
            # 获取关系的头部和尾部实体名称
            headName = ontologyIdNDict.get(item["headId"])
            tailName = ontologyIdNDict.get(item["tailId"])
            # 获取本体中的关系名称
            relationList.append(item["relationName"])
            # 如果找到了头部和尾部的名称，就将关系加入字典
            if headName and tailName:
                # 生成键："headName-tailName"
                key = f"{headName}-{tailName}"
                
                # 如果键已经存在，追加关系名称；如果不存在，创建新的列表
                if key in relationDict:
                    relationDict[key].append(item["relationName"])
                else:
                    relationDict[key] = [item["relationName"]]
        print(ontologyIdNDict)
        print("-------------------------------------")
        print(relationList)
        
        ### 2、预加载抽取模型
        # 根据 modelId 选择模型（此处示例中固定路径，可根据需要扩展）
        print(model_path)
        PATH_NER = os.path.join(model_path,'sequence_labeling/ner_best_model.pth')     # 实体抽取模型文件保存路径
        PATH_REL = os.path.join(model_path,'rel_cls/rel_best_model.pth')    # 关系抽取模型文件保存路径
        relations_path = os.path.join(model_path,'rel_cls/relation_types.json')   # 关系类型文件保存路径
        token_types_path = os.path.join(model_path,'sequence_labeling/entity_types.json')   # 实体类型文件保存路径
        bert_model_dir = BERT_MODEL_PATH      # bert模型保存路径

        # 读取文件获取relations和token_types
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        with open(token_types_path, 'r', encoding='utf-8') as f:
            token_types = json.load(f)
        # print(relations)

        # 初始化实体抽取器（只加载一次BERT模型）
        try:
            config_ner = ConfigNer(relations=relations, token_types_origin=token_types)
            ner_interface = NERModelInterface(PATH_NER, config_ner)

            # 初始化关系预测器（只加载一次BERT模型）
            relation_predictor = RelationPredictor(bert_model_dir, PATH_REL, relations, token_types)
        except Exception as e:
            # 模型加载失败，返回路径错误信息
            print(e)
            logging.error(f"模型加载失败，请检查模型路径是否正确: {e}")
            return jsonify({'error': '模型加载失败，请检查模型路径是否正确'})
            
        ### 3 调用事件抽取函数
        result_dict = {}
        entity_all_dict = {}
        for file_name, file_text in file_texts.items():     # 一个文件一个文件地抽取
            print(file_name)
            extract_result_1 = extract_entities_and_relations(file_text,    # 待抽取的文本内容
                                           ner_interface,  # NER模型接口
                                           relation_predictor,  # 关系预测器
                                           ontologyAll    # 本体(包含实体类型和关系)
                                           )
            print(extract_result_1)
            extract_result = extract_result_1["result"]
            print(extract_result)
            ## 抽取结果过滤 
            # 注意;
            # 1、相比于大模型抽取的三元组，这里抽取的三元组没有属性，所以属性不需要校验
            # 2、这里抽取的内容没有属性
            # 3、这里抽取的本体id就是本体名
            new_extract_result = []   # 存放过滤后的三元组（过滤：三元组实体的本体是否为我们想要的，不符合的直接删掉该实体）
            ### 校验抽取三元组
            # 1、头尾实体都存在
            # 2、头尾实体的本体都在所给本体中
            # 3、关系在所给关系字典中
            for relation in extract_result:
                # 校验头尾实体内容
                if "headEntity" in relation and "tailEntity" in relation:    # 查看三元组中头尾实体是否都存在
                    headEntity = relation["headEntity"]
                    tailEntity = relation["tailEntity"]
                    # 校验本体
                    if "ontologyId" in headEntity and isinstance(headEntity["ontologyId"], str):
                        headId = ontologyIdNDict.get(headEntity["ontologyId"],"")
                        if not headId:    # 大模型给的本体名不在本体列表里面，直接删掉该实体
                            continue
                        # headEntity["ontologyId"] = headId
                        # headEntity.pop("ontologyName")   # 有本体ID后直接去掉本体名
                    else:
                        continue
                    if "ontologyId" in tailEntity and isinstance(tailEntity["ontologyId"], str):
                        tailId = ontologyIdNDict.get(tailEntity["ontologyId"],"")
                        if not tailId:    # 大模型给的本体名不在本体列表里面，直接删掉该实体
                            continue
                        # tailEntity["ontologyId"] = tailId
                        # tailEntity.pop("ontologyName")    # 有本体ID后直接去掉本体名
                    else:
                        continue
                    # print("本体检验完毕")
                    # 校验实体位置
                    if "entityName" in headEntity and isinstance(headEntity["entityName"], str) and "location" in headEntity and isinstance(headEntity["location"], list):
                        entity_flag, entity_local = validate_keyword_position(file_text, headEntity["entityName"], headEntity["location"])
                        if not entity_flag:
                            continue
                        headEntity["location"] = entity_local
                    else:
                        continue
                    if "entityName" in tailEntity and isinstance(tailEntity["entityName"], str) and "location" in tailEntity and isinstance(tailEntity["location"], list):
                        entity_flag, entity_local = validate_keyword_position(file_text, tailEntity["entityName"], tailEntity["location"])
                        if not entity_flag:
                            continue
                        tailEntity["location"] = entity_local
                    else:
                        continue
                    # print("实体位置检验完毕")
                    # 校验实体属性(属性三要素齐备)!!!这里抽取的没有属性，属性置空
                    headEntity["attributes"] = []
                    tailEntity["attributes"] = []


                    ### 实体属性位置不重叠校验1（共两次校验）
                    # 1、检查头尾实体是否有重叠
                    if positions_overlap(headEntity["location"], tailEntity["location"]):
                        continue

                    # 校验关系                                  ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！因验收修改
                    # if relation["relationName"] in relationList:
                    #     new_extract_result.append(relation)
                    new_extract_result.append(relation)
                    # print(relation["relationName"])
            print("-------------------new_extract_result---------------------")
            print(new_extract_result)
            ### 实体属性位置不重叠校验2
            occupied_position = []
            # extract_result_final = []
            # # 实体位置校验
            # for relation in new_extract_result:
            #     headEntity = relation["headEntity"]
            #     tailEntity = relation["tailEntity"]
            #     entity_intersected = False
            #     for pos in occupied_position:
            #         if positions_overlap(pos, headEntity["location"]) or positions_overlap(pos, tailEntity["location"]):
            #             entity_intersected = True
            #             break
            #     if not entity_intersected:
            #         occupied_position.append(headEntity["location"])
            #         occupied_position.append(tailEntity["location"])
            #         extract_result_final.append(relation)
            # print(extract_result_final)


            extract_result_final = new_extract_result    #位置校验注释打开，位置校验关闭则注释掉 
            ## 实体消歧统计
            for relation in extract_result_final:
                headEntity = relation["headEntity"]
                tailEntity = relation["tailEntity"]
                if headEntity["entityName"] not in entity_all_dict:   # 实体不在实体字典中，添加进字典，并把实体类型作为value
                    entity_all_dict[headEntity["entityName"]] = [ontologyIdNDict[headEntity["ontologyId"]]]
                else:     # 实体在实体字典中，找到该实体，看看该实体的实体类型列表包不包含本实体类型，包含直接跳过，不包含则添加
                    if ontologyIdNDict[headEntity["ontologyId"]] not in entity_all_dict[headEntity["entityName"]]:
                        entity_all_dict[headEntity["entityName"]].append(ontologyIdNDict[headEntity["ontologyId"]])

            result_dict[file_name] = extract_result_final
        
        # 4、实体消歧
        disam = {key: value for key, value in entity_all_dict.items() if len(value) != 1}
        
        result = {
            "code": 200,
            "message": "success",
            "data": {
                "extractionId": extractionId,
                "extractionIResult": result_dict,
                "disambiguation": disam
            }
        }

        logging.info(f"知识抽取接口的返回数据为:\n{result}")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"tools/knowledge_extraction/knowledge_extraction_result_{current_time}.json", 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        requests.post(callback_url, json=result)

        ### 删除暂存文件夹
        if DELETE_TEMP_FILE == 1:
            shutil.rmtree(save_path)
        
        ### 删除GPU中加载的模型
        # ===== 添加显存释放逻辑 =====
        if torch.cuda.is_available():
            # 删除模型和其它相关对象引用
            del config_ner, ner_interface, relation_predictor
            # 调用垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
            print("训练模型（知识抽取用）所占用的GPU显存已释放。")

    except Exception as e:
        result = {"code": 500, "message": f"请求失败，错误信息：{e}", "data": {"extractionId": extractionId}}
        print(result)
        requests.post(callback_url, json=result)
        ### 删除暂存文件夹
        if DELETE_TEMP_FILE == 1:
            shutil.rmtree(save_path)

# 异步接口--处理耗时较长的任务
@blueprint_entity_extraction.route('/ekg/knowledgeExtraction', methods=['POST'])
def start_task():
    try:
        # 获取请求参数
        data = request.json
        model_ID = data['modelId']    # 模型ID(包含模型类型、模型路径、模型id)
        if model_ID["model_type"] == 0:  # 0:本地训练模型
            model_path = model_ID["model_path"]
            model_id = model_ID["model_id"]
        elif model_ID["model_type"] == 1:
            model_id = model_ID["model_id"]
        else:     # 否则抛出异常
            raise Exception("模型类型错误")

        files = data['files']    # 文件下载列表
        ontologyAllId = data['ontologyAllId']   # 本体表
        extractionId = data['extractionId']    # 抽取任务ID
        logging.info(f"接收到请求参数：{files}, {ontologyAllId}, {extractionId}")

        # 对不同模型区分处理
        if model_ID["model_type"] == 1:    # 使用大模型抽取
            # 使用executor异步处理耗时较长的任务
            print("使用大模型抽取")
            g.executor.submit(knowledge_extraction, files, ontologyAllId, extractionId, ASYNC_EN_RESULT_URL)
        elif model_ID["model_type"] == 0:
            print("使用本地训练模型抽取")
            g.executor.submit(knowledge_extraction_train_model, model_id, model_path, files, ontologyAllId, extractionId, ASYNC_EN_RESULT_URL)
        else:
            raise Exception("模型类型错误")
    
        response = {
            "code": 200,
            "msg": f"知识抽取任务(ID:{extractionId})已提交，正在后台处理中！"
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"知识抽取任务提交失败: {str(e)}")
        response = {
            "code": 500,
            "msg": f"知识抽取任务提交失败: {str(e)}"
        }
        return jsonify(response)
  
# 模拟异步接口完成后回调(模拟后端)
@blueprint_entity_extraction.route('/async_entity_relation_task_result', methods=['POST'])
def task_result():
    result = request.json.get('data')
    print(f"收到任务结果: {result}")
    return jsonify({"message": f"收到任务结果: {result}"})

# 模拟后端返回本体(模拟后端)
@blueprint_entity_extraction.route('/ontology/getOntologyData/entity/<ontologyAllId>', methods=['GET'])
def getOntologyData(ontologyAllId):
    print("模拟后端获取到本体Id，查询本体的过程")
    # 读取 JSON 文件
    if ontologyAllId == "180d71ac70eb4e9282845ed658daca05":
        with open(os.path.join(os.getcwd(), "./tools/knowledge_extraction/knowledge_ontology.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif ontologyAllId == "180d71ac70eb4e9282845ed658daca06":
        with open(os.path.join(os.getcwd(), "./tools/knowledge_extraction/knowledge_ontology_training_model.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(os.path.join(os.getcwd(), "./tools/knowledge_extraction/knowledge_ontology.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)

    return jsonify(data)