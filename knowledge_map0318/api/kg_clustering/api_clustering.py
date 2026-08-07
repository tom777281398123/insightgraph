'''
聚类接口
'''
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import zipfile
import shutil
import pandas as pd
from flask import Blueprint, request, jsonify
from log.log import logging
from tools.kg_clustering.kg_cluster import kg_clustering

# 注册一个
blueprint_kg_cluster = Blueprint('groupAnalysis', __name__)


# 用蓝图注册路由
# @blueprint_rcsEntityExtraction.route("/ekg/kg_clustering", methods=['post'])
# def fusion():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return jsonify({"code": 500, "msg": "传入文件错误！"})
#         file = request.files['file']  # 获取后端传来的压缩包文件

#         cluster_result = kg_clustering()
#         output = {
#             "code": 200,
#             "msg": "success",
#             "data": cluster_result
#         }
#     return jsonify(response)

@blueprint_kg_cluster.route("/ekg/groupAnalysis", methods=['post'])
def group_analysis():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     return jsonify({"code": 500, "msg": "传入文件错误！"})
        # file = request.files['file']  # 获取后端传来的压缩包文件
        try:
            data = request.json
            logging.info(f"群体分析接收到的请求参数如下：\n_{data}")
            entities = data.get('entities')
            relations = data.get('relations')
            tobeClustered = data.get('tobeClustered')
            clusterNum = data.get('clusterNum')
            
            result = kg_clustering(entities, relations, tobeClustered, clusterNum)

            print(result)
            output = {
                "code": 200,
                "msg": "success",
                "data": result
            }
            logging.info(f"群体分析返回结果如下：\n{output}")
            return jsonify(output)
        except Exception as e:
            logging.error(f"Error in groupAnalysis: {str(e)}")
            return jsonify({
                "code": 500, 
                "msg": f"Server Error: {str(e)}",
                "data": {}
            })



if __name__ == "__main__":
    cluster_result = kg_clustering()

    print(cluster_result)