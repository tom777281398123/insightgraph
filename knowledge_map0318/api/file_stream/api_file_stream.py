"""
文件上传下载接口
"""

import os
from flask import Blueprint, request, send_file, jsonify

from log.log import logging

from value.global_value import FILE_SERVIXE_PATH, LOG_RESULT_PATH


# 创建 Blueprint
blueprint_image_up_down = Blueprint('file_handler', __name__)


# # 设置上传文件存储路径
# if not os.path.exists(IMAGE_PATH):
#     os.makedirs(IMAGE_PATH)


if not os.path.exists(FILE_SERVIXE_PATH):
    os.makedirs(FILE_SERVIXE_PATH)


# # 文件上传接口
# @blueprint_image_up_down.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"message": "No file part in the request!"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"message": "No selected file!"}), 400

#     file.save(os.path.join(IMAGE_PATH, file.filename))
#     return jsonify({"message": f"File {file.filename} uploaded successfully!"}), 200


# # 图片下载接口
# @blueprint_image_up_down.route('/download/<filename>', methods=['GET'])
# def download_file(filename):
#     try:
#         path = os.path.join(os.getcwd(), IMAGE_PATH, filename)
#         print(path)
#         if os.path.exists(path):
#             return send_file(path, as_attachment=True)
#         else:
#             return jsonify({"code": 500, "msg": "Image not existing!"})
#     except FileNotFoundError as e:
#         logging.info("Throw an exception: {}".format(e))
#         return jsonify({"code": 500, "msg": "Server Error, throw an exception: {}".format(e)})


# 文件下载接口 用这个函数生成文件服务
@blueprint_image_up_down.route('/download/file/<path:filename>', methods=['GET'])
def download_file_(filename):
    try:
        path = os.path.join(os.getcwd(), FILE_SERVIXE_PATH, filename)
        # print(os.getcwd())
        print(path)
        if os.path.exists(path):
            return send_file(path, as_attachment=True)
        else:
            return jsonify({"code": 500, "msg": "File not existing!"})
    except FileNotFoundError as e:
        logging.info("Throw an exception: {}".format(e))
        return jsonify({"code": 500, "msg": "Server Error, throw an exception: {}".format(e)})
    
# 训练相关的result、log文件下载接口
@blueprint_image_up_down.route('/download/trainLogResultFile/<filename>', methods=['GET'])
def download_file_result_log(filename):
    try:
        path = os.path.join(os.getcwd(), LOG_RESULT_PATH, filename)
        print(path)
        if os.path.exists(path):
            return send_file(path, as_attachment=True)
        else:
            return jsonify({"code": 500, "msg": "File not existing!"})
    except FileNotFoundError as e:
        logging.info("Throw an exception: {}".format(e))
        return jsonify({"code": 500, "msg": "Server Error, throw an exception: {}".format(e)})
