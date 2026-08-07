"""
Description:
    模拟后端业务系统 (Mock Backend)
    用于接收算法服务  推送的AI论文评审等结果。
"""

import os
import sys

# 确保能找到根目录下的所有模块
sys.path.append('./')

import uvicorn
from fastapi import FastAPI
from log.log import logging

# 导入你的模拟接收器路由
from mock_backend.api.api_simulate_receiver import router_simulate_backend

app = FastAPI(title="模拟后端 (Mock Backend)")

# 记录启动日志
dir_path = os.path.dirname(os.path.abspath(__file__))
logging.info(f'[Mock] Backend current working directory: {dir_path}')

# ---------------------- 注册模拟接口 ---------------------- #
#样例
app.include_router(
    router_simulate_backend, 
    prefix="/backend", 
    tags=["模拟后端-接收算法回传"]
)

@app.get("/")
def index():
    return {"status": "Mock Backend is Running", "port": 10080}

if __name__ == '__main__':
    logging.info('🚀 模拟后端启动中，监听端口: 10080...')
    logging.info('等待算法服务 的回传请求...')
    
    # 按照要求，运行在 10080 端口
    # uvicorn.run(app, host='0.0.0.0', port=10080, log_level="info")
    uvicorn.run(app, host='127.0.0.1', port=10080, log_level="info")