import os
import sys
from fastapi import APIRouter, Request
# 尝试导入自定义log，如果失败则使用标准logging
try:
    from log.log import logging
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)

router_simulate_backend = APIRouter()

# 1. 模拟论文评审的【异步回调接口】
# 对应算法推送地址：http://192.166.32.103:10080/backend/research-paper-review/paperReviewCallback
@router_simulate_backend.post("/research-paper-review/paperReviewCallback")
async def paper_review_callback(request: Request):
    try:
        data = await request.json()
        results = (data or {}).get("data", {}).get("results") or []
        logging.info("="*50)
        logging.info("🚀 [Mock Backend] 收到论文评审异步回调结果！")
        logging.info(f"taskId = {(data or {}).get('data', {}).get('taskId')}")
        logging.info(f"result_count =  { len(results) }")
        logging.info("="*50)
        
        # 核心：在控制台（10080黑窗口）打印出回传的结果
        # logging.info("="*50)
        # logging.info("🚀 [Mock Backend] 收到论文评审异步回调结果！")
        # logging.info(f"📌 RequestId: {data.get('data', {}).get('requestId')}")
        # logging.info(f"📝 评审结果预览 (前5000字): {data.get('data', {}).get('result')[:5000]}...")
        # logging.info("="*50)
        
        return {"code": 200, "message": "后端已成功收到回调数据"}
    except Exception as e:
        logging.error(f"❌ 回调接收失败: {str(e)}")
        return {"code": 500, "message": str(e)}

# 2. 之前的通用结果回传接口（可选保留）
@router_simulate_backend.post("/get_development_context_data")
async def get_development_context_data(request: Request):
    data = await request.json()
    logging.info(f"🚩 [Mock Backend] 收到通用回传数据: {data}")
    return {"code": 200, "msg": "数据接收成功."}
