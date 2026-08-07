import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
from log.log import logging
from values.global_config_variable import DOLPHIN_MODEL_PATH

# 先确保结果目录存在
Path("results").mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：加载 Dolphin 模型（只加载一次）
    try:
        from algorithm.dolphinAdapter.engine import init_dolphin_model
        logging.info("加载 Dolphin 模型...")
        init_dolphin_model(DOLPHIN_MODEL_PATH)
        logging.info("Dolphin 模型加载成功。")
    except Exception as e:
        logging.error(f"Dolphin 模型加载失败: {e}")
    yield
    logging.info("服务关闭。")

app = FastAPI(title="Knowledge Map", lifespan=lifespan)

# 新增：连通性测试接口
@app.get("/ping")
def ping():
    return {"msg": "ok"}

# 可选：根路由，浏览器直接访问更直观
@app.get("/")
def root():
    return {"service": "Knowledge Map", "status": "running"}

# 挂载结果文件目录，供 resultUrl 访问
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/data", StaticFiles(directory="data"), name="data")

from api.fileParser import router_file_parser
app.include_router(router_file_parser, prefix="/api", tags=["知识库-文件解析"])

if __name__ == "__main__":
    logging.info("Knowledge_Map 程序启动")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")