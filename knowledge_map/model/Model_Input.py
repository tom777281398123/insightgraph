# 从modelscope上下载模型
from modelscope import snapshot_download
# model_dir = snapshot_download('Jerry0/m3e-base', cache_dir='./', revision='master')

model_dir = snapshot_download('Qwen/Qwen3-32B', cache_dir='./', revision='master')
