# 从modelscope上下载模型
from modelscope import snapshot_download
# model_dir = snapshot_download('Jerry0/m3e-base', cache_dir='./data/supported_llms', revision='master')
# model_dir = snapshot_download('qwen/Qwen2.5-32B-Instruct-GPTQ-Int8', cache_dir='./data/supported_llms', revision='master')
# model_dir = snapshot_download('Qwen/Qwen2.5-32B-Instruct', cache_dir='./data/supported_llms', revision='master')
# model_dir = snapshot_download('qwen/Qwen2.5-72B-Instruct-GPTQ-Int4', cache_dir='./data/supported_llms', revision='master')
# model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./data/supported_llms', revision='master')

model_dir = snapshot_download('tiansz/bert-base-chinese', cache_dir='./data/downloads/training_model', revision='master')
