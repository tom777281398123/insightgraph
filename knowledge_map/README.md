### 项目介绍

### conda环境

```shell
conda activate baike
```

### 工程目录

```shell
.
├── algorithm   第3层 算法封装
├── api         第2层 api封装
├── app.py      第1层 app注册路由
├── config      存放config相关
├── data        存放数据
├── log         存放log相关
├── mock_backend    模拟后端
├── mock_backend_app.py
├── model       存放models
├── README.md   
└── values      存放全局变量等
```

### 大模型启动 

```shell
python -m vllm.entrypoints.openai.api_server --port 10110 --served-model-name Qwen3-32B --model /mnt/sda/data/caoshen/code/encyclopedia_project/data/models/Qwen/Qwen3-32B --tensor-parallel-size 2 --dtype auto --gpu-memory-utilization 0.8 --max-model-len 32768 --disable-custom-all-reduce --enforce-eager
```



