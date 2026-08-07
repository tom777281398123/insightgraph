# 使用 NVIDIA 官方的 PyTorch 镜像 
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# FROM continuumio/miniconda3
# FROM ubuntu:22.04

# 设置工作目录
WORKDIR /app

# 将依赖文件复制到容器中
COPY requirements.txt .

# 将依赖清单文件复制到镜像中
COPY environment.yml .

# 在镜像里创建独立的conda环境
# RUN conda env create -y -f environment.yml && \
#     conda clean --all -f -y

# 安装依赖 (使用国内镜像源加速)
# 如果 pip 很慢，建议配置阿里云或清华源
# RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 将代码复制到容器中
# 将当前目录下除了 .dockerignore 里的文件全部复制
COPY --chown=appuser:appuser . ./

# 暴露端口 (假设你的服务需要端口，比如 8000)
# 如果没有 Web 服务，这行可以删掉
EXPOSE 8000

# 设置启动命令
CMD ["python", "app.py"]