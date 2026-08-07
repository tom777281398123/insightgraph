#!/bin/bash

# 设置文件目录
directory="/mnt/sda/data/student/liutao/pdfApi/pdf"

# 获取文件列表
files=($directory/*.pdf)

# 文件总数
total_files=${#files[@]}

# 每次上传的文件数
batch_size=10

# 上传文件
for ((i=0; i<total_files; i+=batch_size)); do
  # 获取当前批次的文件
  batch_files="${files[@]:$i:$batch_size}"
  
  # 构造 curl 命令
  curl_command="curl"
  for file in $batch_files; do
    curl_command+=" -F \"file=@$file\""
  done

  # 执行 curl 命令
  eval $curl_command http://127.0.0.1:5000/api/pdf/parse
  echo "Batch $((i / batch_size + 1)) uploaded."
done

echo "All files uploaded."
