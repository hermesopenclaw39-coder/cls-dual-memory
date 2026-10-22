#!/bin/bash
# 环境搭建（Qwen2.5-7B 4-bit QLoRA）
# 用法：bash setup.sh
# 假设 torch 已随镜像安装（AutoDL 镜像自带 torch 2.5.1+cu124）。
set -e

# 大陆网络：默认走 hf-mirror；如在海外可 unset 或改成官方地址
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 模型缓存目录：默认用户缓存；AutoDL 建议指到数据盘
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== 1/2 安装依赖（transformers 须 <5.0 以兼容 torch 2.5.1）==="
python -m pip install -i "${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
  'transformers<5.0' peft bitsandbytes accelerate huggingface_hub

echo "=== 2/2 下载 Qwen2.5-7B-Instruct（约 15G，缓存到 $HF_HOME）==="
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

echo "=== 验证 ==="
python -c "import torch, transformers, peft, bitsandbytes; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available()); print('transformers', transformers.__version__, 'peft', peft.__version__, 'bitsandbytes', bitsandbytes.__version__); print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')"
echo "=== 完成 ==="
