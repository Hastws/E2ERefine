#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=llmws
PYVER=3.10

# 1) 创建并激活 conda 环境
conda create -n "$ENV_NAME" python=$PYVER -y
# 兼容 bash/zsh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 2) 升级 pip
python -m pip install --upgrade pip

# 3) 安装 PyTorch（按平台选择）
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
  # macOS：CPU/MPS 版本（无 CUDA）
  pip install torch==2.3.0 torchvision==0.18.0
else
  # Linux/Windows：可选 CUDA 12.1 预编译轮子
  pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0 torchvision==0.18.0
fi

# 4) 安装其余依赖
pip install -r requirements.txt

# 5) 简单自检
python - <<'PY'
import torch, platform, sys
print("torch:", torch.__version__)
print("python:", sys.version.split()[0], "platform:", platform.platform())
print("cuda available:", torch.cuda.is_available())
print("mps available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo "Done. Use: conda activate $ENV_NAME"
