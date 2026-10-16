#!/bin/bash
# EdgeCrafter API 服务启动脚本

echo "========================================"
echo "  EdgeCrafter API 服务启动脚本"
echo "========================================"

# 切换到脚本所在目录
cd /data2/caimoyao

# 激活 conda 环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null
conda activate cmy_py310

# 检查是否已安装依赖
echo ""
echo "[1/4] 检查依赖..."
if python -c "import fastapi" 2>/dev/null; then
    echo "✅ FastAPI 已安装"
else
    echo "📦 安装 FastAPI 依赖..."
    pip install fastapi uvicorn python-multipart aiofiles pydantic
fi

# 检查 CUDA
echo ""
echo "[2/4] 检查 CUDA..."
if python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
else
    echo "⚠️ PyTorch 未安装或 CUDA 不可用"
fi

# 创建输出目录
echo ""
echo "[3/4] 创建输出目录..."
mkdir -p /data2/caimoyao/api_outputs/{results,uploads,logs}
echo "✅ 输出目录: /data2/caimoyao/api_outputs"

# 启动服务
echo ""
echo "[4/4] 启动 API 服务..."
echo "========================================"
echo "📖 API 文档: http://localhost:8000/docs"
echo "📖 ReDoc: http://localhost:8000/redoc"
echo "========================================"
echo ""

# 启动服务（支持热重载）
python api_service.py
