#!/bin/bash
# Ray 伪集群启动脚本 - 多服务共享同一个 Ray 集群
# 伪节点模式：Head[GPU 5] + Worker[GPU 4]，通过 resources 标签实现 GPU 隔离
# API 服务 -> gpu_5_node (GPU 5)
# Monitor + HandCrop 服务 -> gpu_4_node (GPU 4)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 添加 conda 环境的 ray 到 PATH
export PATH="/home/caimoyao/.conda/envs/api_service/bin:$PATH"

# 强制 Ray 使用 localhost
export RAY_NODE_IP=127.0.0.1

RAY_ADDRESS="auto"
HEAD_PORT=6380
LOG_DIR="/data2/caimoyao/logs"
RAY_TMP_DIR="/data2/caimoyao/ray"
RAY_SPILL_DIR="/data2/caimoyao/ray_spill"
OBJECT_STORE_MEMORY=$((200 * 1024 ** 3))  # 200GB

# ==================== CPU 资源限制配置 ====================
# 例如：限制整个伪集群最多使用 32 核 CPU
# 分配建议：API 服务 (GPU 7) 计算量稍大，分 20 核；Monitor + HandCrop (GPU 4) 分 12 核
HEAD_NUM_CPUS=20      # 给 GPU 7 伪节点的 CPU 核心数
WORKER_NUM_CPUS=12    # 给 GPU 4 伪节点的 CPU 核心数
TOTAL_CPUS=$((HEAD_NUM_CPUS + WORKER_NUM_CPUS))
# ==========================================================

# 创建必要目录
mkdir -p "$LOG_DIR" "$RAY_TMP_DIR" "$RAY_SPILL_DIR"

# 设置 CUDA 库路径
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

API_ENV="/home/caimoyao/.conda/envs/api_service/bin/python"
CONDA_ENV="api_service"

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Ray 伪集群启动脚本 (伪节点模式)${NC}"
echo -e "${CYAN}============================================${NC}"
echo -e "  CPU 总限制      -> ${TOTAL_CPUS} 核 (Head: ${HEAD_NUM_CPUS}, Worker: ${WORKER_NUM_CPUS})"
echo -e "  GPU 5 (Head)    -> API 服务 (quota: 5)"
echo -e "  GPU 4 (Worker)  -> Monitor + HandCrop 服务 (quota: 2)"
echo -e "  Object Store: 200GB, Spill: $RAY_SPILL_DIR"
echo ""

# ========== Step 1: 彻底清理现有 Ray 集群 ==========
echo -e "${YELLOW}[1/5] 彻底清理现有 Ray 集群...${NC}"
ray stop --force 2>/dev/null
sleep 2

rm -rf "$RAY_TMP_DIR/session_latest" 2>/dev/null
rm -rf "$RAY_TMP_DIR/ray_*" 2>/dev/null
sleep 1

echo -e "${GREEN}✅ 清理完成${NC}"

# ========== Step 2: 启动 Ray Head 节点 (GPU 7) ==========
echo -e "${YELLOW}[2/5] 启动 Ray Head 节点 (GPU: 5, CPU: ${HEAD_NUM_CPUS}, resource: gpu_5_node)...${NC}"
echo -e "    绑定地址: 127.0.0.1:$HEAD_PORT"

CUDA_VISIBLE_DEVICES=6 ray start --head \
    --node-ip-address=127.0.0.1 \
    --port=$HEAD_PORT \
    --include-dashboard=true \
    --dashboard-host=127.0.0.1 \
    --dashboard-port=8265 \
    --num-gpus=1 \
    --num-cpus=$HEAD_NUM_CPUS \
    --min-worker-port=15000 \
    --max-worker-port=16999 \
    --object-store-memory=$OBJECT_STORE_MEMORY \
    --object-spilling-directory="$RAY_SPILL_DIR" \
    --temp-dir="$RAY_TMP_DIR" \
    --system-config='{"max_io_workers":4}' \
    --resources='{"gpu_5_node": 5}' \
    2>&1 | tee "$LOG_DIR/ray_head.log"

echo -e "${GREEN}✅ Ray Head 已启动 (port=$HEAD_PORT, CPU=${HEAD_NUM_CPUS}, GPU 5 伪节点)${NC}"

# 等待 Head 完全就绪
echo -e "${YELLOW}    等待 Head 节点 GCS 就绪...${NC}"
sleep 3

# ========== Step 3: 启动 Ray Worker 伪节点 (GPU 4) ==========
echo -e "${YELLOW}[3/5] 启动 Ray Worker 伪节点 (GPU: 4, CPU: ${WORKER_NUM_CPUS}, resource: gpu_4_node)...${NC}"
echo -e "    连接地址: 127.0.0.1:$HEAD_PORT"

CUDA_VISIBLE_DEVICES=6 ray start \
    --node-ip-address=127.0.0.1 \
    --address="0.0.0.0:$HEAD_PORT" \
    --num-gpus=1 \
    --num-cpus=$WORKER_NUM_CPUS \
    --min-worker-port=17000 \
    --max-worker-port=18999 \
    --object-store-memory=$OBJECT_STORE_MEMORY \
    --object-spilling-directory="$RAY_SPILL_DIR" \
    --temp-dir="$RAY_TMP_DIR" \
    --resources='{"gpu_4_node": 4}' \
    2>&1 | tee "$LOG_DIR/ray_worker_gpu4.log"

echo -e "${GREEN}✅ Ray Worker 伪节点已启动 (CPU=${WORKER_NUM_CPUS}, GPU 4 伪节点)${NC}"

# ========== Step 4: 等待 Ray 集群就绪 ==========
echo -e "${YELLOW}[4/5] 等待 Ray 集群就绪...${NC}"
for i in $(seq 1 30); do
    if ray status 2>/dev/null | grep -q "Active:"; then
        echo -e "${GREEN}✅ Ray 集群就绪${NC}"
        break
    fi
    sleep 1
    echo -ne "\r   等待中... ($i/30)"
done
echo ""

# ========== Step 5: 显示集群状态 ==========
echo -e "${YELLOW}[5/5] 显示集群状态${NC}"
ray status

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Ray 伪集群已就绪！${NC}"
echo -e "${GREEN}  Head: localhost:$HEAD_PORT${NC}"
echo -e "${GREEN}  Dashboard: http://localhost:8265${NC}"
echo -e ""
echo -e "${CYAN}  资源配额:${NC}"
echo -e "    CPU 总配额:  ${TOTAL_CPUS} 核"
echo -e "    gpu_5_node: CPU=${HEAD_NUM_CPUS}, GPU=1 (API 服务独占, quota=5)"
echo -e "    gpu_4_node: CPU=${WORKER_NUM_CPUS}, GPU=1 (Monitor + HandCrop 共享, quota=2)"
echo -e "${GREEN}============================================${NC}"
