#!/bin/bash
# 启动 Ray 集群头节点（三个服务共享同一个集群，实现 ObjectRef 零拷贝通信）
# 使用方式: bash ray_start_head.sh

RAY_BIN="/home/caimoyao/.conda/envs/api_service/bin/ray"

# 清理旧集群
$RAY_BIN stop --force

# 清理残留 session
rm -rf /data2/caimoyao/ray/session_* /data2/caimoyao/ray_spill/*

# 启动头节点
# GPU: 总共 1 张卡 (7)
# object_store_memory: 60GB（支持多服务并发）
# object_spilling: 溢出到 /data2/caimoyao/ray_spill
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=7 $RAY_BIN start --head \
    --port=6380 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=1 \
    --object-store-memory=128849018880 \
    --object-spilling-directory=/data2/caimoyao/ray_spill \
    --temp-dir=/data2/caimoyao/ray \
    --system-config='{"max_io_workers":4}'

echo ""
echo "============================================================"
echo "   Ray 集群头节点已启动"
echo "   连接地址: localhost:6380"
echo "   Dashboard: http://localhost:8265"
echo "   可用 GPU: 1 (卡号 7)"
echo "============================================================"
echo ""
echo "现在可以在其他终端启动三个服务:"
echo "   python api_service_kafka_ray.py"
echo "   python mmpose/scripts_cmy/monitor_hold_rope_kafka_stream_ray.py"
echo "   python Jump_rope/classification/scripts_cmy/hand_crop_service_kafka_onnx_ray.py"
