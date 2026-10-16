#!/bin/bash
# =============================================================================
# 视频目录监控服务 启动脚本
#
# 两种模式:
#   1. 纯 Python 模式 (推荐): Python 脚本直接监听 + 幂等 + 发送 Kafka
#   2. Flume 模式: Python 桥接 + Flume spooldir + Flume Kafka sink
#
# 使用方法:
#   bash start_video_watcher.sh [python|flume|both]
#
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_PYTHON="/home/caimoyao/.conda/envs/api_service/bin/python"
LOG_DIR="/data2/caimoyao/logs"
PID_DIR="/data2/caimoyao/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

MODE="${1:-python}"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   视频目录监控服务启动${NC}"
echo -e "${CYAN}   模式: $MODE${NC}"
echo -e "${CYAN}========================================${NC}"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

stop_by_pid_file() {
    local pid_file="$1"
    local name="$2"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}🛑 停止 $name (PID=$pid)...${NC}"
            kill "$pid" 2>/dev/null || true
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

# ---------------------------------------------------------------------------
# 模式 1: 纯 Python (直接发送到 Kafka)
# ---------------------------------------------------------------------------

start_python_mode() {
    echo -e "${GREEN}▶ 启动 Python 监控服务...${NC}"

    stop_by_pid_file "$PID_DIR/video_watcher.pid" "video_watcher"

    nohup "$CONDA_PYTHON" "$SCRIPT_DIR/watch_video_directory.py" \
        --log-file "$LOG_DIR/video_upload_watcher.log" \
        > "$LOG_DIR/video_watcher_stdout.log" 2>&1 &

    echo $! > "$PID_DIR/video_watcher.pid"
    echo -e "${GREEN}✅ Python 监控服务已启动 (PID=$!)${NC}"
    echo -e "   日志: $LOG_DIR/video_upload_watcher.log"
    echo -e "   停止: bash $0 stop-python"
}

# ---------------------------------------------------------------------------
# 模式 2: Flume 模式 (桥接 + Flume Agent)
# ---------------------------------------------------------------------------

start_flume_mode() {
    echo -e "${GREEN}▶ 启动 Flume 桥接服务...${NC}"

    # 创建 Flume 相关目录
    mkdir -p /data2/caimoyao/flume_watch
    mkdir -p /data2/caimoyao/flume_done
    mkdir -p /data2/caimoyao/flume_failed

    # 启动 Python 桥接脚本
    stop_by_pid_file "$PID_DIR/flume_bridge.pid" "flume_bridge"

    nohup "$CONDA_PYTHON" "$SCRIPT_DIR/flume_bridge_video_watcher.py" \
        --log-file "$LOG_DIR/flume_bridge_video_watcher.log" \
        > "$LOG_DIR/flume_bridge_stdout.log" 2>&1 &

    echo $! > "$PID_DIR/flume_bridge.pid"
    echo -e "${GREEN}✅ Flume 桥接脚本已启动 (PID=$!)${NC}"

    # 检查 Flume 是否可用
    if command -v flume-ng &>/dev/null; then
        stop_by_pid_file "$PID_DIR/flume_agent.pid" "flume_agent"

        echo -e "${GREEN}▶ 启动 Flume Agent...${NC}"

        nohup flume-ng agent \
            --conf "$SCRIPT_DIR" \
            --conf-file "$SCRIPT_DIR/flume-video-agent.conf" \
            --name videoAgent \
            -Dflume.root.logger=INFO,console \
            > "$LOG_DIR/flume_agent_stdout.log" 2>&1 &

        echo $! > "$PID_DIR/flume_agent.pid"
        echo -e "${GREEN}✅ Flume Agent 已启动 (PID=$!)${NC}"
        echo -e "   Flume 日志: $LOG_DIR/flume_agent_stdout.log"
    else
        echo -e "${YELLOW}⚠ flume-ng 未在 PATH 中找到，跳过 Flume Agent 启动${NC}"
        echo -e "${YELLOW}   请安装 Flume 后手动启动:${NC}"
        echo -e "   ${CYAN}flume-ng agent --conf $SCRIPT_DIR --conf-file $SCRIPT_DIR/flume-video-agent.conf --name videoAgent${NC}"
    fi

    echo -e "   桥接日志: $LOG_DIR/flume_bridge_video_watcher.log"
    echo -e "   停止: bash $0 stop-flume"
}

# ---------------------------------------------------------------------------
# 停止命令
# ---------------------------------------------------------------------------

stop_all() {
    echo -e "${YELLOW}🛑 停止所有服务...${NC}"
    stop_by_pid_file "$PID_DIR/video_watcher.pid" "video_watcher"
    stop_by_pid_file "$PID_DIR/flume_bridge.pid" "flume_bridge"
    stop_by_pid_file "$PID_DIR/flume_agent.pid" "flume_agent"
    echo -e "${GREEN}✅ 所有服务已停止${NC}"
}

stop_python() {
    stop_by_pid_file "$PID_DIR/video_watcher.pid" "video_watcher"
}

stop_flume() {
    stop_by_pid_file "$PID_DIR/flume_bridge.pid" "flume_bridge"
    stop_by_pid_file "$PID_DIR/flume_agent.pid" "flume_agent"
}

# ---------------------------------------------------------------------------
# 状态检查
# ---------------------------------------------------------------------------

check_status() {
    echo -e "${CYAN}服务状态:${NC}"

    for name in video_watcher flume_bridge flume_agent; do
        pid_file="$PID_DIR/${name}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${GREEN}● $name${NC} (PID=$pid) 运行中"
            else
                echo -e "  ${RED}● $name${NC} (PID=$pid) 已停止"
            fi
        else
            echo -e "  ${YELLOW}○ $name${NC} 未启动"
        fi
    done
}

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

case "$MODE" in
    python)
        start_python_mode
        ;;
    flume)
        start_flume_mode
        ;;
    both)
        start_python_mode
        echo ""
        start_flume_mode
        ;;
    stop)
        stop_all
        ;;
    stop-python)
        stop_python
        ;;
    stop-flume)
        stop_flume
        ;;
    status)
        check_status
        ;;
    restart)
        stop_all
        sleep 2
        start_python_mode
        ;;
    *)
        echo -e "用法: $0 [python|flume|both|stop|stop-python|stop-flume|status|restart]"
        echo ""
        echo "  python      - 纯 Python 模式 (直接发送 Kafka)"
        echo "  flume       - Flume 模式 (桥接 + Flume Agent)"
        echo "  both        - 同时启动两种模式"
        echo "  stop        - 停止所有服务"
        echo "  stop-python - 停止 Python 服务"
        echo "  stop-flume  - 停止 Flume 服务"
        echo "  status      - 查看服务状态"
        echo "  restart     - 重启 Python 服务"
        exit 1
        ;;
esac