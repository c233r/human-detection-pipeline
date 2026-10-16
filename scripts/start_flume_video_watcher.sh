#!/bin/bash
# =============================================================================
# Flume 视频上传服务 — 一键启动脚本
#
# 架构:
#   [Python 桥接] → [flume_watch/*.task] → [Flume Spooldir] → [Kafka video_tasks]
#
# 用法:
#   bash start_flume_video_watcher.sh [start|stop|restart|status]
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

FLUME_WATCH="/data2/caimoyao/flume_watch"
FLUME_DONE="/data2/caimoyao/flume_done"
FLUME_FAILED="/data2/caimoyao/flume_failed"

mkdir -p "$LOG_DIR" "$PID_DIR" "$FLUME_WATCH" "$FLUME_DONE" "$FLUME_FAILED"

ACTION="${1:-start}"

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
# 启动 Python 桥接脚本
# ---------------------------------------------------------------------------

start_bridge() {
    echo -e "${GREEN}▶ 启动 Python 桥接脚本...${NC}"
    stop_by_pid_file "$PID_DIR/flume_bridge.pid" "flume_bridge"

    nohup "$CONDA_PYTHON" "$SCRIPT_DIR/flume_bridge_video_watcher.py" \
        --log-file "$LOG_DIR/flume_bridge_video_watcher.log" \
        > "$LOG_DIR/flume_bridge_stdout.log" 2>&1 &

    echo $! > "$PID_DIR/flume_bridge.pid"
    echo -e "${GREEN}✅ Python 桥接已启动 (PID=$!)${NC}"
    echo -e "   日志: $LOG_DIR/flume_bridge_video_watcher.log"
}

# ---------------------------------------------------------------------------
# 启动 Flume Agent
# ---------------------------------------------------------------------------

start_flume() {
    echo -e "${GREEN}▶ 启动 Flume Agent...${NC}"

    # 查找 flume-ng
    FLUME_NG=""
    if command -v flume-ng &>/dev/null; then
        FLUME_NG="flume-ng"
    elif [ -n "$FLUME_HOME" ] && [ -x "$FLUME_HOME/bin/flume-ng" ]; then
        FLUME_NG="$FLUME_HOME/bin/flume-ng"
    elif [ -x "/opt/apache-flume/bin/flume-ng" ]; then
        FLUME_NG="/opt/apache-flume/bin/flume-ng"
    elif [ -x "/home/caimoyao/apache-flume/bin/flume-ng" ]; then
        FLUME_NG="/home/caimoyao/apache-flume/bin/flume-ng"
    fi

    if [ -z "$FLUME_NG" ]; then
        echo -e "${RED}❌ 未找到 flume-ng 命令${NC}"
        echo -e "   请设置 FLUME_HOME 环境变量，或确认 Flume 已安装"
        echo -e "   安装方法:"
        echo -e "     cd /opt && wget https://archive.apache.org/dist/flume/1.11.0/apache-flume-1.11.0-bin.tar.gz"
        echo -e "     tar xzf apache-flume-1.11.0-bin.tar.gz && mv apache-flume-1.11.0-bin apache-flume"
        echo -e "     export FLUME_HOME=/opt/apache-flume"
        return 1
    fi

    echo -e "   使用 Flume: $FLUME_NG"

    stop_by_pid_file "$PID_DIR/flume_agent.pid" "flume_agent"

    nohup "$FLUME_NG" agent \
        --conf "$SCRIPT_DIR" \
        --conf-file "$SCRIPT_DIR/flume-video-agent.conf" \
        --name videoAgent \
        -Dflume.root.logger=INFO,console \
        > "$LOG_DIR/flume_agent_stdout.log" 2>&1 &

    echo $! > "$PID_DIR/flume_agent.pid"
    echo -e "${GREEN}✅ Flume Agent 已启动 (PID=$!)${NC}"
    echo -e "   日志: $LOG_DIR/flume_agent_stdout.log"
}

# ---------------------------------------------------------------------------
# 停止所有服务
# ---------------------------------------------------------------------------

stop_all() {
    echo -e "${YELLOW}🛑 停止所有服务...${NC}"
    stop_by_pid_file "$PID_DIR/flume_bridge.pid" "flume_bridge"
    stop_by_pid_file "$PID_DIR/flume_agent.pid" "flume_agent"
    echo -e "${GREEN}✅ 所有服务已停止${NC}"
}

# ---------------------------------------------------------------------------
# 状态检查
# ---------------------------------------------------------------------------

check_status() {
    echo -e "${CYAN}服务状态:${NC}"
    for name in flume_bridge flume_agent; do
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
    echo ""
    echo -e "${CYAN}目录状态:${NC}"
    echo -e "  flume_watch:  $(ls "$FLUME_WATCH" 2>/dev/null | wc -l) 个文件"
    echo -e "  flume_done:   $(ls "$FLUME_DONE" 2>/dev/null | wc -l) 个文件"
    echo -e "  flume_failed: $(ls "$FLUME_FAILED" 2>/dev/null | wc -l) 个文件"
}

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

case "$ACTION" in
    start)
        echo -e "${CYAN}============================================${NC}"
        echo -e "${CYAN}   Flume 视频上传服务 - 启动${NC}"
        echo -e "${CYAN}============================================${NC}"
        start_bridge
        echo ""
        start_flume
        echo ""
        echo -e "${GREEN}============================================${NC}"
        echo -e "${GREEN}   ✅ 服务已启动${NC}"
        echo -e "${GREEN}============================================${NC}"
        echo -e "   视频目录: /data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos/videos"
        echo -e "   幂等检查: /data2/caimoyao/api_outputs/temp"
        echo -e "   Kafka:    localhost:9094 → video_tasks"
        echo -e ""
        echo -e "   查看状态: bash $0 status"
        echo -e "   停止服务: bash $0 stop"
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_bridge
        echo ""
        start_flume
        ;;
    status)
        check_status
        ;;
    *)
        echo "用法: $0 [start|stop|restart|status]"
        echo ""
        echo "  start   - 启动 Python 桥接 + Flume Agent"
        echo "  stop    - 停止所有服务"
        echo "  restart - 重启所有服务"
        echo "  status  - 查看服务状态"
        exit 1
        ;;
esac