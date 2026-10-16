#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh

PID_DIR="/data2/caimoyao/pids"
LOG_DIR="/data2/caimoyao/logs"

mkdir -p "$PID_DIR"
mkdir -p "$LOG_DIR"

PID_FILES=(
    "$PID_DIR/api_service.pid"
    "$PID_DIR/monitor_hold_rope.pid"
    "$PID_DIR/hand_crop.pid"
)

SERVICE_NAMES=(
    "api_service"
    "monitor_hold_rope"
    "hand_crop"
)

start_service() {
    local idx=$1
    local pid_file=${PID_FILES[$idx]}
    local name=${SERVICE_NAMES[$idx]}

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "❌ $name 已经在运行 (PID: $pid)"
            return 1
        else
            rm -f "$pid_file"
        fi
    fi

    case $idx in
        0)
            echo "🚀 启动 $name..."
            cd /data2/caimoyao && \
            conda activate cmy_py310 && \
            python api_service_kafka.py &
            ;;
        1)
            echo "🚀 启动 $name..."
            cd /data2/caimoyao/mmpose/scripts_cmy && \
            conda activate mmpose && \
            python monitor_hold_rope_kafka.py &
            ;;
        2)
            echo "🚀 启动 $name..."
            cd /data2/caimoyao/Jump_rope/classification && \
            conda activate hold_rope_classify && \
            python scripts_cmy/hand_crop_service_kafka.py &
            ;;
    esac

    echo $! > "$pid_file"
    echo "✅ $name 已启动 (PID: $!)"
}

stop_service() {
    local idx=$1
    local pid_file=${PID_FILES[$idx]}
    local name=${SERVICE_NAMES[$idx]}

    if [ ! -f "$pid_file" ]; then
        echo "❌ $name 未运行"
        return 1
    fi

    local pid=$(cat "$pid_file")
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "⚠️ $name PID文件存在但进程已退出，清理PID文件"
        rm -f "$pid_file"
        return 1
    fi

    echo "⏹️ 停止 $name (PID: $pid)..."
    kill "$pid"

    local wait=0
    while [ $wait -lt 10 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
        wait=$((wait + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "⚠️ $name 超时未退出，强制终止"
        kill -9 "$pid"
        sleep 1
    fi

    rm -f "$pid_file"
    echo "✅ $name 已停止"
}

show_status() {
    echo "=========================================="
    echo "           服务状态监控"
    echo "=========================================="

    for i in 0 1 2; do
        local pid_file=${PID_FILES[$i]}
        local name=${SERVICE_NAMES[$i]}

        if [ ! -f "$pid_file" ]; then
            echo "[$name] ❌ 未运行"
            continue
        fi

        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$name] ✅ 运行中 (PID: $pid)"
        else
            echo "[$name] ⚠️ PID文件存在但进程已退出"
        fi
    done

    echo "------------------------------------------"
    echo "日志文件:"
    echo "  - api_service: $LOG_DIR/api_service.log"
    echo "  - monitor_hold_rope: $LOG_DIR/monitor_hold_rope.log"
    echo "  - hand_crop: $LOG_DIR/hand_crop.log"
    echo "=========================================="
}

show_help() {
    echo "用法: $0 [start|stop|restart|status]"
    echo ""
    echo "命令:"
    echo "  start      - 启动所有服务"
    echo "  stop       - 停止所有服务"
    echo "  restart    - 重启所有服务"
    echo "  status     - 查看服务状态"
    echo ""
    echo "单独控制某个服务:"
    echo "  start api_service|monitor_hold_rope|hand_crop"
    echo "  stop api_service|monitor_hold_rope|hand_crop"
    echo ""
    echo "查看日志:"
    echo "  tail -f $LOG_DIR/api_service.log"
    echo "  tail -f $LOG_DIR/monitor_hold_rope.log"
    echo "  tail -f $LOG_DIR/hand_crop.log"
}

get_service_idx() {
    local name=$1
    case $name in
        api_service) echo 0 ;;
        monitor_hold_rope) echo 1 ;;
        hand_crop) echo 2 ;;
        *) echo -1 ;;
    esac
}

case $1 in
    start)
        if [ -n "$2" ]; then
            idx=$(get_service_idx "$2")
            if [ "$idx" -eq -1 ]; then
                echo "❌ 未知服务: $2"
                exit 1
            fi
            start_service $idx
        else
            for i in 0 1 2; do
                start_service $i
            done
            echo ""
            echo "✅ 所有服务已启动"
        fi
        ;;

    stop)
        if [ -n "$2" ]; then
            idx=$(get_service_idx "$2")
            if [ "$idx" -eq -1 ]; then
                echo "❌ 未知服务: $2"
                exit 1
            fi
            stop_service $idx
        else
            for i in 0 1 2; do
                stop_service $i
            done
            echo ""
            echo "✅ 所有服务已停止"
        fi
        ;;

    restart)
        echo "重启所有服务..."
        for i in 0 1 2; do
            stop_service $i
        done
        sleep 2
        for i in 0 1 2; do
            start_service $i
        done
        echo ""
        echo "✅ 所有服务已重启"
        ;;

    status)
        show_status
        ;;

    *)
        show_help
        exit 1
        ;;
esac