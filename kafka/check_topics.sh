#!/bin/bash

PYTHON_SCRIPT="/data2/caimoyao/check_partitions.py"
BROKER="localhost:9094"
PYTHON="/home/caimoyao/.conda/envs/api_service/bin/python"

echo "=============================================="
echo "          Kafka Topic 分区状态检查"
echo "=============================================="
echo "Broker: $BROKER"
echo "Python: $PYTHON"
echo "=============================================="

for entry in \
    "video_tasks:video_task_consumers" \
    "video_results:hold_rope_consumers" \
    "hold_rope_keypoints:hand_crop_consumers" \
    "final_videos:process_hold_rope_consumers" \
    "video_tasks_test:video_task_consumers_test" \
    "video_results_test:hold_rope_consumers_test"\
    "hold_rope_keypoints_test:hand_crop_consumers_test"
do
    topic="${entry%%:*}"
    group="${entry##*:}"
    echo ""
    echo "──────────────────────────────────────────────"
    echo "📌 Topic: $topic (消费组: $group)"
    echo "──────────────────────────────────────────────"
    $PYTHON "$PYTHON_SCRIPT" -t "$topic" -g "$group" -b "$BROKER"
done

echo ""
echo "=============================================="
echo "                    检查完成"
echo "=============================================="