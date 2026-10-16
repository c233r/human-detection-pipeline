#!/bin/bash
# 一次性重置 hand_crop_consumers 的 offset 到 earliest
# 在重启 hand_crop_service 前执行一次

# """
# GROUP="hand_crop_consumers"
# TOPIC="hold_rope_keypoints"

# KAFKA_CONSUMER_GROUP = "process_hold_rope_consumers"
# KAFKA_CONSUME_TOPIC = "final_videos"
# """

KAFKA_HOME="/data2/caimoyao/Kafka/kafka"
BOOTSTRAP="localhost:9094"
GROUP="video_task_consumers_test"
TOPIC="video_tasks_test"

echo "正在重置消费组 $GROUP 的 offset 到最早位置（topic: $TOPIC）..."

"$KAFKA_HOME/bin/kafka-consumer-groups.sh" \
  --bootstrap-server "$BOOTSTRAP" \
  --group "$GROUP" \
  --topic "$TOPIC" \
  --reset-offsets --to-earliest --execute

echo "✅ 重置完成！请重启 process_hold_rope_kafka.py"
