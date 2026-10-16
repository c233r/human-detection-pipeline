#!/usr/bin/env python3
"""Kafka 消息桥接层 — 用于视频检测推理系统三个阶段的解耦通信。

阶段架构:
    Stage 1 (api_service)     → detection_done topic → Stage 2 (monitor_hold_rope)
    Stage 2 (monitor_hold_rope) → pose_done topic → Stage 3 (hand_crop_service)

消息 Schema:
    {
        "task_id": str,           # 任务唯一标识
        "video_path": str,        # 视频文件路径
        "labels_dir": str,        # 标签目录路径
        "timestamp": float,       # 消息发送时间戳
        "stage": int,             # 来源阶段: 1=检测, 2=姿态, 3=手部
        "status": str,            # 状态: completed/failed
        "error": str | None,      # 错误信息(如果失败)
    }

Topic 定义:
    - detection_done: Stage 1 完成检测后发送
    - pose_done: Stage 2 完成姿态估计后发送

特性:
    - 幂等性去重: consumer 端维护 task_id set，防止 at-least-once 导致重复处理
    - 自动重试: producer 失败自动重试
    - 优雅关闭: 支持信号处理，确保消息发送完成
"""

import os
import sys
import json
import time
import signal
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError, NoBrokersAvailable
except ImportError:
    raise ImportError("请安装 kafka-python: pip install kafka-python")


# ──────────────────────────────────────────────────────────────────────
# 配置和常量
# ──────────────────────────────────────────────────────────────────────

class KafkaConfig:
    """Kafka 配置"""
    BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    TOPIC_DETECTION_DONE = "detection_done"
    TOPIC_POSE_DONE = "pose_done"
    CONSUMER_GROUP_DETECTION = "monitor_hold_rope_group"
    CONSUMER_GROUP_POSE = "hand_crop_service_group"
    MAX_RETRIES = 3
    RETRY_BACKOFF_MS = 1000
    AUTO_OFFSET_RESET = "latest"
    ENABLE_AUTO_COMMIT = True
    AUTO_COMMIT_INTERVAL_MS = 1000


class TaskStatus:
    """任务状态枚举"""
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStage:
    """任务阶段枚举"""
    DETECTION = 1
    POSE_ESTIMATION = 2
    HAND_CLASSIFICATION = 3


# ──────────────────────────────────────────────────────────────────────
# 消息 Schema 验证
# ──────────────────────────────────────────────────────────────────────

def validate_task_message(msg: Dict) -> bool:
    """验证任务消息格式"""
    required = ["task_id", "video_path", "labels_dir", "timestamp", "stage", "status"]
    for key in required:
        if key not in msg:
            return False
    if msg["stage"] not in (1, 2, 3):
        return False
    if msg["status"] not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        return False
    return True


def serialize_message(msg: Dict) -> bytes:
    """序列化消息为 JSON bytes"""
    return json.dumps(msg, ensure_ascii=False).encode("utf-8")


def deserialize_message(data: bytes) -> Optional[Dict]:
    """反序列化消息"""
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


# ──────────────────────────────────────────────────────────────────────
# Kafka Producer 封装
# ──────────────────────────────────────────────────────────────────────

class TaskProducer:
    """任务消息生产者"""

    def __init__(self, bootstrap_servers: str = KafkaConfig.BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self._running = False

    def connect(self):
        """连接 Kafka"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                retries=KafkaConfig.MAX_RETRIES,
                retry_backoff_ms=KafkaConfig.RETRY_BACKOFF_MS,
                acks='all',
                linger_ms=50,
            )
            self._running = True
            print(f"✅ Kafka Producer 已连接: {self.bootstrap_servers}")
        except NoBrokersAvailable:
            print(f"❌ 无法连接 Kafka: {self.bootstrap_servers}")
            raise

    def send_detection_done(self, task_id: str, video_path: str, labels_dir: str,
                            error: Optional[str] = None):
        """发送 Stage 1 检测完成消息"""
        if not self._running or not self.producer:
            raise RuntimeError("Producer 未连接")

        msg = {
            "task_id": task_id,
            "video_path": video_path,
            "labels_dir": labels_dir,
            "timestamp": time.time(),
            "stage": TaskStage.DETECTION,
            "status": TaskStatus.FAILED if error else TaskStatus.COMPLETED,
            "error": error,
        }

        future = self.producer.send(
            KafkaConfig.TOPIC_DETECTION_DONE,
            value=msg,
            key=task_id.encode("utf-8"),
        )

        try:
            future.get(timeout=10)
            print(f"📤 发送 detection_done: task_id={task_id}, status={msg['status']}")
        except KafkaError as e:
            print(f"❌ 发送失败: {e}")
            raise

    def send_pose_done(self, task_id: str, video_path: str, labels_dir: str,
                       error: Optional[str] = None):
        """发送 Stage 2 姿态估计完成消息"""
        if not self._running or not self.producer:
            raise RuntimeError("Producer 未连接")

        msg = {
            "task_id": task_id,
            "video_path": video_path,
            "labels_dir": labels_dir,
            "timestamp": time.time(),
            "stage": TaskStage.POSE_ESTIMATION,
            "status": TaskStatus.FAILED if error else TaskStatus.COMPLETED,
            "error": error,
        }

        future = self.producer.send(
            KafkaConfig.TOPIC_POSE_DONE,
            value=msg,
            key=task_id.encode("utf-8"),
        )

        try:
            future.get(timeout=10)
            print(f"📤 发送 pose_done: task_id={task_id}, status={msg['status']}")
        except KafkaError as e:
            print(f"❌ 发送失败: {e}")
            raise

    def flush(self):
        """刷新所有待发送消息"""
        if self.producer:
            self.producer.flush()

    def close(self):
        """关闭连接"""
        self._running = False
        if self.producer:
            self.producer.close()
            print("✅ Kafka Producer 已关闭")


# ──────────────────────────────────────────────────────────────────────
# Kafka Consumer 封装（带幂等去重）
# ──────────────────────────────────────────────────────────────────────

class TaskConsumer:
    """任务消息消费者（带幂等去重）"""

    def __init__(self, topic: str, group_id: str,
                 bootstrap_servers: str = KafkaConfig.BOOTSTRAP_SERVERS,
                 handler: Optional[Callable[[Dict], bool]] = None):
        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.handler = handler
        self.consumer = None
        self._running = False
        self._processed_tasks: set = set()

    def connect(self):
        """连接 Kafka"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset=KafkaConfig.AUTO_OFFSET_RESET,
                enable_auto_commit=KafkaConfig.ENABLE_AUTO_COMMIT,
                auto_commit_interval_ms=KafkaConfig.AUTO_COMMIT_INTERVAL_MS,
                max_poll_records=10,
                consumer_timeout_ms=1000,
            )
            self._running = True
            print(f"✅ Kafka Consumer 已连接: topic={self.topic}, group={self.group_id}")
        except NoBrokersAvailable:
            print(f"❌ 无法连接 Kafka: {self.bootstrap_servers}")
            raise

    def _is_duplicate(self, task_id: str) -> bool:
        """检查是否重复处理"""
        if task_id in self._processed_tasks:
            return True
        self._processed_tasks.add(task_id)
        # 限制内存使用，保留最近 10000 个 task_id
        if len(self._processed_tasks) > 10000:
            self._processed_tasks = set(list(self._processed_tasks)[-5000:])
        return False

    def poll_once(self) -> List[Dict]:
        """轮询一次消息"""
        if not self._running or not self.consumer:
            return []

        messages = []
        try:
            for msg in self.consumer:
                value = msg.value
                if not validate_task_message(value):
                    print(f"⚠ 消息格式无效: {value}")
                    continue
                if self._is_duplicate(value["task_id"]):
                    print(f"⏭ 跳过重复任务: {value['task_id']}")
                    continue
                messages.append(value)
        except Exception as e:
            print(f"❌ 轮询异常: {e}")

        return messages

    def run_loop(self):
        """运行消费循环"""
        if not self._running or not self.consumer:
            raise RuntimeError("Consumer 未连接")

        print(f"🔄 开始消费 {self.topic}...")
        while self._running:
            try:
                messages = self.poll_once()
                for msg in messages:
                    task_id = msg["task_id"]
                    print(f"📥 收到消息: task_id={task_id}, stage={msg['stage']}, status={msg['status']}")

                    if self.handler:
                        try:
                            success = self.handler(msg)
                            if success:
                                print(f"✅ 处理完成: {task_id}")
                            else:
                                print(f"❌ 处理失败: {task_id}")
                        except Exception as e:
                            print(f"❌ 处理异常 {task_id}: {e}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 消费循环异常: {e}")
                time.sleep(5)

        print(f"🛑 消费循环已停止")

    def close(self):
        """关闭连接"""
        self._running = False
        if self.consumer:
            self.consumer.close()
            print("✅ Kafka Consumer 已关闭")


# ──────────────────────────────────────────────────────────────────────
# 桥接层工具函数（便于三个服务快速接入）
# ──────────────────────────────────────────────────────────────────────

def create_detection_producer() -> TaskProducer:
    """创建 Stage 1 检测完成消息生产者"""
    producer = TaskProducer()
    producer.connect()
    return producer


def create_pose_producer() -> TaskProducer:
    """创建 Stage 2 姿态完成消息生产者"""
    producer = TaskProducer()
    producer.connect()
    return producer


def create_monitor_consumer(handler: Callable[[Dict], bool]) -> TaskConsumer:
    """创建 Stage 2 的消费者（订阅 detection_done）"""
    consumer = TaskConsumer(
        topic=KafkaConfig.TOPIC_DETECTION_DONE,
        group_id=KafkaConfig.CONSUMER_GROUP_DETECTION,
        handler=handler,
    )
    consumer.connect()
    return consumer


def create_hand_crop_consumer(handler: Callable[[Dict], bool]) -> TaskConsumer:
    """创建 Stage 3 的消费者（订阅 pose_done）"""
    consumer = TaskConsumer(
        topic=KafkaConfig.TOPIC_POSE_DONE,
        group_id=KafkaConfig.CONSUMER_GROUP_POSE,
        handler=handler,
    )
    consumer.connect()
    return consumer


# ──────────────────────────────────────────────────────────────────────
# 示例用法
# ──────────────────────────────────────────────────────────────────────

def example_producer():
    """示例：发送任务消息"""
    producer = create_detection_producer()
    try:
        producer.send_detection_done(
            task_id="test_task_001",
            video_path="/data2/caimoyao/api_outputs/temp/test_task_001/input.mp4",
            labels_dir="/data2/caimoyao/api_outputs/temp/test_task_001/labels",
        )
    finally:
        producer.flush()
        producer.close()


def example_consumer():
    """示例：消费任务消息"""
    def handle_task(msg: Dict) -> bool:
        task_id = msg["task_id"]
        video_path = msg["video_path"]
        labels_dir = msg["labels_dir"]
        print(f"处理任务: {task_id}")
        print(f"视频: {video_path}")
        print(f"标签: {labels_dir}")
        return True

    consumer = create_monitor_consumer(handle_task)
    try:
        consumer.run_loop()
    except KeyboardInterrupt:
        consumer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kafka Bridge 工具")
    parser.add_argument("--mode", choices=["produce", "consume"], required=True,
                        help="运行模式: produce=发送示例消息, consume=消费消息")
    parser.add_argument("--topic", choices=["detection", "pose"], default="detection",
                        help="Topic 类型")
    args = parser.parse_args()

    if args.mode == "produce":
        example_producer()
    elif args.mode == "consume":
        example_consumer()
