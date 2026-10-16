"""
压缩服务（Kafka）
订阅 final_videos topic，收到任务后对 api_outputs/temp 中的对应文件夹进行 7z 压缩
压缩完成后删除原文件夹
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
import shutil
from pathlib import Path

from confluent_kafka import Consumer, TopicPartition

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/compress_service.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("compress_service")


class Config:
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_CONSUME_TOPIC = "final_videos"
    KAFKA_CONSUMER_GROUP = "compress_consumers"

    TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")
    COMPRESS_OUTPUT_DIR = Path("/data2/caimoyao/api_outputs/compressed")


config = Config()


def compress_folder(task_id: str, source_dir: Path, output_path: Path) -> bool:
    """使用 7z 压缩文件夹。"""
    if not source_dir.exists():
        logger.error(f"任务 {task_id}: 源目录不存在: {source_dir}")
        return False

    parent = output_path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "7z", "a", "-t7z", "-mx=5", "-bb1",
        str(output_path),
        str(source_dir),
    ]

    logger.info(f"任务 {task_id}: 开始压缩: {source_dir} -> {output_path}")
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        dt = (time.perf_counter() - t0) * 1000

        if result.returncode == 0:
            logger.info(f"任务 {task_id}: 压缩成功, 耗时 {dt:.1f}ms")
            return True
        else:
            logger.error(f"任务 {task_id}: 压缩失败 (code={result.returncode}): {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"任务 {task_id}: 压缩超时")
        return False
    except Exception as e:
        logger.error(f"任务 {task_id}: 压缩异常: {e}")
        return False


def delete_folder(task_id: str, folder: Path) -> bool:
    """删除文件夹。"""
    try:
        shutil.rmtree(folder)
        logger.info(f"任务 {task_id}: 已删除原文件夹: {folder}")
        return True
    except Exception as e:
        logger.error(f"任务 {task_id}: 删除文件夹失败: {e}")
        return False


def process_task(task_data: dict, logger):
    """处理单个压缩任务。"""
    task_id = task_data.get("task_id", "")
    status = task_data.get("status", "")

    if status != "completed":
        logger.debug(f"跳过非 completed 消息: {task_id}")
        return True

    logger.info(f"收到压缩任务: {task_id}")

    source_dir = config.TEMP_DIR / task_id
    if not source_dir.exists():
        logger.warning(f"任务 {task_id}: 源目录不存在: {source_dir}, 跳过")
        return True

    output_path = config.COMPRESS_OUTPUT_DIR / f"{task_id}.7z"

    try:
        if compress_folder(task_id, source_dir, output_path):
            delete_folder(task_id, source_dir)
            logger.info(f"任务 {task_id}: 压缩+删除完成")
            return True
        else:
            logger.error(f"任务 {task_id}: 压缩失败，保留原文件")
            return False
    except Exception as e:
        logger.error(f"任务 {task_id}: 处理异常: {e}")
        return False


def create_consumer():
    """创建 Kafka Consumer。"""
    return Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "group.id": config.KAFKA_CONSUMER_GROUP,
        "group.instance.id": "compress-service-1",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.poll.interval.ms": 3600000,
        "session.timeout.ms": 180000,
        "heartbeat.interval.ms": 3000,
        "partition.assignment.strategy": "cooperative-sticky",
        "broker.address.family": "v4",
        "queued.min.messages": 1,
    })


def main():
    print("=" * 60)
    print("   压缩服务 (Kafka)")
    print("=" * 60)
    print(f"🌐 Kafka: {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"📥 消费: {config.KAFKA_CONSUME_TOPIC}")
    print(f"📂 源目录: {config.TEMP_DIR}")
    print(f"📦 输出目录: {config.COMPRESS_OUTPUT_DIR}")
    print("=" * 60)

    config.COMPRESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    consumer = create_consumer()
    consumer.subscribe([config.KAFKA_CONSUME_TOPIC])
    logger.info(f"✅ Consumer 已连接")

    shutdown_event = False

    def handle_signal(signum, frame):
        nonlocal shutdown_event
        signal_name = signal.Signals(signum).name
        logger.info(f"🛑 收到停止信号 ({signal_name})")
        shutdown_event = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not shutdown_event:
        try:
            msg = consumer.poll(timeout=0.5)
        except Exception as e:
            logger.error(f"poll 异常: {e}")
            time.sleep(1)
            continue

        if msg is None:
            time.sleep(0.05)
            continue

        if msg.error():
            logger.error(f"poll error: {msg.error()}")
            time.sleep(1)
            continue

        try:
            task_data = json.loads(msg.value().decode('utf-8'))
        except Exception as e:
            logger.error(f"消息解析失败: {e}")
            continue

        success = process_task(task_data, logger)

        if success:
            for commit_attempt in range(3):
                try:
                    consumer.commit(asynchronous=False)
                    logger.info(f"任务 {task_data.get('task_id')}: offset 提交成功")
                    break
                except Exception as e:
                    logger.error(f"任务 {task_data.get('task_id')}: offset 提交失败 ({commit_attempt + 1}/3): {e}")
                    time.sleep(2)
            else:
                logger.error(f"任务 {task_data.get('task_id')}: offset 提交失败")

        logger.info(f"任务完成: {task_data.get('task_id')}")

    consumer.close()
    logger.info("🛑 服务已关闭")


if __name__ == "__main__":
    main()
