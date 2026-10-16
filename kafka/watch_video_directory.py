#!/usr/bin/env python3
"""
视频目录监控 + Kafka 上传服务
监听指定目录中的 mp4 文件，实现幂等检查后实时上传到 Kafka

使用方法:
    conda activate api_service
    python watch_video_directory.py

特性:
    - 实时监听目录（轮询方式，可配置间隔）
    - 幂等检查: 基于 /data2/caimoyao/api_outputs/temp/{task_id} 是否已存在
    - 支持初次扫描已有文件 (--scan-once 只扫描一次就退出)
    - 与现有 Kafka 管线无缝集成 (video_tasks topic)
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from kafka import KafkaProducer
from kafka.errors import KafkaError


# ============================================================
# 配置
# ============================================================

class Config:
    VIDEO_DIR = Path("/data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos/videos")
    TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_TOPIC = "video_tasks"
    KAFKA_KEY_PREFIX = ""

    POLL_INTERVAL = 5
    FILE_STABLE_TIME = 3
    SCAN_ON_STARTUP = True
    SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    CONF_THRESHOLD = 0.5
    PARALLEL_SEND = 4

    SEND_LOG_FILE = "/data2/caimoyao/logs/video_upload_watcher.log"


# ============================================================
# 日志
# ============================================================

def setup_logging(log_file: str):
    logger = logging.getLogger("video_watcher")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        print(f"⚠ 无法创建日志文件: {e}")

    return logger


# ============================================================
# Kafka 生产者
# ============================================================

class VideoTaskProducer:
    def __init__(self, bootstrap_servers: str, topic: str,
                 logger: logging.Logger, max_retries: int = 3):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.logger = logger
        self.max_retries = max_retries
        self.producer: Optional[KafkaProducer] = None

    def connect(self):
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            max_block_ms=30000,
            retries=3,
            retry_backoff_ms=1000,
            acks="all",
            linger_ms=100,
        )
        self.logger.info(f"✅ Kafka Producer 已连接: {self.bootstrap_servers}")

    def send_task(self, task_id: str, video_path: str,
                  conf_threshold: float = 0.5) -> bool:
        if not self.producer:
            raise RuntimeError("Producer 未连接")

        message = {
            "task_id": task_id,
            "video_path": os.path.abspath(video_path),
            "filename": os.path.basename(video_path),
            "conf_threshold": conf_threshold,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        for attempt in range(self.max_retries):
            try:
                future = self.producer.send(
                    self.topic,
                    key=task_id.encode("utf-8"),
                    value=json.dumps(message).encode("utf-8"),
                )
                future.get(timeout=30)
                return True
            except KafkaError as e:
                self.logger.error(
                    f"❌ 发送失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"❌ 发送异常: {e}")
                break

        return False

    def flush(self):
        if self.producer:
            self.producer.flush()

    def close(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None
            self.logger.info("🔌 Kafka Producer 已关闭")


# ============================================================
# 幂等检查
# ============================================================

def load_processed_task_ids(temp_dir: Path) -> Set[str]:
    """加载 temp/ 下已处理的 task_id 集合。"""
    task_ids = set()
    if not temp_dir.exists():
        return task_ids

    try:
        for entry in temp_dir.iterdir():
            if entry.is_dir():
                task_ids.add(entry.name)
    except PermissionError as e:
        print(f"⚠ 读取 temp 目录权限不足: {e}")

    return task_ids


def is_already_processed(task_id: str, temp_dir: Path) -> bool:
    """检查 task_id 是否已处理（temp 目录下存在对应文件夹）。"""
    return (temp_dir / task_id).is_dir()


# ============================================================
# 主监控器
# ============================================================

class VideoDirectoryWatcher:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.producer = VideoTaskProducer(
            config.KAFKA_BOOTSTRAP_SERVERS, config.KAFKA_TOPIC, logger
        )
        self.running = False
        self._seen_files: Set[str] = set()
        self._processed_in_this_session: Set[str] = set()
        self._file_mtime_cache: dict = {}

    def _get_video_files(self) -> list:
        """获取目录下所有视频文件。"""
        files = []
        if not self.config.VIDEO_DIR.exists():
            self.logger.error(f"❌ 视频目录不存在: {self.config.VIDEO_DIR}")
            return files

        for entry in self.config.VIDEO_DIR.iterdir():
            if entry.is_file() and entry.suffix.lower() in self.config.SUPPORTED_EXTS:
                files.append(entry)

        return sorted(files)

    def _wait_for_file_stable(self, file_path: Path) -> bool:
        """等待文件大小稳定（确保文件写入完成）。"""
        try:
            prev_size = -1
            stable_count = 0
            for _ in range(10):
                current_size = file_path.stat().st_size
                if current_size == prev_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= self.config.FILE_STABLE_TIME:
                        return True
                else:
                    stable_count = 0
                prev_size = current_size
                time.sleep(1)
            return True
        except Exception as e:
            self.logger.warning(f"⚠ 检查文件稳定性失败 {file_path.name}: {e}")
            return True

    def _process_file(self, file_path: Path,
                      precomputed_processed: Optional[Set[str]] = None) -> bool:
        """处理单个视频文件：幂等检查 + 发送 Kafka。"""
        task_id = file_path.stem

        if precomputed_processed is not None:
            if task_id in precomputed_processed:
                self.logger.debug(f"⏭ 跳过已处理: {task_id}")
                return False
        else:
            if is_already_processed(task_id, self.config.TEMP_DIR):
                self.logger.debug(f"⏭ 跳过已处理 (temp 存在): {task_id}")
                return False

        if task_id in self._processed_in_this_session:
            return False

        self.logger.info(f"📹 发现新视频: {file_path.name} (task_id={task_id})")

        if not self._wait_for_file_stable(file_path):
            self.logger.warning(f"⚠ 文件可能仍在写入: {file_path.name}")

        try:
            success = self.producer.send_task(
                task_id=task_id,
                video_path=str(file_path),
                conf_threshold=self.config.CONF_THRESHOLD,
            )
            if success:
                self.logger.info(
                    f"✅ 任务已发送: task_id={task_id}, file={file_path.name}"
                )
                self._processed_in_this_session.add(task_id)
                return True
            else:
                self.logger.error(f"❌ 发送失败: {task_id}")
                return False
        except Exception as e:
            self.logger.error(f"❌ 处理异常 {task_id}: {e}")
            return False

    def scan_existing(self, parallel: int = 0):
        """扫描目录中已有的文件并发送未处理的任务。"""
        self.logger.info(f"🔍 开始扫描已有视频文件...")
        self.logger.info(f"   目录: {self.config.VIDEO_DIR}")
        self.logger.info(f"   幂等检查: {self.config.TEMP_DIR}")

        processed_ids = load_processed_task_ids(self.config.TEMP_DIR)
        self.logger.info(f"   已处理 task_id 数: {len(processed_ids)}")

        video_files = self._get_video_files()
        self.logger.info(f"   视频文件总数: {len(video_files)}")

        pending_files = []
        skipped = 0
        for f in video_files:
            task_id = f.stem
            if task_id in processed_ids:
                skipped += 1
                continue
            if task_id in self._processed_in_this_session:
                skipped += 1
                continue
            pending_files.append(f)

        self.logger.info(f"   待发送: {len(pending_files)}, 跳过: {skipped}")

        if not pending_files:
            self.logger.info("   🎉 没有需要发送的新视频")
            return

        send_count = parallel if parallel > 0 else self.config.PARALLEL_SEND
        sent = 0

        if send_count > 1:
            with ThreadPoolExecutor(max_workers=send_count) as pool:
                futures = {
                    pool.submit(self._process_file, f, processed_ids): f
                    for f in pending_files
                }
                for future in as_completed(futures):
                    try:
                        if future.result():
                            sent += 1
                    except Exception as e:
                        self.logger.error(f"❌ 发送异常: {e}")
        else:
            for f in pending_files:
                if self._process_file(f, processed_ids):
                    sent += 1

        self.logger.info(f"✅ 扫描完成: 发送 {sent}/{len(pending_files)}")

    def watch_forever(self):
        """持续监听目录变化。"""
        self.running = True

        self.logger.info(f"🔄 开始实时监听 (轮询间隔 {self.config.POLL_INTERVAL}s)...")
        self.logger.info(f"   监听目录: {self.config.VIDEO_DIR}")

        self.producer.connect()

        last_files = set()
        last_processed_check = 0
        processed_ids = load_processed_task_ids(self.config.TEMP_DIR)

        while self.running:
            try:
                current_files = {
                    f.name for f in self._get_video_files()
                }

                new_files = current_files - last_files
                if new_files:
                    self.logger.info(f"📁 检测到 {len(new_files)} 个新文件")
                    new_file_paths = [
                        self.config.VIDEO_DIR / name for name in sorted(new_files)
                    ]
                    for f in new_file_paths:
                        self._process_file(f, processed_ids)

                last_files = current_files

                if (time.time() - last_processed_check) > 300:
                    processed_ids = load_processed_task_ids(self.config.TEMP_DIR)
                    last_processed_check = time.time()
                    self.logger.debug(
                        f"🔄 刷新已处理列表: {len(processed_ids)} 个 task_id"
                    )

                time.sleep(self.config.POLL_INTERVAL)

            except KeyboardInterrupt:
                self.logger.info("🛑 收到中断信号，准备退出...")
                break
            except Exception as e:
                self.logger.error(f"❌ 监听循环异常: {e}")
                time.sleep(self.config.POLL_INTERVAL)

        self.stop()

    def stop(self):
        self.running = False
        self.producer.close()
        self.logger.info("👋 服务已停止")


# ============================================================
# 启动入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="视频目录监控 + Kafka 上传服务（支持幂等）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 实时监听（启动时扫描已有文件）
  python watch_video_directory.py

  # 只扫描一次就退出
  python watch_video_directory.py --scan-once

  # 不扫描已有文件，只监听新文件
  python watch_video_directory.py --no-initial-scan

  # 自定义参数
  python watch_video_directory.py --interval 10 --parallel 8

  # 指定其他目录
  python watch_video_directory.py --video-dir /path/to/videos --temp-dir /path/to/temp
        """,
    )
    parser.add_argument(
        "--video-dir",
        default=str(Config.VIDEO_DIR),
        help="视频目录路径",
    )
    parser.add_argument(
        "--temp-dir",
        default=str(Config.TEMP_DIR),
        help="已处理任务的临时目录 (用于幂等检查)",
    )
    parser.add_argument(
        "--kafka-servers",
        default=Config.KAFKA_BOOTSTRAP_SERVERS,
        help="Kafka broker 地址",
    )
    parser.add_argument(
        "--kafka-topic",
        default=Config.KAFKA_TOPIC,
        help="Kafka topic 名称",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=Config.POLL_INTERVAL,
        help="轮询间隔 (秒)",
    )
    parser.add_argument(
        "--scan-once",
        action="store_true",
        help="只扫描一次就退出 (不进入持续监听模式)",
    )
    parser.add_argument(
        "--no-initial-scan",
        action="store_true",
        help="启动时不扫描已有文件，只监听新文件",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=Config.PARALLEL_SEND,
        help="并行发送数",
    )
    parser.add_argument(
        "--stable-time",
        type=int,
        default=Config.FILE_STABLE_TIME,
        help="文件稳定性检测时间 (秒)",
    )
    parser.add_argument(
        "--log-file",
        default=Config.SEND_LOG_FILE,
        help="日志文件路径",
    )

    args = parser.parse_args()

    config = Config()
    config.VIDEO_DIR = Path(args.video_dir)
    config.TEMP_DIR = Path(args.temp_dir)
    config.KAFKA_BOOTSTRAP_SERVERS = args.kafka_servers
    config.KAFKA_TOPIC = args.kafka_topic
    config.POLL_INTERVAL = args.interval
    config.FILE_STABLE_TIME = args.stable_time

    logger = setup_logging(args.log_file)

    print("=" * 60)
    print("   视频目录监控 + Kafka 上传服务")
    print("=" * 60)
    print(f"   📹 视频目录: {config.VIDEO_DIR}")
    print(f"   🔍 幂等检查: {config.TEMP_DIR}")
    print(f"   📡 Kafka: {config.KAFKA_BOOTSTRAP_SERVERS} → {config.KAFKA_TOPIC}")
    print(f"   ⏱ 轮询间隔: {config.POLL_INTERVAL}s")
    print("=" * 60)

    if not config.VIDEO_DIR.exists():
        logger.error(f"❌ 视频目录不存在: {config.VIDEO_DIR}")
        sys.exit(1)

    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    watcher = VideoDirectoryWatcher(config, logger)

    def signal_handler(signum, frame):
        logger.info(f"🛑 收到信号 {signum}，准备退出...")
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.scan_once:
        watcher.producer.connect()
        watcher.scan_existing(parallel=args.parallel)
        watcher.producer.close()
        logger.info("✅ 扫描完成，退出")
        return

    if not args.no_initial_scan:
        watcher.producer.connect()
        watcher.scan_existing(parallel=args.parallel)

    watcher.watch_forever()


if __name__ == "__main__":
    main()