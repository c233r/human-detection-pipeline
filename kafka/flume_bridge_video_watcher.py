#!/usr/bin/env python3
"""
Flume 桥接脚本 — 视频目录监控 + 幂等检查 + 生成 Flume 触发文件

架构:
  [视频目录] → [Python 幂等检查] → [flume_watch/ .task 文件] → [Flume spooldir] → [Kafka video_tasks]

工作流程:
  1. 监听 /data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos/videos/
  2. 对每个视频文件，检查 /data2/caimoyao/api_outputs/temp/{task_id} 是否已存在
  3. 若不存在（未处理），则在 /data2/caimoyao/flume_watch/ 下创建 {task_id}.task 文件
  4. Flume spooldir 监听该目录，将 JSON 内容发送到 Kafka video_tasks topic
  5. 发送成功后 Flume 将 .task 文件移至 flume_done/

使用方法:
    conda activate api_service
    python flume_bridge_video_watcher.py [--scan-once] [--interval N]
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Set, List, Optional


# ============================================================
# 配置
# ============================================================

class Config:
    VIDEO_DIR = Path("/data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos/videos")
    TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")
    FLUME_WATCH_DIR = Path("/data2/caimoyao/flume_watch")
    FLUME_DONE_DIR = Path("/data2/caimoyao/flume_done")
    FLUME_FAILED_DIR = Path("/data2/caimoyao/flume_failed")

    POLL_INTERVAL = 5
    FILE_STABLE_TIME = 3
    SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    CONF_THRESHOLD = 0.5
    TASK_SUFFIX = ".task"

    LOG_FILE = "/data2/caimoyao/logs/flume_bridge_video_watcher.log"


# ============================================================
# 日志
# ============================================================

def setup_logging(log_file: str):
    logger = logging.getLogger("flume_bridge")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
# 幂等检查
# ============================================================

def load_processed_task_ids(temp_dir: Path) -> Set[str]:
    task_ids = set()
    if not temp_dir.exists():
        return task_ids
    try:
        for entry in temp_dir.iterdir():
            if entry.is_dir():
                task_ids.add(entry.name)
    except PermissionError:
        pass
    return task_ids


def is_task_id_processed(task_id: str, temp_dir: Path,
                         flume_done_dir: Path) -> bool:
    if (temp_dir / task_id).is_dir():
        return True
    if (flume_done_dir / f"{task_id}.task").is_file():
        return True
    return False


# ============================================================
# 生成 Flume 触发文件
# ============================================================

def create_task_file(video_path: Path, conf_threshold: float,
                     flume_watch_dir: Path, task_suffix: str,
                     logger: logging.Logger) -> Optional[Path]:
    task_id = video_path.stem
    message = {
        "task_id": task_id,
        "video_path": str(video_path.resolve()),
        "filename": video_path.name,
        "conf_threshold": conf_threshold,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    task_file = flume_watch_dir / f"{task_id}{task_suffix}"
    tmp_file = flume_watch_dir / f"{task_id}{task_suffix}.tmp"

    try:
        flume_watch_dir.mkdir(parents=True, exist_ok=True)
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(message, f, ensure_ascii=False)
        tmp_file.rename(task_file)
        logger.info(f"📝 创建触发文件: {task_file.name} (task_id={task_id})")
        return task_file
    except Exception as e:
        logger.error(f"❌ 创建触发文件失败 {task_id}: {e}")
        if tmp_file.exists():
            tmp_file.unlink()
        return None


# ============================================================
# 主监控器
# ============================================================

class FlumeBridgeWatcher:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.running = False
        self._processed_in_session: Set[str] = set()

        config.FLUME_WATCH_DIR.mkdir(parents=True, exist_ok=True)
        config.FLUME_DONE_DIR.mkdir(parents=True, exist_ok=True)
        config.FLUME_FAILED_DIR.mkdir(parents=True, exist_ok=True)

    def _get_video_files(self) -> List[Path]:
        files = []
        if not self.config.VIDEO_DIR.exists():
            self.logger.error(f"❌ 视频目录不存在: {self.config.VIDEO_DIR}")
            return files
        for entry in self.config.VIDEO_DIR.iterdir():
            if entry.is_file() and entry.suffix.lower() in self.config.SUPPORTED_EXTS:
                files.append(entry)
        return sorted(files)

    def _wait_for_file_stable(self, file_path: Path) -> bool:
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
            self.logger.warning(f"⚠ 文件稳定性检查失败 {file_path.name}: {e}")
            return True

    def process_file(self, file_path: Path,
                     precomputed_processed: Optional[Set[str]] = None) -> bool:
        task_id = file_path.stem

        if task_id in self._processed_in_session:
            return False

        if precomputed_processed is not None:
            if task_id in precomputed_processed:
                self.logger.debug(f"⏭ 跳过(temp已存在): {task_id}")
                return False
        else:
            if is_task_id_processed(task_id, self.config.TEMP_DIR,
                                    self.config.FLUME_DONE_DIR):
                self.logger.debug(f"⏭ 跳过(已处理): {task_id}")
                return False

        self.logger.info(f"📹 新视频: {file_path.name} (task_id={task_id})")

        if not self._wait_for_file_stable(file_path):
            self.logger.warning(f"⚠ 文件可能仍在写入: {file_path.name}")

        result = create_task_file(
            file_path, self.config.CONF_THRESHOLD,
            self.config.FLUME_WATCH_DIR, self.config.TASK_SUFFIX,
            self.logger
        )
        if result:
            self._processed_in_session.add(task_id)
            return True
        return False

    def scan_existing(self):
        self.logger.info(f"🔍 扫描已有视频文件...")
        self.logger.info(f"   视频目录: {self.config.VIDEO_DIR}")
        self.logger.info(f"   幂等检查: {self.config.TEMP_DIR} + {self.config.FLUME_DONE_DIR}")
        self.logger.info(f"   Flume watch: {self.config.FLUME_WATCH_DIR}")

        processed_ids = load_processed_task_ids(self.config.TEMP_DIR)
        for f in self.config.FLUME_DONE_DIR.glob("*.task"):
            processed_ids.add(f.stem)

        self.logger.info(f"   已处理 task_id 数: {len(processed_ids)}")

        video_files = self._get_video_files()
        self.logger.info(f"   视频文件总数: {len(video_files)}")

        pending = []
        skipped = 0
        for f in video_files:
            task_id = f.stem
            if task_id in processed_ids or task_id in self._processed_in_session:
                skipped += 1
                continue
            pending.append(f)

        self.logger.info(f"   待发送: {len(pending)}, 跳过: {skipped}")

        sent = 0
        for f in pending:
            if self.process_file(f, processed_ids):
                sent += 1

        self.logger.info(f"✅ 扫描完成: 生成 {sent}/{len(pending)} 个触发文件")

    def watch_forever(self):
        self.running = True
        self.logger.info(f"🔄 实时监听 (轮询 {self.config.POLL_INTERVAL}s)...")

        last_files = set()
        last_refresh = 0
        processed_ids = load_processed_task_ids(self.config.TEMP_DIR)

        while self.running:
            try:
                current_files = {f.name for f in self._get_video_files()}
                new_files = current_files - last_files

                if new_files:
                    self.logger.info(f"📁 检测到 {len(new_files)} 个新文件")
                    for name in sorted(new_files):
                        f = self.config.VIDEO_DIR / name
                        self.process_file(f, processed_ids)

                last_files = current_files

                if (time.time() - last_refresh) > 300:
                    processed_ids = load_processed_task_ids(self.config.TEMP_DIR)
                    last_refresh = time.time()

                time.sleep(self.config.POLL_INTERVAL)

            except KeyboardInterrupt:
                self.logger.info("🛑 中断信号，退出...")
                break
            except Exception as e:
                self.logger.error(f"❌ 监听异常: {e}")
                time.sleep(self.config.POLL_INTERVAL)

    def stop(self):
        self.running = False
        self.logger.info("👋 桥接服务已停止")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Flume 桥接: 视频目录监控 + 幂等检查 + 触发文件生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动时扫描 + 实时监听
  python flume_bridge_video_watcher.py

  # 只扫描一次就退出
  python flume_bridge_video_watcher.py --scan-once

  # 不扫描已有文件，只监听新文件
  python flume_bridge_video_watcher.py --no-initial-scan
        """,
    )
    parser.add_argument("--video-dir", default=str(Config.VIDEO_DIR))
    parser.add_argument("--temp-dir", default=str(Config.TEMP_DIR))
    parser.add_argument("--flume-dir", default=str(Config.FLUME_WATCH_DIR))
    parser.add_argument("--interval", type=int, default=Config.POLL_INTERVAL)
    parser.add_argument("--scan-once", action="store_true")
    parser.add_argument("--no-initial-scan", action="store_true")
    parser.add_argument("--log-file", default=Config.LOG_FILE)

    args = parser.parse_args()

    config = Config()
    config.VIDEO_DIR = Path(args.video_dir)
    config.TEMP_DIR = Path(args.temp_dir)
    config.FLUME_WATCH_DIR = Path(args.flume_dir)
    config.FLUME_DONE_DIR = Path(str(args.flume_dir).replace("flume_watch", "flume_done"))
    config.FLUME_FAILED_DIR = Path(str(args.flume_dir).replace("flume_watch", "flume_failed"))
    config.POLL_INTERVAL = args.interval

    logger = setup_logging(args.log_file)

    print("=" * 60)
    print("   Flume 桥接服务 - 视频目录监控")
    print("=" * 60)
    print(f"   📹 视频目录: {config.VIDEO_DIR}")
    print(f"   🔍 幂等检查: {config.TEMP_DIR}")
    print(f"   📂 Flume watch: {config.FLUME_WATCH_DIR}")
    print(f"   ✅ Flume done:  {config.FLUME_DONE_DIR}")
    print(f"   ⏱ 轮询间隔: {config.POLL_INTERVAL}s")
    print("=" * 60)

    if not config.VIDEO_DIR.exists():
        logger.error(f"❌ 视频目录不存在: {config.VIDEO_DIR}")
        sys.exit(1)

    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    watcher = FlumeBridgeWatcher(config, logger)

    def signal_handler(signum, frame):
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.scan_once:
        watcher.scan_existing()
        return

    if not args.no_initial_scan:
        watcher.scan_existing()

    watcher.watch_forever()


if __name__ == "__main__":
    main()