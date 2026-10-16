import os
import sys
import json
import time
import threading
import signal
import requests
import cv2
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from video_utils import read_video_frames

multiprocessing.set_start_method('spawn', force=True)

from inference_onnx_multi_rtmpose_l import RTMPoseModel

logging_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

import logging
logger = logging.getLogger("hold_rope_monitor")
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler("hold_rope_monitor.log")
file_handler.setFormatter(logging.Formatter(logging_config['format']))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(logging_config['format']))
logger.addHandler(stream_handler)


def gpu_worker(gpu_id: int, model_path: str, task_queue: multiprocessing.Queue, 
               temp_dir: Path, crops_dir: Path, keypoints_dir: Path, 
               processed_tasks_file: Path, api_url: str, frame_cache):
    worker_logger = logging.getLogger(f"hold_rope_monitor.gpu_{gpu_id}")
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False
    file_handler = logging.FileHandler("hold_rope_monitor.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(logging_config['format']))
    worker_logger.addHandler(file_handler)

    worker_logger.info(f"Worker GPU {gpu_id}: 启动中...")
    model = RTMPoseModel(model_path, device_id=gpu_id)
    worker_logger.info(f"Worker GPU {gpu_id}: 模型加载完成")

    while True:
        task = task_queue.get()
        if task is None:
            worker_logger.info(f"Worker GPU {gpu_id}: 收到停止信号，退出")
            break

        try:
            _process_task_worker(task, model, temp_dir, crops_dir, keypoints_dir, 
                                processed_tasks_file, api_url, gpu_id, worker_logger, frame_cache)
        except Exception as e:
            worker_logger.error(f"Worker GPU {gpu_id}: 处理任务失败 {task.get('task_id')}: {e}")


def _process_task_worker(task: Dict, model, temp_dir: Path, crops_dir: Path, 
                         keypoints_dir: Path, processed_tasks_file: Path, 
                         api_url: str, gpu_id: int, worker_logger, frame_cache) -> bool:
    task_id = task.get("task_id")
    if not task_id:
        return False

    task_dir = temp_dir / task_id
    if not task_dir.exists():
        worker_logger.warning(f"Worker GPU {gpu_id}: 任务目录不存在: {task_dir}")
        _mark_task_processed(task_id, processed_tasks_file)
        return False

    labels_dir = task_dir / "labels"

    if not labels_dir.exists():
        worker_logger.warning(f"Worker GPU {gpu_id}: 任务 {task_id} 的 labels 目录不存在")
        _mark_task_processed(task_id, processed_tasks_file)
        return False

    # 从 frame_cache 获取帧数据
    _t_wait = time.time()
    while task_id not in frame_cache:
        time.sleep(0.01)
    frames_np = frame_cache.pop(task_id)
    n_frames = len(frames_np)
    worker_logger.info(f"Worker GPU {gpu_id}: 任务 {task_id}: 获取到 {n_frames} 帧, "
                       f"等待耗时: {(time.time() - _t_wait) * 1000:.0f}ms")

    label_files = sorted(labels_dir.glob("*.json"))
    worker_logger.info(f"Worker GPU {gpu_id}: 任务 {task_id}: 找到 {len(label_files)} 个标签文件")

    has_error = False

    for label_file in label_files:
        frame_idx = int(label_file.stem)
        if frame_idx >= n_frames:
            worker_logger.warning(f"Worker GPU {gpu_id}: 帧索引 {frame_idx} 超出范围 ({n_frames})，跳过")
            continue
        frame_np = frames_np[frame_idx]  # (H, W, 3) RGB uint8
        # RGB → BGR for cv2
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[:2]

        try:
            with open(label_file, 'r') as f:
                labels = json.load(f)

            for label in labels:
                label_val = label.get("label")
                label_int = int(label_val) if label_val is not None else -1
                if label_int != 1:
                    continue

                box = label.get("box", [0, 0, 0, 0])
                x1, y1, x2, y2 = map(float, box)

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(width, int(x2))
                y2 = min(height, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                new_w = int(w * 1.3)
                new_h = int(h * 1.3)
                x1 = max(0, int(cx - new_w / 2))
                y1 = max(0, int(cy - new_h / 2))
                x2 = min(width, int(cx + new_w / 2))
                y2 = min(height, int(cy + new_h / 2))

                if x2 <= x1 or y2 <= y1:
                    continue

                cropped_image = frame_bgr[y1:y2, x1:x2]

                if cropped_image.size == 0:
                    continue

                try:
                    worker_logger.debug(f"Worker GPU {gpu_id}: 帧 {frame_idx}, box=({x1},{y1},{x2},{y2}), 裁剪尺寸={cropped_image.shape}")
                    keypoints = model.inference(cropped_image)
                    worker_logger.debug(f"Worker GPU {gpu_id}: 帧 {frame_idx}, 关键点数量={len(keypoints)}")
                    if keypoints:
                        worker_logger.debug(f"Worker GPU {gpu_id}: 帧 {frame_idx}, 第一个关键点={keypoints[0]}")
                    orig_keypoints = []
                    for kp in keypoints:
                        orig_keypoints.append([
                            float(kp['x']) + x1,
                            float(kp['y']) + y1,
                            float(kp['score']),
                        ])
                    label["keypoints"] = orig_keypoints
                    worker_logger.debug(f"Worker GPU {gpu_id}: 帧 {frame_idx}, 原始关键点数量={len(orig_keypoints)}")
                except Exception as e:
                    worker_logger.error(f"Worker GPU {gpu_id}: 关键点检测失败 {label_file}: {e}")
                    has_error = True

            with open(label_file, 'w') as f:
                json.dump(labels, f, indent=2)

        except Exception as e:
            worker_logger.error(f"Worker GPU {gpu_id}: 处理标签文件 {label_file} 失败: {e}")
            has_error = True

    if not has_error:
        _mark_task_processed(task_id, processed_tasks_file)
        worker_logger.info(f"Worker GPU {gpu_id}: 任务 {task_id} 处理完成")
        return True
    else:
        worker_logger.warning(f"Worker GPU {gpu_id}: 任务 {task_id} 处理过程中出现错误")
        return False


def _mark_task_processed(task_id: str, processed_tasks_file: Path):
    try:
        tasks = set()
        if processed_tasks_file.exists():
            with open(processed_tasks_file, 'r') as f:
                data = json.load(f)
                tasks = set(data.get("tasks", []))
        tasks.add(task_id)
        with open(processed_tasks_file, 'w') as f:
            json.dump({"tasks": list(tasks)}, f)
    except Exception as e:
        logger.error(f"保存任务记录失败 {task_id}: {e}")


def _delete_task_from_api(task_id: str, api_url: str):
    try:
        url = f"{api_url}/tasks/{task_id}"
        response = requests.delete(url, timeout=10)
        if response.status_code == 200:
            logger.info(f"已删除 API 任务记录: {task_id}")
        elif response.status_code == 404:
            logger.debug(f"API 任务记录不存在: {task_id}")
    except Exception as e:
        logger.error(f"删除 API 任务记录异常: {task_id} - {e}")


class HoldRopeMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.api_url = config.get("api_url", "http://localhost:8000")
        self.temp_dir = Path(config.get("temp_dir", "/data2/caimoyao/api_outputs/temp"))
        self.output_dir = Path(config.get("output_dir", "/data2/caimoyao/api_outputs/hold_rope_results"))
        self.poll_interval = config.get("poll_interval", 30)
        self.model_path = config.get("model_path", os.path.join(os.path.dirname(__file__), "weights", "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"))
        self.processed_tasks_file = self.output_dir / "processed_tasks.json"
        self.crops_dir = self.output_dir / "crops"
        self.keypoints_dir = self.output_dir / "keypoints"
        self.gpu_ids = config.get("gpu_ids", [0])
        self.processed_tasks = set()
        self.running = False
        self.thread = None
        self.task_queue = None
        self.workers = []
        self.frame_cache = None  # 在 start 中初始化（Manager.dict）
        self._load_processed_tasks()
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.keypoints_dir.mkdir(parents=True, exist_ok=True)

    def _load_processed_tasks(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.processed_tasks_file.exists():
            try:
                with open(self.processed_tasks_file, 'r') as f:
                    data = json.load(f)
                    self.processed_tasks = set(data.get("tasks", []))
                logger.info(f"已加载 {len(self.processed_tasks)} 个已处理任务")
            except Exception as e:
                logger.error(f"加载已处理任务失败: {e}")

    def _get_completed_tasks(self) -> Optional[List[Dict]]:
        try:
            url = f"{self.api_url}/tasks?status=completed"
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                logger.error(f"API 不存在 (404): {url}，监控将关闭")
                return None
            response.raise_for_status()
            data = response.json()
            return data.get("tasks", [])
        except Exception as e:
            logger.error(f"获取任务列表失败: {e}")
            return []

    def _monitor_loop(self):
        logger.info(f"监控循环启动，轮询间隔: {self.poll_interval}s")
        while self.running:
            try:
                tasks = self._get_completed_tasks()
                if tasks is None:
                    self.stop()
                    break

                new_tasks = []
                for task in tasks:
                    task_id = task.get("task_id")
                    if task_id and task_id not in self.processed_tasks:
                        new_tasks.append(task)
                        self.processed_tasks.add(task_id)

                if new_tasks:
                    logger.info(f"轮询: 发现 {len(new_tasks)} 个新任务")
                    for task in new_tasks:
                        task_id = task.get("task_id")
                        # 读取视频帧，存入 frame_cache
                        video_path = self.temp_dir / task_id / "input.mp4"
                        if video_path.exists():
                            try:
                                frames_np = read_video_frames(str(video_path))
                                self.frame_cache[task_id] = frames_np
                                logger.info(f"📖 任务 {task_id}: 读取 {len(frames_np)} 帧")
                            except Exception as e:
                                logger.error(f"任务 {task_id}: 读取视频失败 {e}")
                                continue
                        else:
                            logger.warning(f"任务 {task_id}: 视频文件不存在 {video_path}")
                            continue
                        self.task_queue.put(task)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")

            if self.running:
                time.sleep(self.poll_interval)

        logger.info("监控循环已停止")

    def start(self):
        if self.running:
            logger.warning("监控已在运行")
            return

        self.running = True
        self.task_queue = multiprocessing.Queue()
        self.frame_cache = multiprocessing.Manager().dict()

        for gpu_id in self.gpu_ids:
            worker = multiprocessing.Process(
                target=gpu_worker,
                args=(gpu_id, self.model_path, self.task_queue,
                      self.temp_dir, self.crops_dir, self.keypoints_dir,
                      self.processed_tasks_file, self.api_url, self.frame_cache),
                name=f"GPU_{gpu_id}"
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"Worker GPU {gpu_id}: 已启动")

        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"HoldRope 监控服务已启动，使用 GPU {self.gpu_ids}")

    def stop(self):
        logger.info("正在停止 HoldRope 监控服务...")
        self.running = False

        if self.task_queue:
            for _ in self.gpu_ids:
                self.task_queue.put(None)

        if self.workers:
            for worker in self.workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} 超时未退出，强制终止")
                    worker.terminate()
                    worker.join()

        if self.thread:
            self.thread.join(timeout=10)

        logger.info("HoldRope 监控服务已停止")


def main():
    config = {
        "api_url": "http://localhost:8000",
        "temp_dir": "/data2/caimoyao/api_outputs/temp",
        "output_dir": "/data2/caimoyao/api_outputs/hold_rope_results",
        "poll_interval": 30,
        "model_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"),
        "gpu_ids": [2, 3, 4],
    }

    monitor = HoldRopeMonitor(config)

    def signal_handler(signum, frame):
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main()
