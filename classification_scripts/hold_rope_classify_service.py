import os
import sys
import json
import time
import signal
import cv2
import numpy as np
import threading
import shutil
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.getcwd())
import torch

from utils.inference import inference_model, init_model
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

logging_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

import logging
logger = logging.getLogger("hold_rope_classify")
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler("hold_rope_classify.log")
file_handler.setFormatter(logging.Formatter(logging_config['format']))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(logging_config['format']))
logger.addHandler(stream_handler)


CONFIG = {
    "tasks_file": "/data2/caimoyao/api_outputs/hold_rope_results/hand_crop_tasks.json",
    "processed_tasks_file": "/data2/caimoyao/api_outputs/hold_rope_results/classify_tasks.json",
    "crops_dir": "/data2/caimoyao/api_outputs/hold_rope_results/hand_crops",
    "labels_dir": "/data2/caimoyao/api_outputs/hold_rope_results/hand_crop_labels",
    "output_dir": "/data2/caimoyao/api_outputs/hold_rope_results/classify_results",
    "poll_interval": 30,
    "classes_map": '/data2/caimoyao/Jump_rope/classification/datas/annotations.txt',
    "config": '/data2/caimoyao/Jump_rope/classification/models/resnet/resnet18_fpn.py',
    "device_id": 4,
    "confidence_threshold": 0.8,
}


class HoldRopeClassifyService:
    def __init__(self):
        self.running = False
        self.thread = None
        self.processed_tasks = set()
        self._load_processed_tasks()
        self._init_model()
        self._init_output_dirs()

    def _load_processed_tasks(self):
        p = Path(CONFIG["processed_tasks_file"])
        if p.exists():
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
                    self.processed_tasks = set(data.get("tasks", []))
                logger.info(f"已加载 {len(self.processed_tasks)} 个已分类任务")
            except Exception as e:
                logger.error(f"读取已分类任务文件失败: {e}")

    def _save_processed_tasks(self):
        p = Path(CONFIG["processed_tasks_file"])
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                json.dump({"tasks": list(self.processed_tasks)}, f)
        except Exception as e:
            logger.error(f"保存已分类任务失败: {e}")

    def _init_model(self):
        logger.info("加载分类模型...")
        classes_map = CONFIG["classes_map"]
        config_path = CONFIG["config"]
        device_id = CONFIG["device_id"]

        self.classes_names, self.label_names = get_info(classes_map)
        model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config_path)

        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        model = BuildNet(model_cfg)
        self.model = init_model(model, data_cfg, device=self.device, mode='eval')
        self.val_pipeline = val_pipeline
        logger.info("分类模型加载完成")

    def _init_output_dirs(self):
        output_dir = Path(CONFIG["output_dir"])
        self.hold_rope_high_dir = output_dir / "hold_rope_high"
        self.hold_rope_low_dir = output_dir / "hold_rope_low"
        self.no_hold_rope_high_dir = output_dir / "no_hold_rope_high"
        self.no_hold_rope_low_dir = output_dir / "no_hold_rope_low"

        self.hold_rope_high_labels_dir = output_dir / "hold_rope_high_labels"
        self.hold_rope_low_labels_dir = output_dir / "hold_rope_low_labels"
        self.no_hold_rope_high_labels_dir = output_dir / "no_hold_rope_high_labels"
        self.no_hold_rope_low_labels_dir = output_dir / "no_hold_rope_low_labels"

        dirs = [
            self.hold_rope_high_dir, self.hold_rope_low_dir,
            self.no_hold_rope_high_dir, self.no_hold_rope_low_dir,
            self.hold_rope_high_labels_dir, self.hold_rope_low_labels_dir,
            self.no_hold_rope_high_labels_dir, self.no_hold_rope_low_labels_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _get_new_tasks(self) -> List[str]:
        self._load_processed_tasks()
        tasks_file = Path(CONFIG["tasks_file"])
        if not tasks_file.exists():
            return []

        try:
            with open(tasks_file, 'r') as f:
                data = json.load(f)
            all_tasks = data.get("tasks", [])
            new_tasks = [t for t in all_tasks if t not in self.processed_tasks]
            return new_tasks
        except Exception as e:
            logger.error(f"读取 tasks 文件失败: {e}")
            return []

    def _process_task(self, task_id: str) -> bool:
        logger.info(f"开始处理任务: {task_id}")
        crops_dir = Path(CONFIG["crops_dir"])
        labels_dir = Path(CONFIG["labels_dir"])
        threshold = CONFIG["confidence_threshold"]

        crop_files = sorted(crops_dir.glob(f"{task_id}_*_leftHand.jpg"))
        crop_files += sorted(crops_dir.glob(f"{task_id}_*_rightHand.jpg"))

        if not crop_files:
            logger.warning(f"任务 {task_id} 没有找到裁剪图片")
            return False

        hold_high_count = 0
        hold_low_count = 0
        no_hold_high_count = 0
        no_hold_low_count = 0
        moved_label_files = set()

        for crop_file in crop_files:
            stem = crop_file.stem
            parts = stem.split('_')
            if len(parts) < 6:
                continue

            is_left = parts[-1] == 'leftHand'
            label_file = labels_dir / ('_'.join(parts[:-1]) + '.txt')

            image = cv2.imread(str(crop_file))
            if image is None:
                continue

            h, w = image.shape[:2]
            if h != w:
                max_side = max(h, w)
                pad_h = (max_side - h) // 2
                pad_w = (max_side - w) // 2
                image = cv2.copyMakeBorder(
                    image,
                    pad_h, max_side - h - pad_h,
                    pad_w, max_side - w - pad_w,
                    borderType=cv2.BORDER_CONSTANT, value=0
                )

            try:
                result = inference_model(self.model, image, self.val_pipeline, self.classes_names, self.label_names)
                label = result['pred_label']
                score = result['pred_score']
            except Exception as e:
                logger.error(f"推理失败 {crop_file}: {e}")
                continue

            is_high_confidence = score >= threshold

            if label == 0:
                if is_high_confidence:
                    save_dir = self.hold_rope_high_dir
                    hold_high_count += 1
                else:
                    save_dir = self.hold_rope_low_dir
                    hold_low_count += 1
                label_name = 'hold_rope'
            else:
                if is_high_confidence:
                    save_dir = self.no_hold_rope_high_dir
                    no_hold_high_count += 1
                else:
                    save_dir = self.no_hold_rope_low_dir
                    no_hold_low_count += 1
                label_name = 'no_hold_rope'

            save_image_path = save_dir / crop_file.name
            if save_image_path.exists():
                save_image_path.unlink()
            shutil.move(str(crop_file), str(save_image_path))

            if label_file.exists() and label_file not in moved_label_files:
                moved_label_files.add(label_file)

        for label_file in moved_label_files:
            try:
                label_file.unlink()
            except:
                pass

        logger.info(f"任务 {task_id} 处理完成: "
                    f"持有绳子(高置信) {hold_high_count} 张, 持有绳子(低置信) {hold_low_count} 张, "
                    f"不持有绳子(高置信) {no_hold_high_count} 张, 不持有绳子(低置信) {no_hold_low_count} 张")
        return True

    def _monitor_loop(self):
        logger.info(f"监控循环启动，轮询间隔: {CONFIG['poll_interval']}s")
        while self.running:
            try:
                new_tasks = self._get_new_tasks()
                if new_tasks:
                    logger.info(f"轮询: 发现 {len(new_tasks)} 个新任务")
                    for task_id in new_tasks:
                        if not self.running:
                            break
                        try:
                            self._process_task(task_id)
                            self.processed_tasks.add(task_id)
                            self._save_processed_tasks()
                        except Exception as e:
                            logger.error(f"处理任务 {task_id} 失败: {e}")

            except Exception as e:
                logger.error(f"监控循环异常: {e}")

            if self.running:
                time.sleep(CONFIG["poll_interval"])

        logger.info("监控循环已停止")

    def start(self):
        if self.running:
            logger.warning("服务已在运行")
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("HoldRope 分类服务已启动")

    def stop(self):
        logger.info("正在停止 HoldRope 分类服务...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("HoldRope 分类服务已停止")


def main():
    service = HoldRopeClassifyService()

    def signal_handler(signum, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    main()
