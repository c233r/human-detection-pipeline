import os
import sys
import json
import time
import math
import cv2
import numpy as np
import multiprocessing
import signal
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import torch

from utils.inference import inference_model, init_model
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from core.datasets.compose import Compose

logging_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

import logging
logger = logging.getLogger("hand_crop_classify")
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler("hand_crop_service.log")
file_handler.setFormatter(logging.Formatter(logging_config['format']))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(logging_config['format']))
logger.addHandler(stream_handler)


CONFIG = {
    "processed_tasks_file": "/data2/caimoyao/api_outputs/hold_rope_results/processed_tasks.json",
    "temp_dir": "/data2/caimoyao/api_outputs/temp",
    "processed_tasks_file_local": "/data2/caimoyao/api_outputs/hold_rope_results/hand_crop_tasks.json",
    "poll_interval": 30,
    # 分类模型配置
    "classes_map": '/data2/caimoyao/Jump_rope/classification/datas/annotations.txt',
    "config": '/data2/caimoyao/Jump_rope/classification/models/resnet/resnet18_fpn.py',
    "gpu_ids": [ 2], # , 3, 4,5, 6
    "confidence_threshold": 0.8,
    # 手部裁剪参数
    "hand_shoulder_ratio": 0.8,
    "min_hand_area": 1800,
    "keypoint_score_threshold": 0.3,
    "batch_chunk_size": 100,  # 每 chunk 最多 500 张手图，防 OOM
}


# ══════════════════════════════════════════════════════════════════
#  GPU Worker 进程（每个 GPU 一个）
# ══════════════════════════════════════════════════════════════════
class GPUWorker:
    """每个 Worker 绑定一张 GPU，顺序处理任务。"""

    def __init__(self, gpu_id: int, task_q: multiprocessing.Queue,
                 result_q: multiprocessing.Queue):
        self.gpu_id = gpu_id
        self.task_queue = task_q
        self.result_queue = result_q
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self._init_model()

    def _init_model(self):
        logger.info(f"[GPU-{self.gpu_id}] 加载分类模型...")
        classes_map = CONFIG["classes_map"]
        config_path = CONFIG["config"]
        self.classes_names, self.label_names = get_info(classes_map)
        model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config_path)
        self.model = init_model(BuildNet(model_cfg), data_cfg, device=self.device, mode='eval')
        self.val_pipeline = val_pipeline
        logger.info(f"[GPU-{self.gpu_id}] 分类模型加载完成，设备: {self.device}")

    # ── CPU 工具函数 ────────────────────────────────────────────

    @staticmethod
    def _crop_person_hands(person: dict, image: np.ndarray, width: int, height: int):
        label = person.get("label")
        label_int = int(label) if label is not None else -1
        if label_int != 1:
            return None
        kps = person.get("keypoints", [])
        if len(kps) < 25:
            return None
        _t0 = time.perf_counter()
        ls, rs = kps[5], kps[6]
        lh, rh = kps[23], kps[24]
        if ls[2] < CONFIG["keypoint_score_threshold"] or rs[2] < CONFIG["keypoint_score_threshold"]:
            return None
        shoulder_dist = math.sqrt((ls[0] - rs[0])**2 + (ls[1] - rs[1])**2)
        if shoulder_dist < 1:
            return None

        def _crop(kp):
            hx, hy, hs = kp
            if hs < CONFIG["keypoint_score_threshold"]:
                return None
            s = int(shoulder_dist * CONFIG["hand_shoulder_ratio"])
            x1 = int(max(0, hx - s)); x2 = int(min(width, hx + s))
            y1 = int(max(0, hy - s)); y2 = int(min(height, hy + s))
            if x2 <= x1 or y2 <= y1:
                return None
            return image[y1:y2, x1:x2]

        left = _crop(lh)
        right = _crop(rh)
        logger.debug(f"裁剪耗时: {(time.perf_counter() - _t0) * 1000:.2f}ms")
        return left, right

    @staticmethod
    def _pad_square(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == w:
            return img
        m = max(h, w)
        ph = (m - h) // 2
        pw = (m - w) // 2
        return cv2.copyMakeBorder(img, ph, m - h - ph, pw, m - w - pw,
                                  borderType=cv2.BORDER_CONSTANT, value=0)

    @staticmethod
    def _write_json(label_file: Path, labels: list):
        with open(label_file, 'w') as f:
            json.dump(labels, f, indent=2)

    # ── GPU 分类 ────────────────────────────────────────────────

    def _classify_hand(self, hand_img: np.ndarray, hand_name: str = "") -> tuple:
        if hand_img is None or hand_img.size == 0:
            return None, 0.0
        _t0 = time.perf_counter()
        padded = self._pad_square(hand_img)
        try:
            result = inference_model(
                self.model, padded, self.val_pipeline,
                self.classes_names, self.label_names
            )
            elapsed = (time.perf_counter() - _t0) * 1000
            logger.debug(f"[GPU-{self.gpu_id}] {hand_name}分类耗时: {elapsed:.1f}ms, "
                         f"label={result['pred_label']}, score={result['pred_score']:.3f}")
            return result['pred_label'], result['pred_score']
        except Exception as e:
            logger.error(f"[GPU-{self.gpu_id}] 分类失败: {e}")
            return None, 0.0

    def _classify_hands_batch(self, hand_images: list[np.ndarray]) -> list[tuple]:
        """Batch 分类多张手图，自动分 chunk 避免 OOM。"""
        if not hand_images:
            return []

        all_results = []
        chunk_size = CONFIG["batch_chunk_size"]
        n_chunks = (len(hand_images) + chunk_size - 1) // chunk_size
        _batch_t0 = time.perf_counter()

        for ci in range(n_chunks):
            chunk = hand_images[ci * chunk_size:(ci + 1) * chunk_size]
            _t0 = time.perf_counter()
            device = next(self.model.parameters()).device
            tensors = []

            pipeline = Compose(
                [t for t in self.val_pipeline if t.get('type') != 'LoadImageFromFile']
            )

            for img in chunk:
                padded = self._pad_square(img)
                data = dict(img=padded, filename=None)
                t = pipeline(data)['img']
                tensors.append(t)

            batch = torch.stack(tensors, dim=0).to(device)
            logger.debug(f"[GPU-{self.gpu_id}] chunk {ci + 1}/{n_chunks} "
                         f"预处理: {(time.perf_counter() - _t0) * 1000:.1f}ms, "
                         f"size={batch.shape}")

            with torch.no_grad():
                scores = self.model(batch, return_loss=False)
                pred_scores, pred_labels = torch.max(scores, axis=1)

            chunk_results = list(zip(pred_labels.cpu().tolist(), pred_scores.cpu().tolist()))
            all_results.extend(chunk_results)

        logger.info(f"[GPU-{self.gpu_id}] batch 分类 {len(hand_images)} 张手图, "
                    f"{n_chunks} chunk, "
                    f"总耗时: {(time.perf_counter() - _batch_t0) * 1000:.0f}ms")
        return all_results

    # ── 任务处理 ────────────────────────────────────────────────

    def _process_task(self, task_id: str):
        task_dir = Path(CONFIG["temp_dir"]) / task_id
        labels_dir = task_dir / "labels"
        video_path = task_dir / "input.mp4"
        if not labels_dir.exists():
            logger.warning(f"[GPU-{self.gpu_id}] 任务 {task_id} labels 目录不存在")
            return

        label_files = sorted(labels_dir.glob("*.json"))
        logger.info(f"[GPU-{self.gpu_id}] 任务 {task_id}: {len(label_files)} 帧, 开始读取视频...")
        _task_t0 = time.perf_counter()

        # ── 使用 video_utils 读取视频帧 ─────────────────────
        from video_utils import read_video_frames
        frames_rgb, indices = read_video_frames(
            str(video_path), return_indices=True,
        )
        n_total = indices[-1] + 1 if indices else 0
        # 帧索引 → BGR ndarray（OpenCV 操作需要 BGR）
        frames_bgr = {}
        for i, idx in enumerate(indices):
            frames_bgr[idx] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        del frames_rgb
        all_indices = sorted({int(lf.stem) for lf in label_files if int(lf.stem) < n_total})
        _t_read = time.perf_counter()
        logger.info(f"[GPU-{self.gpu_id}] 任务 {task_id}: 读取 {len(indices)} 帧 "
                    f"(范围 {indices[0]}-{indices[-1]}, 含标签{len(all_indices)}帧), "
                    f"耗时: {(_t_read - _task_t0) * 1000:.0f}ms")

        # ══════════════════════════════════════════════════════════
        #  Phase 1（CPU 并行）：从本地帧缓存裁剪所有手图
        # ══════════════════════════════════════════════════════════
        all_hands = []           # 所有手图
        all_mappings = []        # (label_file, person, side) 对应每张手图
        frame_labels_map = {}    # label_file → labels（写回时用）
        total_persons = 0
        total_hold = 0
        n_skipped = 0
        n_no_kps = 0       # 统计：有 label=1 但 keypoints 为空的帧数
        n_low_shoulder = 0  # 统计：肩部分数不足的帧数

        def _proc_one(lf):
            """单帧处理：加载 JSON → 裁剪手图"""
            fidx = int(lf.stem)
            if fidx not in frames_bgr:
                return None, False, None
            img = frames_bgr[fidx]
            try:
                with open(lf, 'r') as f:
                    labs = json.load(f)
            except Exception:
                return None, False, None

            h, w = img.shape[:2]
            hands, mappings = [], []
            any_person = False
            no_kps = False
            low_shoulder = False
            for p in labs:
                plabel = p.get("label")
                if (int(plabel) if plabel is not None else -1) != 1:
                    continue
                any_person = True
                kps = p.get("keypoints", [])
                kps_len = len(kps)
                if kps_len == 0:
                    no_kps = True
                if kps_len < 25:
                    continue
                ls_score, rs_score = kps[5][2], kps[6][2]
                if ls_score < CONFIG["keypoint_score_threshold"] or rs_score < CONFIG["keypoint_score_threshold"]:
                    low_shoulder = True
                    continue
                crops = self._crop_person_hands(p, img, w, h)
                if crops is None:
                    continue
                left, right = crops
                la = 0 if left is None else left.shape[0] * left.shape[1]
                ra = 0 if right is None else right.shape[0] * right.shape[1]
                if la < CONFIG["min_hand_area"] or ra < CONFIG["min_hand_area"]:
                    p["hold_rope"] = False
                    p["hand_left_hold"] = False
                    p["hand_right_hold"] = False
                    p["hand_left_score"] = 0.0
                    p["hand_right_score"] = 0.0
                else:
                    hands.append(left)
                    mappings.append((lf, p, "left"))
                    hands.append(right)
                    mappings.append((lf, p, "right"))
            return (lf, labs, hands, mappings), any_person, (no_kps, low_shoulder)

        with ThreadPoolExecutor(max_workers=4) as pool:
            # 诊断：主线程读取第一帧验证
            fut_map = {pool.submit(_proc_one, lf): lf for lf in label_files}
            for fut in as_completed(fut_map):
                result, has_person, stats = fut.result()
                if result is None:
                    n_skipped += 1
                    continue
                lf, labs, hands, mappings = result
                frame_labels_map[lf] = labs
                all_hands.extend(hands)
                all_mappings.extend(mappings)
                if not has_person:
                    n_skipped += 1
                if stats:
                    nk, ls = stats
                    if nk:
                        n_no_kps += 1
                    if ls:
                        n_low_shoulder += 1

        _phase1_end = time.perf_counter()
        phase1_ms = (_phase1_end - _t_read) * 1000
        logger.info(f"[GPU-{self.gpu_id}] 任务 {task_id}: 预处理完成, "
                    f"{len(all_hands)} 张手图, "
                    f"耗时: {phase1_ms:.0f}ms")
        if n_no_kps > 0:
            logger.warning(f"[GPU-{self.gpu_id}] 任务 {task_id}: {n_no_kps} 帧有 label=1 但 keypoints 为空")
        if n_low_shoulder > 0:
            logger.warning(f"[GPU-{self.gpu_id}] 任务 {task_id}: {n_low_shoulder} 帧肩部分数低于阈值")

        # ══════════════════════════════════════════════════════════
        #  Phase 2（GPU）：一次性 batch 分类所有手图
        # ══════════════════════════════════════════════════════════
        if all_hands:
            batch_results = self._classify_hands_batch(all_hands)
            for (label_file, person, side), (label, score) in zip(all_mappings, batch_results):
                if side == "left":
                    person["hand_left_hold"] = (label == 0)
                    person["hand_left_score"] = float(score)
                else:
                    person["hand_right_hold"] = (label == 0)
                    person["hand_right_score"] = float(score)

        _phase2_end = time.perf_counter()
        logger.info(f"[GPU-{self.gpu_id}] 任务 {task_id}: 分类完成, "
                    f"耗时: {(_phase2_end - _phase1_end) * 1000:.0f}ms")

        # ══════════════════════════════════════════════════════════
        #  Phase 3（CPU）：汇总结果 + 写回 JSON + 绘制到原视频
        # ══════════════════════════════════════════════════════════
        task_dir = Path(CONFIG["temp_dir"]) / task_id

        for idx, (label_file, labels) in enumerate(frame_labels_map.items()):
            frame_persons = 0
            frame_hold = 0
            for person in labels:
                label = person.get("label")
                label_int = int(label) if label is not None else -1
                if label_int != 1 or "hand_left_hold" not in person:
                    continue
                if "hand_right_hold" not in person:
                    continue
                hold = (person["hand_left_hold"] and person["hand_right_hold"])
                person["hold_rope"] = hold
                frame_persons += 1
                if hold:
                    frame_hold += 1

            self._write_json(label_file, labels)
            total_persons += frame_persons
            total_hold += frame_hold

        # 按原帧序逐帧绘制后合成为 output.mp4
        if n_total > 0 and video_path.exists():
            first_idx = indices[0]
            first_bgr = frames_bgr.get(first_idx)
            if first_bgr is not None:
                h, w = first_bgr.shape[:2]
                output_path = video_path.parent / "output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, 25.0, (w, h))

                for idx in indices:
                    frame_bgr = frames_bgr[idx]
                    draw = frame_bgr.copy()
                    labels = frame_labels_map.get(Path(labels_dir / f"{idx:06d}.json"))
                    if labels:
                        for p in labels:
                            plabel = p.get("label")
                            plabel_int = int(plabel) if plabel is not None else -1
                            if plabel_int != 1:
                                continue
                            x1, y1, x2, y2 = map(int, p["box"])
                            tid = p.get("track_id", -1)
                            if "hold_rope" in p:
                                hold = p["hold_rope"]
                                color = (0, 255, 0) if hold else (0, 0, 255)
                                text = "hold_rope" if hold else "no_hold"
                            else:
                                color = (128, 128, 128)
                                text = "unclassified"
                            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(draw, f"ID:{tid}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            cv2.putText(draw, text, (x1, y1 - 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    out.write(draw)

                out.release()

                logger.info(f"[GPU-{self.gpu_id}] 任务 {task_id}: 标注视频已保存 {output_path} ({len(indices)}帧, {len(frame_labels_map)}帧有标签)")
        
        
        video_path.unlink()
        logger.info(f"[GPU-{self.gpu_id}] 任务 {task_id}: "
                    f"{total_persons} 人体, {total_hold} 握绳, "
                    f"跳过 {n_skipped} 帧, "
                    f"总耗时: {(time.perf_counter() - _task_t0) * 1000:.0f}ms")

    # ── 运行入口 ────────────────────────────────────────────────

    def run_forever(self):
        logger.info(f"[GPU-{self.gpu_id}] Worker 启动，等待任务...")
        while True:
            task_id = self.task_queue.get()
            if task_id is None:
                break
            logger.info(f"[GPU-{self.gpu_id}] 收到任务: {task_id}")
            _t0 = time.perf_counter()
            try:
                self._process_task(task_id)
                self.result_queue.put(task_id)
            except Exception as e:
                logger.error(f"[GPU-{self.gpu_id}] 任务 {task_id} 失败: {e}", exc_info=True)


def _worker_main(gpu_id: int, task_q: multiprocessing.Queue,
                 result_q: multiprocessing.Queue):
    """Worker 进程入口。"""
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.addHandler(logging.FileHandler("hand_crop_service.log"))
    logger.addHandler(logging.StreamHandler())

    logger.info(f"[GPU-{gpu_id}] 子进程启动...")
    try:
        worker = GPUWorker(gpu_id, task_q, result_q)
        worker.run_forever()
    except Exception as e:
        logger.error(f"[GPU-{gpu_id}] 子进程异常退出: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════
#  主进程（任务调度）
# ══════════════════════════════════════════════════════════════════


class HandCropOrchestrator:
    """主进程：轮询任务 → 读取帧 → 分发 → 收集完成结果。"""

    def __init__(self):
        self.running = False
        self.workers: list[multiprocessing.Process] = []
        self._processed_tasks = set()
        self._load_processed_tasks()
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()

    def _load_processed_tasks(self):
        p = Path(CONFIG["processed_tasks_file_local"])
        if p.exists():
            try:
                with open(p, 'r') as f:
                    self._processed_tasks = set(json.load(f).get("tasks", []))
                logger.info(f"已加载 {len(self._processed_tasks)} 个已处理任务")
            except Exception as e:
                logger.error(f"加载已处理任务失败: {e}")

    def _save_processed_tasks(self):
        p = Path(CONFIG["processed_tasks_file_local"])
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                json.dump({"tasks": list(self._processed_tasks)}, f)
        except Exception as e:
            logger.error(f"保存已处理任务失败: {e}")

    def _get_new_task_ids(self) -> list[str]:
        tasks_file = Path(CONFIG["processed_tasks_file"])
        if not tasks_file.exists():
            return []
        try:
            with open(tasks_file, 'r') as f:
                all_tasks = json.load(f).get("tasks", [])
            return [t for t in all_tasks if t not in self._processed_tasks]
        except Exception as e:
            logger.error(f"读取 processed_tasks.json 失败: {e}")
            return []

    def _monitor_loop(self):
        logger.info(f"主监控循环启动，轮询间隔: {CONFIG['poll_interval']}s")
        while self.running:
            try:
                new_tasks = self._get_new_task_ids()
                if new_tasks:
                    logger.info(f"发现 {len(new_tasks)} 个新任务")
                    for tid in new_tasks:
                        video_path = Path(CONFIG["temp_dir"]) / tid / "input.mp4"
                        if video_path.exists():
                            logger.info(f"📖 任务 {tid}: 发现待处理")
                        else:
                            logger.warning(f"任务 {tid}: 视频文件不存在 {video_path}，标记为已完成")
                            self._processed_tasks.add(tid)
                            continue
                        self.task_queue.put(tid)
                        self._processed_tasks.add(tid)

                while not self.result_queue.empty():
                    try:
                        tid = self.result_queue.get_nowait()
                        self._processed_tasks.add(tid)
                        logger.info(f"任务 {tid} 已完成")
                    except Exception:
                        break
                if new_tasks:
                    self._save_processed_tasks()
            except Exception as e:
                logger.error(f"监控循环异常: {e}")

            if self.running:
                time.sleep(CONFIG["poll_interval"])

        logger.info("主监控循环已停止")

    def start(self):
        if self.running:
            logger.warning("服务已在运行")
            return
        self.running = True

        for gpu_id in CONFIG["gpu_ids"]:
            p = multiprocessing.Process(
                target=_worker_main,
                args=(gpu_id, self.task_queue, self.result_queue),
                name=f"GPU_{gpu_id}_worker"
            )
            p.start()
            self.workers.append(p)
            logger.info(f"Worker GPU {gpu_id} 已启动 (PID={p.pid})")

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"HandCropOrchestrator 已启动，{len(CONFIG['gpu_ids'])} 个 GPU Worker")

    def stop(self):
        logger.info("正在停止 HandCropOrchestrator...")
        self.running = False

        for _ in self.workers:
            self.task_queue.put(None)

        for w in self.workers:
            w.join(timeout=15)
            if w.is_alive():
                logger.warning(f"{w.name} 超时未退出，强制终止")
                w.terminate()
                w.join()

        logger.info("HandCropOrchestrator 已停止")


def main():
    orchestrator = HandCropOrchestrator()

    def signal_handler(signum, frame):
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    orchestrator.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
