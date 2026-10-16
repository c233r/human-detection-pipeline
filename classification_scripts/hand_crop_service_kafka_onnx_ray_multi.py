"""
HandCrop 手部裁剪分类服务（最终阶段）- 流水线架构
基于 Kafka 消息队列 + Ray 流水线并行推理
消费 hold_rope_keypoints topic 并执行手部裁剪分类
结果直接写入 label 文件，不再产出新 topic

架构：
  1. hand_preprocess (CPU Task): 读取视频帧，多线程裁剪人手
  2. HandGPUInferActor (GPU Actor): ONNX 批推理
  3. hand_postprocess (CPU Task): 映射分类结果，写入 labels 文件
  4. HandCropSchedulerActor: 调度流水线，维护 pipeline_depth
"""

import os
import sys
import json
import time
import math
import cv2
import numpy as np
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ray

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, _PROJECT_ROOT)

from utils.train_utils import get_info
from core.datasets.compose import Compose


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/hand_crop_onnx_ray.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("hand_crop_classify")


class Config:
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_CONSUME_TOPIC = "hold_rope_keypoints"
    KAFKA_PRODUCE_TOPIC = "final_videos"
    KAFKA_CONSUMER_GROUP = "hand_crop_consumers"

    TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")

    CLASSES_MAP = '/data2/caimoyao/Jump_rope/classification/datas/annotations.txt'
    CONFIG_PATH = '/data2/caimoyao/Jump_rope/classification/logs/ConvNeXt/2026-07-16-11-48-44/convnext_base_fix.py'
    ONNX_MODEL_PATH = '/data2/caimoyao/Jump_rope/classification/logs/ConvNeXt/2026-07-16-11-48-44/convnext_base_fix_fp16.onnx'
    INPUT_SIZE = 192
    GPU_IDS = [6]
    WORKERS_PER_GPU = 4  # 每张 GPU 最多 2 个 Worker (monitor + hand_crop 共用 GPU 4)
    RESOURCE_GPU = "gpu_4_node"  # 伪节点资源标签，锁定 GPU 4
    CONFIDENCE_THRESHOLD = 0.8

    HAND_SHOULDER_RATIO = 0.8
    MIN_HAND_AREA = 1800
    BATCH_CHUNK_SIZE = 512

    # RAY_OBJECT_STORE_MEMORY = 16 * 1024 ** 3

    LAG_CHECK_INTERVAL = 600
    SCALE_UP_THRESHOLD = 5
    SCALE_DOWN_THRESHOLD = 1
    MAX_TOTAL_WORKERS = 4
    MIN_TOTAL_WORKERS = 2

    PIPELINE_DEPTH = 3


config = Config()


# ==================== Kafka 初始化 ====================
def init_topics():
    """确保所需的 topic 存在。"""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
    })
    topics_needed = [config.KAFKA_CONSUME_TOPIC]
    try:
        metadata = admin.list_topics(timeout=10)
        to_create = [t for t in topics_needed if t not in metadata.topics]
    except Exception:
        to_create = topics_needed

    if not to_create:
        return

    new_topics = [NewTopic(t, num_partitions=1, replication_factor=1) for t in to_create]
    fs = admin.create_topics(new_topics)
    for t in to_create:
        try:
            fs[t].result(timeout=10)
            logger.info(f"✅ Topic {t} 已创建")
        except Exception as e:
            logger.warning(f"创建 Topic {t} 失败: {e}")


def ensure_kafka_partitions(total_consumers: int):
    """检查 consume topic 分区数，不足时扩展。"""
    from confluent_kafka.admin import AdminClient, NewPartitions

    topic = config.KAFKA_CONSUME_TOPIC
    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
    })

    try:
        metadata = admin.list_topics(timeout=10)
        if topic in metadata.topics:
            current_partitions = len(metadata.topics[topic].partitions)
        else:
            current_partitions = 0
    except Exception:
        current_partitions = 0

    logger.info(f"Topic {topic} 当前分区数: {current_partitions}, 消费者(worker)数: {total_consumers}")

    if current_partitions >= total_consumers:
        return

    logger.info(f"分区数不足，扩展至 {total_consumers} 个分区")
    fs = admin.create_partitions([NewPartitions(topic, total_consumers)])
    fs[topic].result(timeout=10)
    logger.info(f"✅ Topic {topic} 分区已扩展至 {total_consumers}")


# ==================== 图像处理工具函数 ====================
def _crop_person_hands(person: dict, image: np.ndarray, width: int, height: int):
    label = person.get("label")
    label_int = int(label) if label is not None else -1
    if label_int != 1:
        return None
    kps = person.get("keypoints", [])
    if len(kps) < 25:
        return None

    ls, rs = kps[5], kps[6]
    lh, rh = kps[23], kps[24]
    ls = [float(v) for v in ls]
    rs = [float(v) for v in rs]
    lh = [float(v) for v in lh]
    rh = [float(v) for v in rh]
    shoulder_dist = math.sqrt((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2)
    if shoulder_dist < 1:
        return None

    def _crop(kp):
        hx, hy, hs = kp
        s = int(shoulder_dist * config.HAND_SHOULDER_RATIO)
        x1 = int(max(0, hx - s)); x2 = int(min(width, hx + s))
        y1 = int(max(0, hy - s)); y2 = int(min(height, hy + s))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    left = _crop(lh)
    right = _crop(rh)
    return left, right


def _pad_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    m = max(h, w)
    ph = (m - h) // 2
    pw = (m - w) // 2
    return cv2.copyMakeBorder(img, ph, m - h - ph, pw, m - w - pw,
                              borderType=cv2.BORDER_CONSTANT, value=0)


# ==================== 流水线阶段 1: Preprocess (CPU) ====================
@ray.remote(num_cpus=2)
def hand_preprocess(task_data: dict, gpu_actor=None) -> dict:
    """
    预处理阶段（CPU 密集型，无状态）：
    - 读取 labels 目录，检查目录/文件是否存在
    - 读取视频帧
    - 多线程裁剪人手
    - 如果有 gpu_actor，直接把 frame_labels_map 存入 Actor（全程 1 份正本）
    """
    _log = _setup_worker_logger("hand_preprocess")

    task_id = task_data.get("task_id", "")
    output_labels = task_data.get("output_labels", "")
    video_path_str = task_data.get("video_path", "")
    _log.info(f"任务 {task_id}: 开始预处理, video={video_path_str}")
    _t0 = time.perf_counter()

    labels_dir = Path(output_labels)
    if not labels_dir.exists():
        _log.warning(f"任务 {task_id}: labels 目录不存在: {labels_dir}")
        _release_keeper_ref(task_id, _log)
        return _make_error_result(task_id, video_path_str, output_labels, "labels dir not found")

    video_path = Path(video_path_str) if video_path_str else None
    if video_path and not video_path.exists():
        _log.warning(f"任务 {task_id}: 视频文件不存在")
        _release_keeper_ref(task_id, _log)
        return _make_error_result(task_id, video_path_str, output_labels, "video not found")

    label_files = sorted(labels_dir.glob("*.json"))
    _log.info(f"任务 {task_id}: {len(label_files)} 帧")
    _task_t0 = time.perf_counter()

    # === 从 ObjectKeeper 获取帧数据（直接获取 numpy，零拷贝）===
    frames_bgr = {}
    try:
        keeper = ray.get_actor("ObjectKeeper", namespace="production")
        frames_rgb = ray.get(keeper.get_ref.remote(task_id), timeout=5)
        if frames_rgb is not None:
            for idx in range(len(frames_rgb)):
                frames_bgr[idx] = cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2BGR)
            del frames_rgb
            _get_ref_elapsed = (time.perf_counter() - _t0) * 1000
            _log.info(
                f"任务 {task_id}: 从 ObjectKeeper 获取 {len(frames_bgr)} 帧 (零拷贝), "
                f"耗时 {_get_ref_elapsed:.0f}ms"
            )
    except Exception as e:
        _log.warning(
            f"任务 {task_id}: 从 ObjectKeeper 获取帧数据失败: {e}, 回退到磁盘"
        )

    if not frames_bgr and video_path:
        _log.info(
            f"任务 {task_id}: ObjectKeeper 不可用，回退到磁盘读取: {video_path}"
        )
        from video_utils import read_video_frames
        frames_rgb, indices = read_video_frames(str(video_path), return_indices=True)
        n_total = indices[-1] + 1 if indices else 0
        for i, idx in enumerate(indices):
            frames_bgr[idx] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        del frames_rgb
        all_indices = sorted({int(lf.stem) for lf in label_files if int(lf.stem) < n_total})
        _log.info(
            f"任务 {task_id}: 从磁盘读取 {len(indices)} 帧, "
            f"含标签{len(all_indices)}帧, "
            f"耗时: {(time.perf_counter() - _task_t0) * 1000:.0f}ms"
        )

    all_hands = []
    all_mappings = []
    frame_labels_map = {}

    def _proc_one(lf):
        fidx = int(lf.stem)
        if fidx not in frames_bgr:
            return None
        img = frames_bgr[fidx]
        try:
            with open(lf, 'r') as f:
                labs = json.load(f)
        except Exception:
            return None

        h, w = img.shape[:2]
        hands, mappings = [], []
        for p in labs:
            plabel = p.get("label")
            if (int(plabel) if plabel is not None else -1) != 1:
                continue
            kps = p.get("keypoints", [])
            if len(kps) < 25:
                continue
            crops = _crop_person_hands(p, img, w, h)
            if crops is None:
                continue
            left, right = crops
            la = 0 if left is None else left.shape[0] * left.shape[1]
            ra = 0 if right is None else right.shape[0] * right.shape[1]
            if la < config.MIN_HAND_AREA or ra < config.MIN_HAND_AREA:
                p["hold_rope"] = False
                p["hand_left_hold"] = False
                p["hand_right_hold"] = False
                p["hand_left_score"] = 0.0
                p["hand_right_score"] = 0.0
            else:
                hands.append(left)
                mappings.append((str(lf), p, "left"))
                hands.append(right)
                mappings.append((str(lf), p, "right"))
        return (lf, labs, hands, mappings)

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_map = {pool.submit(_proc_one, lf): lf for lf in label_files}
        for fut in as_completed(fut_map):
            result = fut.result()
            if result is None:
                continue
            lf, labs, hands, mappings = result
            frame_labels_map[str(lf)] = labs
            all_hands.extend(hands)
            all_mappings.extend(mappings)

    _pre_elapsed = (time.perf_counter() - _t0) * 1000
    _log.info(
        f"任务 {task_id}: 预处理完成, {len(all_hands)} 张手图, "
        f"总耗时 {_pre_elapsed:.0f}ms"
    )

    # 直接 stack 成单块连续 4D 数组 (N, H, W, C)，避免 List[np.ndarray] 被 pickle 膨胀
    # 关键：统一 resize 到 INPUT_SIZE（模型输入尺寸），而不是 pad 到 max_size！
    # 否则若个别手图很大，全部手图都会被 pad 成巨型尺寸，内存爆炸（44k 张 × 567² = 43GB）
    all_hands_array = np.empty((0,), dtype=np.uint8)
    if all_hands:
        input_size = config.INPUT_SIZE
        resized_hands = []
        for hand in all_hands:
            square = _pad_square(hand)  # 先 pad 成正方形（保持宽高比）
            if square.shape[0] != input_size:
                square = cv2.resize(
                    square, (input_size, input_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            resized_hands.append(square)
        # 全部统一为 (N, 192, 192, 3) 的连续内存，Zero-Copy
        all_hands_array = np.stack(resized_hands, axis=0)

    # 直接存入 GPU Actor —— 全程只有 1 份正本，无需序列化传递
    if gpu_actor is not None:
        gpu_actor.store_data.remote(task_id, frame_labels_map)

    return {
        "task_id": task_id,
        "video_path_str": video_path_str,
        "output_labels": output_labels,
        "all_hands_array": all_hands_array,
        "all_mappings": all_mappings,
        "error": None,
    }


def _make_error_result(task_id, video_path_str, output_labels, error_msg):
    return {
        "task_id": task_id,
        "video_path_str": video_path_str,
        "output_labels": output_labels,
        "all_hands_array": np.empty((0,), dtype=np.uint8),
        "all_mappings": [],
        "frame_labels_map": {},
        "error": error_msg,
    }


def _release_keeper_ref(task_id: str, log: logging.Logger = None):
    """安全释放 ObjectKeeper 引用，错误路径必须调用。"""
    try:
        keeper = ray.get_actor("ObjectKeeper", namespace="production")
        keeper.release_ref.remote(task_id)
        if log:
            log.info(f"已释放 ObjectKeeper 引用: task={task_id}")
    except ValueError:
        pass
    except Exception as e:
        if log:
            log.warning(f"释放 ObjectKeeper 引用失败: task={task_id}, error={e}")


def _setup_worker_logger(name: str):
    """为 Ray Task/ Actor 设置独立的 logger（防止重复添加 handler）。"""
    logger_name = f"hand_crop.{name}"
    log = logging.getLogger(logger_name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    fh = logging.FileHandler('/data2/caimoyao/logs/hand_crop_onnx_ray.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(sh)
    return log


# ==================== 流水线阶段 2: GPU 推理 (Stateful Actor) ====================
@ray.remote
class HandGPUInferActor:
    """
    GPU 推理 Actor（有状态）：
    - 持有 ONNX 模型、val_pipeline、类别映射
    - 接收 preprocess 产出，执行批量推理
    - 返回 batch_results 及透传的映射信息
    """

    def __init__(self, worker_idx: int):
        self.worker_idx = worker_idx
        self._log = _setup_worker_logger(f"gpu_infer_{worker_idx}")
        self._frame_labels_maps = {}  # task_id -> frame_labels_map, 避免跨 Actor 序列化
        self.shutdown_event = False  # 预取线程检查停止信号

        _t_total = time.time()

        _t0 = time.time()
        self.classes_names, self.label_names = get_info(config.CLASSES_MAP)
        self._log.info(f"[GPU-{worker_idx}] 类别信息加载完成: {len(self.classes_names)} 个类别, 耗时 {time.time()-_t0:.2f}s")

        _t0 = time.time()
        import torch as _torch
        _torch.cuda.set_device(0)
        _torch.empty((1,), device='cuda:0')
        self._log.info(f"[GPU-{worker_idx}] CUDA 上下文已初始化 (cuda:0), 耗时 {time.time()-_t0:.2f}s")

        _t0 = time.time()
        from utils.train_utils import file2dict
        model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config.CONFIG_PATH)
        self.val_pipeline = val_pipeline
        self._log.info(f"[GPU-{worker_idx}] 配置文件解析完成, 耗时 {time.time()-_t0:.2f}s")

        _t0 = time.time()
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(config.ONNX_MODEL_PATH, sess_options, providers=providers)
        self._log.info(f"[GPU-{worker_idx}] ONNX 模型加载完成, 耗时 {time.time()-_t0:.2f}s")

        self._log.info(f"[GPU-{worker_idx}] ✅ Actor 预加载完成, 总耗时 {time.time()-_t_total:.2f}s")

    def ready(self):
        """用于检测 Actor 是否已完成初始化。"""
        return True

    def store_data(self, task_id: str, frame_labels_map: dict):
        """由 hand_preprocess 直接调用，把 frame_labels_map 存入 Actor，全程只有 1 份正本。"""
        self._frame_labels_maps[task_id] = frame_labels_map

    def run_postprocess(self, task_id: str, original_video_path: str, output_labels: str) -> dict:
        """在 GPU Actor 进程内直接完成后处理——全程 1 份正本，零序列化。"""
        _log = self._log

        _task_t0 = time.perf_counter()
        frame_labels_map = self._frame_labels_maps.get(task_id, {})

        if not frame_labels_map:
            _log.warning(f"任务 {task_id}: frame_labels_map 为空")
            _release_keeper_ref(task_id, _log)
            return {
                "task_id": task_id,
                "status": "failed",
                "error": "frame_labels_map 为空",
                "video_path": original_video_path,
                "output_labels": output_labels,
            }

        # 计算 hold_rope 并写入文件
        total_persons = 0
        total_hold = 0

        for label_file_str, labels in frame_labels_map.items():
            frame_persons = 0
            frame_hold = 0
            for person in labels:
                label = person.get("label")
                label_int = int(label) if label is not None else -1
                if label_int != 1:
                    continue
                if "hand_left_hold" not in person:
                    continue
                if "hand_right_hold" not in person:
                    continue
                hold = (person["hand_left_hold"] and person["hand_right_hold"])
                person["hold_rope"] = hold
                frame_persons += 1
                if hold:
                    frame_hold += 1

            try:
                with open(label_file_str, 'w') as f:
                    json.dump(labels, f, indent=2)
            except Exception as e:
                _log.error(f"任务 {task_id}: 写入 {label_file_str} 失败: {e}")

            total_persons += frame_persons
            total_hold += frame_hold

        _log.info(
            f"任务 {task_id}: {total_persons} 人体, {total_hold} 握绳, "
            f"后处理耗时: {(time.perf_counter() - _task_t0) * 1000:.0f}ms"
        )

        # 清理存储
        self._frame_labels_maps.pop(task_id, None)
        _release_keeper_ref(task_id, _log)

        return {
            "task_id": task_id,
            "status": "completed",
            "total_persons": total_persons,
            "total_hold": total_hold,
            "video_path": original_video_path,
            "output_labels": output_labels,
        }

    def infer(self, preprocessed_data: dict) -> dict:
        """
        执行 GPU 批量推理。
        输入: preprocessed_data 来自 hand_preprocess
        输出: dict 包含 batch_results 及透传的 mapping 信息
        """
        task_id = preprocessed_data.get("task_id", "")
        error = preprocessed_data.get("error")
        self._log.info(f"[GPU-{self.worker_idx}] 任务 {task_id}: 开始推理")

        if error is not None:
            self._log.warning(f"[GPU-{self.worker_idx}] 任务 {task_id} 预处理已失败, 跳过推理")
            _release_keeper_ref(task_id, self._log)
            return {
                "task_id": task_id,
                "batch_results": [],
                "video_path_str": preprocessed_data.get("video_path_str", ""),
                "output_labels": preprocessed_data.get("output_labels", ""),
                "error": error,
            }

        hand_images = preprocessed_data.get("all_hands_array", np.empty((0,), dtype=np.uint8))
        all_mappings = preprocessed_data.get("all_mappings", [])
        video_path_str = preprocessed_data.get("video_path_str", "")
        output_labels = preprocessed_data.get("output_labels", "")

        # frame_labels_map 已由 hand_preprocess 通过 store_data() 存入 Actor，直接取正本
        frame_labels_map = self._frame_labels_maps.get(task_id, {})

        if len(hand_images) == 0:
            self._log.info(f"[GPU-{self.worker_idx}] 任务 {task_id}: 无手部图像, 跳过推理")
            _release_keeper_ref(task_id, self._log)
            return {
                "task_id": task_id,
                "batch_results": [],
                "video_path_str": video_path_str,
                "output_labels": output_labels,
                "error": None,
            }

        try:
            # 批量推理（多线程后台预取 + GPU 显存预读）
            all_results = []
            chunk_size = config.BATCH_CHUNK_SIZE
            n_chunks = (len(hand_images) + chunk_size - 1) // chunk_size
            _batch_t0 = time.perf_counter()

            pipeline = Compose(
                [t for t in self.val_pipeline if t.get('type') != 'LoadImageFromFile']
            )

            # ══════════════════════════════════════════════════════════
            # CPU/GPU 并行流水线：预取线程负责 CPU 预处理，主线程负责 GPU 推理
            #   ┌──────────┐   batch_np(下一批)  ┌──────────┐
            #   │ 预取线程  │ ──────────────────▶ │  主线程   │
            #   │ (CPU)     │ ◀────────────────── │ (GPU推理) │
            #   └──────────┘   batch_q(maxsize=2) └──────────┘
            # 推理 batch N 期间，预取线程已在预处理 batch N+1
            # ══════════════════════════════════════════════════════════
            import queue as _queue
            import threading as _threading
            from concurrent.futures import ThreadPoolExecutor

            PREFETCH_DEPTH = 2  # 同时驻留 2 个 chunk（内存 ~226MB）
            batch_q = _queue.Queue(maxsize=PREFETCH_DEPTH)

            # chunk 内多线程执行 pipeline（CPU 密集），进一步压榨 CPU 核
            preprocess_pool = ThreadPoolExecutor(max_workers=4)

            def _preprocess_chunk(chunk):
                tensors = list(
                    preprocess_pool.map(
                        lambda img: pipeline(dict(img=img, filename=None))['img'],
                        chunk,
                    )
                )
                return np.stack([t.numpy() for t in tensors], axis=0).astype(np.float16)

            def _prefetch_worker():
                """后台预取：CPU pipeline 预处理，产出 batch_np 交给主线程。"""
                try:
                    for ci in range(n_chunks):
                        if self.shutdown_event:
                            break
                        batch_np = _preprocess_chunk(
                            hand_images[ci * chunk_size:(ci + 1) * chunk_size])
                        batch_q.put((ci, batch_np))
                except Exception as _e:
                    # 记录完整 traceback，并通知主线程 fail-fast
                    self._log.error(
                        f"[GPU-{self.worker_idx}] 任务 {task_id}: 预取线程异常: {_e!r}",
                        exc_info=True,
                    )
                    batch_q.put(("ERROR", _e))
                finally:
                    batch_q.put(None)  # 结束哨兵

            prefetch_thread = _threading.Thread(
                target=_prefetch_worker,
                daemon=True,
                name=f"hand-prefetch-{self.worker_idx}",
            )
            prefetch_thread.start()

            # 主循环：CPU 预取与 GPU 推理并行
            n_done = 0
            while n_done < n_chunks:
                item = batch_q.get()
                if item is None:
                    self._log.warning(f"[GPU-{self.worker_idx}] 任务 {task_id}: 预取线程提前结束")
                    break
                if isinstance(item, tuple) and item[0] == "ERROR":
                    # 预取线程异常 → 任务直接失败，避免静默输出错误分类结果
                    raise item[1]
                ci, batch_np = item

                # GPU 推理（session.run 内部负责 H2D 拷贝 + CUDA kernel）
                scores = self.session.run(['output'], {'input': batch_np})[0]

                pred_labels = np.argmax(scores, axis=1)
                pred_scores = np.max(scores, axis=1)

                chunk_results = list(zip(pred_labels.tolist(), pred_scores.tolist()))
                all_results.extend(chunk_results)
                n_done += 1
                del batch_np  # 显式释放 CPU 大数组

            # 等待预取线程退出，确保资源清理
            prefetch_thread.join(timeout=5)

            # ── 显式释放预取资源，防止内存泄漏 ──
            preprocess_pool.shutdown(wait=True)
            del batch_q

            self._log.info(
                f"[GPU-{self.worker_idx}] 任务（预取情况） {task_id}: batch 分类 {len(hand_images)} 张手图, "
                f"{n_chunks} chunk, "
                f"总耗时: {(time.perf_counter() - _batch_t0) * 1000:.0f}ms"
            )

            # 直接修改 Actor 内存储的 frame_labels_map —— 改的是"正本"，不是副本
            stored_map = self._frame_labels_maps.get(task_id, {})
            for (label_file_str, person, side), (label, score) in zip(all_mappings, all_results):
                frame_labels = stored_map.get(label_file_str, [])
                for p in frame_labels:
                    if p is person or (p.get("track_id") == person.get("track_id")):
                        if side == "left":
                            p["hand_left_hold"] = (label == 0)
                            p["hand_left_score"] = float(score)
                        else:
                            p["hand_right_hold"] = (label == 0)
                            p["hand_right_score"] = float(score)
                        break

            return {
                "task_id": task_id,
                "batch_results": all_results,
                "video_path_str": video_path_str,
                "output_labels": output_labels,
                "error": None,
            }
        finally:
            # 清理中间变量，防止 Python 堆内存碎片
            import gc
            gc.collect()


# ==================== 流水线调度器 ====================
@ray.remote
class HandCropSchedulerActor:
    """
    流水线调度 Actor（轻量，无 GPU）：
    - 轮询 Kafka，对消息启动 preprocess -> gpu_infer -> postprocess 流水线
    - 维护 pipeline_depth，实现 CPU 和 GPU 重叠
    - 处理各阶段结果/错误
    """

    def __init__(self, worker_idx: int, gpu_actor):
        self.worker_idx = worker_idx
        self.gpu_actor = gpu_actor
        self.shutdown_event = False
        self.pipeline_depth = config.PIPELINE_DEPTH
        self.pending = {}  # task_id -> {post_ref, task_data, start_time, msg}

        self._log = _setup_worker_logger(f"scheduler_{worker_idx}")

        _t0 = time.time()
        self._init_kafka()
        self._log.info(f"[Scheduler-{worker_idx}] Kafka 初始化完成, 耗时 {time.time()-_t0:.2f}s")
        self._log.info(f"[Scheduler-{worker_idx}] ✅ Actor 预加载完成")

    def _init_kafka(self):
        from confluent_kafka import Consumer, Producer

        def on_assign(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._log.info(f"[Scheduler-{self.worker_idx}] 分配到分区: {p_ids}")

        def on_revoke(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._log.info(f"[Scheduler-{self.worker_idx}] 被回收分区: {p_ids}")

        def on_lost(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._log.warning(f"[Scheduler-{self.worker_idx}] 丢失分区: {p_ids}")

        self.kafka_consumer = Consumer({
            "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": config.KAFKA_CONSUMER_GROUP,
            "group.instance.id": f"handcrop-ray-{self.worker_idx}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": 14400000,
            "session.timeout.ms": 100000,
            "heartbeat.interval.ms": 3000,
            "partition.assignment.strategy": "cooperative-sticky",
            "broker.address.family": "v4",
            "queued.min.messages": 1,
        })
        self.kafka_consumer.subscribe(
            [config.KAFKA_CONSUME_TOPIC],
            on_assign=on_assign,
            on_revoke=on_revoke,
            on_lost=on_lost,
        )
        self._log.info(f"[Scheduler-{self.worker_idx}] Kafka Consumer 已连接")

        self.kafka_producer = Producer({
            "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
            "broker.address.family": "v4",
        })
        self._log.info(f"[Scheduler-{self.worker_idx}] Kafka Producer 已连接")

    def _send_result(self, result_msg: dict):
        """发送结果消息到 Kafka。"""
        task_id = result_msg.get("task_id", "")
        try:
            self.kafka_producer.produce(
                config.KAFKA_PRODUCE_TOPIC,
                key=task_id.encode('utf-8'),
                value=json.dumps(result_msg).encode('utf-8'),
            )
            self.kafka_producer.flush(timeout=10)
            self._log.info(f"[Scheduler-{self.worker_idx}] 结果已发送 -> {config.KAFKA_PRODUCE_TOPIC}: {task_id}")
            
            # 释放 ObjectKeeper 引用（串行流水线，能处理到此处说明上一阶段已完成）
            _release_keeper_ref(task_id, self._log)
                
        except Exception as e:
            self._log.error(f"[Scheduler-{self.worker_idx}] task {task_id}: 发送结果失败: {e}")
            _release_keeper_ref(task_id, self._log)

    def _poll_and_launch(self):
        """从 Kafka 拉取消息并启动流水线（不阻塞）。"""
        try:
            msg = self.kafka_consumer.poll(timeout=0.5)
        except Exception as e:
            self._log.error(f"[Scheduler-{self.worker_idx}] Kafka poll 异常: {e}")
            return

        if msg is None:
            return

        if msg.error():
            self._log.error(f"[Scheduler-{self.worker_idx}] Kafka poll error: {msg.error()}")
            return

        try:
            task_data = json.loads(msg.value().decode('utf-8'))
        except Exception as e:
            self._log.error(f"[Scheduler-{self.worker_idx}] 消息解析失败: {e}")
            self.kafka_consumer.commit(message=msg)
            return

        task_id = task_data.get("task_id", "")
        status = task_data.get("status", "")

        if status != "completed":
            self._log.debug(f"[Scheduler-{self.worker_idx}] 跳过非 completed 消息: {task_id}")
            self.kafka_consumer.commit(message=msg)
            return

        if task_id in self.pending:
            self._log.warning(f"[Scheduler-{self.worker_idx}] 任务 {task_id} 已在处理中, 跳过")
            self.kafka_consumer.commit(message=msg)
            return

        output_labels = task_data.get("output_labels", "")
        video_path_str = task_data.get("video_path", "")

        if not output_labels:
            self._log.error(f"[Scheduler-{self.worker_idx}] 任务 {task_id}: output_labels 为空")
            self._send_result({
                "task_id": task_id,
                "video_path": video_path_str,
                "error": "output_labels 为空",
                "failed_at": "hand_crop",
                "status": "failed",
            })
            self.kafka_consumer.commit(message=msg)
            return

        self._log.info(f"[Scheduler-{self.worker_idx}] 启动流水线: {task_id}")

        # 启动流水线: preprocess -> gpu_infer -> run_postprocess (全部在 GPU Actor 内完成，零序列化)
        pre_ref = hand_preprocess.remote(task_data, self.gpu_actor)
        infer_ref = self.gpu_actor.infer.remote(pre_ref)
        post_ref = self.gpu_actor.run_postprocess.remote(
            task_id,
            task_data.get("video_path", ""),
            task_data.get("output_labels", ""),
        )

        # pre_ref 传给 infer 后即可释放，infer_ref 是 fire-and-forget（直接改 Actor 内部 _frame_labels_maps）
        # 只有 post_ref 需要等待完成
        self.pending[task_id] = {
            "post_ref": post_ref,
            "task_data": task_data,
            "start_time": time.time(),
            "msg": msg,
        }

    def _check_completed(self):
        """检查已完成后处理的任务，发送结果，并处理超时任务。"""
        if not self.pending:
            return

        now = time.time()
        # task_timeout = 1800  # 30 分钟超时
        # expired_ids = []

        # 1. 检查超时任务
        # for tid, info in self.pending.items():
        #     if now - info["start_time"] > task_timeout:
        #         expired_ids.append(tid)

        # for tid in expired_ids:
        #     self._log.warning(f"[Scheduler-{self.worker_idx}] 任务 {tid} 超时，强制清理")
        #     info = self.pending.pop(tid)
        #     _release_keeper_ref(tid, self._log)
        #     try:
        #         self.kafka_consumer.commit(message=info["msg"], asynchronous=False)
        #     except Exception:
        #         pass

        # if not self.pending:
        #     return

        # 2. 收集所有 post_ref
        post_refs = {tid: info["post_ref"] for tid, info in self.pending.items()}

        done_refs = []
        try:
            done_refs, _ = ray.wait(list(post_refs.values()), timeout=0, num_returns=len(post_refs))
        except Exception:
            # 如果 wait 因为太多 ready ref 出错，逐个检查
            for tid, ref in post_refs.items():
                try:
                    ready, _ = ray.wait([ref], timeout=0)
                    if ready:
                        done_refs.append(ref)
                except Exception:
                    pass

        if not done_refs:
            return

        # 找出哪些 task 的 post_ref 已完成
        done_task_ids = []
        for ref in done_refs:
            for tid, info in self.pending.items():
                if info["post_ref"] is ref:
                    done_task_ids.append(tid)
                    break

        for task_id in done_task_ids:
            info = self.pending.pop(task_id)
            task_data = info["task_data"]
            elapsed = time.time() - info["start_time"]

            try:
                result = ray.get(info["post_ref"])
                status = result.get("status", "failed")
                self._log.info(
                    f"[Scheduler-{self.worker_idx}] 流水线完成: {task_id}, "
                    f"status={status}, 耗时: {elapsed * 1000:.0f}ms"
                )

                if status == "completed":
                    result_msg = {
                        "task_id": task_id,
                        "video_path": result.get("video_path", ""),
                        "output_labels": result.get("output_labels", ""),
                        "status": "completed",
                    }
                else:
                    result_msg = {
                        "task_id": task_id,
                        "video_path": result.get("video_path", ""),
                        "error": result.get("error", "unknown error"),
                        "failed_at": "hand_crop",
                        "status": "failed",
                    }

                self._send_result(result_msg)
                self.kafka_consumer.commit(message=info["msg"], asynchronous=False)  # 后处理完成后提交offset

            except Exception as e:
                self._log.error(
                    f"[Scheduler-{self.worker_idx}] 任务 {task_id} 流水线执行失败: {e}"
                )
                _release_keeper_ref(task_id, self._log)  # 失败也释放 ObjectKeeper 引用
                original_video = task_data.get("video_path", "")
                self._send_result({
                    "task_id": task_id,
                    "video_path": original_video,
                    "error": str(e),
                    "failed_at": "hand_crop",
                    "status": "failed",
                })
                # 确保失败任务也 commit offset，防止死循环消费
                try:
                    self.kafka_consumer.commit(message=info["msg"], asynchronous=False)
                except Exception:
                    pass

    def run(self):
        """主循环：轮询 Kafka + 检查完成 + 启动新流水线。"""
        self._log.info(f"[Scheduler-{self.worker_idx}] 开始运行流水线调度 (pipeline_depth={self.pipeline_depth})")

        while not self.shutdown_event:
            # 1. 检查已完成后处理的任务
            self._check_completed()

            # 2. 如果流水线有空闲槽位，拉取新消息
            while len(self.pending) < self.pipeline_depth:
                old_count = len(self.pending)
                self._poll_and_launch()
                if len(self.pending) == old_count:
                    # 没有新任务被启动
                    break

            # 3. 短暂休眠避免忙等
            if len(self.pending) < self.pipeline_depth:
                time.sleep(0.1)
            else:
                time.sleep(0.05)

        # 清理
        self._log.info(f"[Scheduler-{self.worker_idx}] 等待进行中的任务完成...")
        if self.pending:
            all_post_refs = [info["post_ref"] for info in self.pending.values()]
            try:
                ray.get(all_post_refs, timeout=300)
            except Exception:
                pass
            self._check_completed()

        # 强制释放所有残留任务的 ObjectKeeper 引用
        for task_id in self.pending:
            _release_keeper_ref(task_id, self._log)
        self.pending.clear()

        self.kafka_consumer.close()
        self.kafka_producer.close()
        self._log.info(f"[Scheduler-{self.worker_idx}] 调度器退出")

    def stop(self):
        self.shutdown_event = True

    def ready(self):
        """用于检测 Actor 是否已完成初始化。"""
        return True


# ==================== Kafka 积压监控 ====================
@ray.remote
class KafkaLagMonitor:
    def __init__(self, bootstrap_servers: str, topics: list, consumer_groups: list):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.consumer_groups = consumer_groups

    def get_lag(self) -> dict:
        from confluent_kafka import Consumer, TopicPartition
        
        kafka_conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "broker.address.family": "v4",
            "group.id": "lag_monitor_tmp",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        results = {}
        for topic, group in zip(self.topics, self.consumer_groups):
            c = None
            try:
                c = Consumer(kafka_conf)
                
                # 1. 获取 Topic 元数据与分区数
                md = c.list_topics(topic, timeout=10)
                if topic not in md.topics or md.topics[topic].error:
                    results[topic] = {"lag": -1, "error": f"topic {topic} 不存在或不可用"}
                    continue

                partitions_meta = md.topics[topic].partitions
                tps = [TopicPartition(topic, p) for p in partitions_meta.keys()]

                # 2. 查询真实消费者组对这些分区的 committed offset
                try:
                    c.close()
                    kafka_conf["group.id"] = group
                    c = Consumer(kafka_conf)
                    committed_tps = c.committed(tps, timeout=10)
                except Exception as e:
                    results[topic] = {"lag": -1, "error": f"committed offset 查询失败: {e}"}
                    continue

                committed_map = {tp.partition: tp.offset for tp in committed_tps}

                total_lag = 0
                partition_lags = {}
                max_partition_lag = 0

                # 3. 遍历每个分区查 watermark (low, high)
                for p_id in partitions_meta.keys():
                    try:
                        low, high = c.get_watermark_offsets(TopicPartition(topic, p_id), timeout=10)
                        committed_val = committed_map.get(p_id, -1001)
                        
                        # 如果 offset < 0 (例如 OFFSET_INVALID -1001)，从 low 开始算
                        if committed_val < 0:
                            committed_val = low

                        lag = max(0, high - committed_val)
                        total_lag += lag
                        partition_lags[p_id] = lag
                        if lag > max_partition_lag:
                            max_partition_lag = lag
                    except Exception:
                        partition_lags[p_id] = -1

                results[topic] = {
                    "lag": total_lag,
                    "partitions": len(partitions_meta),
                    "partition_lags": partition_lags,
                    "max_partition_lag": max_partition_lag,
                }
            except Exception as e:
                results[topic] = {"lag": -1, "error": str(e)}
            finally:
                if c is not None:
                    c.close()

        return results


# ==================== 辅助函数 ====================
def _get_gpu_free_memory():
    _result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    _gpu_free = {}
    for line in _result.stdout.strip().split("\n"):
        parts = line.split(", ")
        if len(parts) == 2:
            _gpu_free[int(parts[0])] = int(parts[1])
    return _gpu_free


def _calculate_max_workers(gpu_free, max_total):
    WORKER_MEM_MB = 11 * 1024
    total_workers = 0
    for gpu_id in config.GPU_IDS:
        free_mb = gpu_free.get(gpu_id, 0)
        n_workers = min(free_mb // WORKER_MEM_MB, config.WORKERS_PER_GPU)
        total_workers += n_workers
    return min(total_workers, max_total)


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("   HandCrop 手部裁剪分类服务 (Ray 流水线架构)")
    print("=" * 60)
    print(f"Kafka: {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"消费: {config.KAFKA_CONSUME_TOPIC}")
    print(f"消费者组: {config.KAFKA_CONSUMER_GROUP}")
    print(f"GPU: {config.GPU_IDS}")
    print(f"流水线深度: {config.PIPELINE_DEPTH}")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.GPU_IDS))
    print(f"已设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    ray.init(
        address="auto",
        runtime_env={"env_vars": {"PYTHONPATH": _PROJECT_ROOT, "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.GPU_IDS)), "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/targets/x86_64-linux/lib"}},
    )
    logger.info(f"✅ 已连接到 Ray 集群, GPU可用: {ray.available_resources().get('GPU', 0)}")

    init_topics()
    ensure_kafka_partitions(config.MIN_TOTAL_WORKERS)

    # 每个 worker = 1 个 GPU Actor + 1 个 Scheduler Actor
    gpu_actors = []
    scheduler_actors = []

    gpu_fraction = 1.0 / config.WORKERS_PER_GPU
    gpu_resource = {config.RESOURCE_GPU: 1.0}  # 锁定到伪节点，每个 Worker 占用 1 单位资源

    _t_preload = time.time()
    for w_idx in range(config.MIN_TOTAL_WORKERS):
        _t_w = time.time()
        gpu_actor = HandGPUInferActor.options(
            num_gpus=gpu_fraction,
            resources=gpu_resource
        ).remote(w_idx)
        scheduler = HandCropSchedulerActor.remote(w_idx, gpu_actor)
        # 等待 Actor 初始化完成
        ray.get([gpu_actor.ready.remote(), scheduler.ready.remote()])
        gpu_actors.append(gpu_actor)
        scheduler_actors.append(scheduler)
        logger.info(
            f"Worker-{w_idx} ✅ 预加载完成 (GPU {gpu_fraction:.2f}, "
            f"resource={config.RESOURCE_GPU} + Scheduler), 耗时 {time.time()-_t_w:.2f}s"
        )
    logger.info(f"🚀 共创建 {config.MIN_TOTAL_WORKERS} 个 Worker, 预加载总耗时 {time.time()-_t_preload:.2f}s")

    for s in scheduler_actors:
        s.run.remote()

    lag_monitor = KafkaLagMonitor.remote(
        config.KAFKA_BOOTSTRAP_SERVERS,
        [config.KAFKA_CONSUME_TOPIC],
        [config.KAFKA_CONSUMER_GROUP],
    )

    last_scale_time = time.time()
    scale_cooldown = 60

    try:
        while True:
            time.sleep(config.LAG_CHECK_INTERVAL)

            current_time = time.time()
            if current_time - last_scale_time < scale_cooldown:
                continue

            try:
                lag_info = ray.get(lag_monitor.get_lag.remote())
                topic_lag = lag_info.get(config.KAFKA_CONSUME_TOPIC, {})
                lag = topic_lag.get("lag", 0)

                if lag < 0:
                    logger.warning(f"⚠️ Kafka 积压监控获取失败: {topic_lag.get('error', '未知错误')}")
                    continue

                partitions = topic_lag.get("partitions", 0)
                max_partition_lag = topic_lag.get("max_partition_lag", 0)
                partition_lags = topic_lag.get("partition_lags", {})
                logger.info(
                    f"Kafka 积压检测: {config.KAFKA_CONSUME_TOPIC} "
                    f"total_lag={lag}, partitions={partitions}, "
                    f"max_partition_lag={max_partition_lag}, 当前 workers={len(scheduler_actors)}"
                )
                if partition_lags:
                    logger.info(f"各分区积压: {partition_lags}")

                num_workers = len(scheduler_actors)
                lag_per_worker = lag / num_workers if num_workers > 0 else 0
                need_scale_up = False

                if lag > 0:
                    if num_workers == 0:
                        need_scale_up = True
                        logger.info(f"📈 无消费者但 topic 有积压 (lag={lag}), 需要立即添加消费者")
                    elif lag_per_worker > config.SCALE_UP_THRESHOLD and num_workers < config.MAX_TOTAL_WORKERS:
                        need_scale_up = True
                        logger.info(f"📈 单消费者积压超过阈值 (lag={lag}, workers={num_workers}, lag/worker={lag_per_worker:.1f} > {config.SCALE_UP_THRESHOLD})")

                if need_scale_up:
                    gpu_free = _get_gpu_free_memory()
                    max_possible = _calculate_max_workers(gpu_free, config.MAX_TOTAL_WORKERS)

                    if max_possible > 0:
                        logger.info(f"📈 扩展 1 个 worker")

                        ensure_kafka_partitions(num_workers + 1)

                        w_idx = num_workers
                        gpu_actor = HandGPUInferActor.options(
                            num_gpus=gpu_fraction,
                            resources=gpu_resource
                        ).remote(w_idx)
                        gpu_actors.append(gpu_actor)
                        scheduler = HandCropSchedulerActor.remote(w_idx, gpu_actor)
                        scheduler_actors.append(scheduler)
                        logger.info(f"🚀 新增 Worker-{w_idx} (GPU+Scheduler, resource={config.RESOURCE_GPU})")
                        scheduler.run.remote()

                        last_scale_time = current_time

                elif lag_per_worker < config.SCALE_DOWN_THRESHOLD and len(scheduler_actors) > config.MIN_TOTAL_WORKERS:
                    logger.info(
                        f"📉 平均每 worker 积压低于阈值 "
                        f"(lag_per_worker={lag_per_worker:.1f} < {config.SCALE_DOWN_THRESHOLD}), "
                        f"收缩 1 个 worker"
                    )

                    if len(scheduler_actors) == 0:
                        pass
                    else:
                        s_to_remove = scheduler_actors.pop()
                        gpu_to_remove = gpu_actors.pop()

                        # 1. 尝试优雅停止 Scheduler
                        try:
                            ray.get(s_to_remove.stop.remote(), timeout=5)
                        except Exception:
                            pass

                        # 2. 核心：强行 Kill 掉两个 Actor 进程，彻底释放 GPU 显存与私有堆内存
                        ray.kill(s_to_remove)
                        ray.kill(gpu_to_remove)
                        logger.info(f"⏹️ 已强行销毁 Worker Actor 对 (Scheduler + GPU) 并释放显存/内存")

                    last_scale_time = current_time

            except Exception as e:
                logger.error(f"动态扩缩容失败: {e}")

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务...")
    finally:
        logger.info(f"停止 {len(scheduler_actors)} 个 Scheduler Actor...")
        try:
            ray.get([s.stop.remote() for s in scheduler_actors], timeout=10)
        except Exception:
            pass
        for s in scheduler_actors:
            ray.kill(s)

        logger.info(f"停止 {len(gpu_actors)} 个 GPU Actor...")
        for g in gpu_actors:
            ray.kill(g)

        logger.info("停止 LagMonitor...")
        ray.kill(lag_monitor)

        time.sleep(3)
        ray.shutdown()
        logger.info("服务已关闭")


if __name__ == "__main__":
    main()
