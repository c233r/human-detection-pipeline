"""

EdgeCrafter 推理服务 (Ray版本)
基于 Kafka 消息队列 + Ray Actor 流水线分布式推理（方案B：流水线并行）
"""

import os
import sys

# ──────────────────────────────────────────────────────────────
# TensorRT 环境变量（必须在 import onnxruntime 之前设置）
# ──────────────────────────────────────────────────────────────
TRT_PATH = "/data2/caimoyao/TensorRT-10.16.1.11"
os.environ["TRT_PATH"] = TRT_PATH
os.environ["LD_LIBRARY_PATH"] = f"{TRT_PATH}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = f"{TRT_PATH}/bin:" + os.environ.get("PATH", "")
import re
import json
import time
import signal
import threading
import queue
from pathlib import Path
from typing import Dict, Any
import logging
import subprocess

import ray


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/api_service_ray.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def get_worker_logger(name: str) -> logging.Logger:
    """创建/获取带文件+控制台 handler 的独立 logger（幂等，避免重复添加 handler）。

    与 GPU Actor 的 _worker_logger 注册方式一致：propagate=False，
    日志同时写入 /data2/caimoyao/logs/api_service_ray.log 和控制台。
    Ray 任务函数会复用 worker 进程，故用 handlers 判空做幂等保护。
    """
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler('/data2/caimoyao/logs/api_service_ray.log', encoding='utf-8')
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log



class Config:
    PROJECT_ROOT = Path("/data2/caimoyao/EdgeCrafter-main/ecdetseg")
    CONFIG_PATH = PROJECT_ROOT / "configs/ecdet/ecdet_l.yml"
    CHECKPOINT_PATH = PROJECT_ROOT / "models/last.pth"

    CONF_THRESHOLD = 0.5
    CLASSES = ["person", "face", "ball"]

    OUTPUT_DIR = Path("/data2/caimoyao/api_outputs")

    GPU_IDS = [6]
    WORKERS_PER_GPU = 4  # 每张 GPU 最多 2 个 Worker
    RESOURCE_GPU = "gpu_5_node"  # 伪节点资源标签，锁定 GPU 5
    NUM_STREAMS = 4

    RAG_CACHE = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/rag_db.pkl"
    RAG_THRESHOLD = 0.75

    SAVE_OUTPUT = False

    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_TASK_TOPIC = "video_tasks"
    KAFKA_RESULT_TOPIC = "video_results"
    KAFKA_CONSUMER_GROUP = "video_task_consumers"

    KAFKA_MAX_POLL_INTERVAL_MS = 14400000
    KAFKA_SESSION_TIMEOUT_MS = 60000
    KAFKA_HEARTBEAT_INTERVAL_MS = 3000

    RAY_OBJECT_STORE_MEMORY = None  # 设为 None 时自动计算

    PIPELINE_DEPTH = 1

    @staticmethod
    def get_object_store_memory():
        """按需计算 object store memory（上限，不提前占用物理内存）。"""
        if Config.RAY_OBJECT_STORE_MEMORY is not None:
            return Config.RAY_OBJECT_STORE_MEMORY
        # 初始按 MIN_TOTAL_WORKERS 算，后续靠 spill + _system_config 兜底
        per_worker_gb = 25
        total_gb = Config.MIN_TOTAL_WORKERS * per_worker_gb
        total_gb = max(total_gb, 60)  # 至少 60GB
        logger.info(f"⚙️ 动态计算 object_store_memory: {total_gb}GB (MIN_TOTAL_WORKERS={Config.MIN_TOTAL_WORKERS})")
        return total_gb * 1024 ** 3

    LAG_CHECK_INTERVAL = 900
    SCALE_UP_THRESHOLD = 30
    SCALE_DOWN_THRESHOLD = 1
    MAX_TOTAL_WORKERS = 4
    MIN_TOTAL_WORKERS = 2
    # 背压机制：Ray object store 内存使用超过该值 (GB) 时，禁止扩容并强制缩减 worker
    BACKPRESSURE_MEMORY_GB = 80


config = Config()


def init_topics():
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
        "security.protocol": "PLAINTEXT",
    })
    topics_to_create = []
    for topic_name in [config.KAFKA_TASK_TOPIC, config.KAFKA_RESULT_TOPIC]:
        try:
            metadata = admin.list_topics(timeout=10)
            if topic_name not in metadata.topics:
                topics_to_create.append(topic_name)
        except Exception:
            topics_to_create.append(topic_name)

    if not topics_to_create:
        logger.info("所有 topic 已存在，无需创建")
        return

    new_topics = [NewTopic(t, num_partitions=1, replication_factor=1) for t in topics_to_create]
    fs = admin.create_topics(new_topics)
    for t in topics_to_create:
        try:
            fs[t].result(timeout=10)
            logger.info(f"✅ Topic {t} 已创建")
        except Exception as e:
            logger.warning(f"创建 Topic {t} 失败: {e}")


def ensure_kafka_partitions(total_consumers: int):
    """检查 topic 分区数，不足时扩展（不负责创建 topic）。"""
    from confluent_kafka.admin import AdminClient, NewPartitions

    topic = config.KAFKA_TASK_TOPIC
    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
        "security.protocol": "PLAINTEXT",
    })

    try:
        metadata = admin.list_topics(timeout=10)
        if topic in metadata.topics:
            current_partitions = len(metadata.topics[topic].partitions)
        else:
            current_partitions = 0
    except Exception:
        current_partitions = 0

    logger.info(f"Topic {topic} 当前分区数: {current_partitions}, 消费者数: {total_consumers}")

    if current_partitions >= total_consumers:
        return

    logger.info(f"分区数不足，新增 {total_consumers - current_partitions} 个分区 (总数 -> {total_consumers})")
    fs = admin.create_partitions([NewPartitions(topic, total_consumers)])
    fs[topic].result(timeout=10)
    logger.info(f"✅ Topic {topic} 分区已扩展至 {total_consumers}")


# ============================================================
# 流水线阶段 1: 预处理 (CPU, stateless)
# ============================================================

@ray.remote(num_cpus=2)
def api_preprocess(task_data: dict, config_dict: dict) -> dict:
    """CPU 预处理：解析任务、检查视频文件、读取全部视频帧。"""
    try:
        from video_utils import  read_video_frames

        proc_logger = get_worker_logger("api_service_ray.preprocess")

        task_id = task_data.get("task_id", "")
        video_path = task_data.get("video_path", "")

        if not video_path or not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        frames_np = read_video_frames(video_path)
        total_frames = len(frames_np)
        proc_logger.info(f"[Preprocess] Task {task_id}: 读取视频 {video_path}, 共 {total_frames} 帧")
        frames_ref = ray.put(frames_np)

        # 注册到 ObjectKeeper 进行保活（后续 GPU Actor 直接通过 ObjectKeeper 获取）
        try:
            keeper = ray.get_actor("ObjectKeeper", namespace="production")
            # 记录注册前的详细信息
            proc_logger.info(f"[Preprocess] Task {task_id}: 正在注册到 ObjectKeeper, frames_np.shape={frames_np.shape}, dtype={frames_np.dtype}")
            # ⚠️ 关键修复：必须同步等待注册完成！
            # register_ref 的入参是 frames_ref（几 GB 数据），Ray 执行该 Actor 任务前
            # 需先把数据传到 keeper 所在节点（Head），耗时数秒。
            # 若 fire-and-forget，preprocess 一返回，Scheduler 立即下发 GPU 推理，
            # GPU Actor 马上 get_ref 时注册还没落地 → 返回 None → 回退磁盘重读。
            _t_reg = time.perf_counter()
            registered = ray.get(keeper.register_ref.remote(task_id, frames_ref, ttl=1800))
            proc_logger.info(f"[Preprocess] Task {task_id}: ✅ ObjectKeeper 注册完成 ok={registered}, "
                             f"耗时 {(time.perf_counter() - _t_reg) * 1000:.0f}ms (TTL=1800s)")
        except ValueError:
            proc_logger.warning(f"[Preprocess] Task {task_id}: ObjectKeeper 未启动，跳过保活注册")
        except Exception as e:
            proc_logger.error(f"[Preprocess] Task {task_id}: 注册 ObjectKeeper 失败: {e}")

        # 释放本地引用，不再将 frames_ref 通过返回值传递
        del frames_ref, frames_np
        import gc
        gc.collect()

        return {
            "task_id": task_id,
            "video_path": video_path,
            "total_frames": total_frames,
        }
    except Exception as e:
        logger.error(f"[Preprocess] 预处理失败 task {task_data.get('task_id', '')}: {e}")
        return {"task_id": task_data.get("task_id", ""), "error": str(e)}


# ============================================================
# 流水线阶段 2: GPU 推理 (stateful, 持有 PyTorch 模型)
# ============================================================

@ray.remote
class ApiGPUInferActor:
    def __init__(self, worker_idx: int, config_dict: dict):
        self.worker_idx = worker_idx
        self.config_dict = config_dict
        self.shutdown_event = False

        self._worker_logger = logging.getLogger(f"api_service_ray.gpu_{worker_idx}")
        self._worker_logger.setLevel(logging.INFO)
        self._worker_logger.propagate = False
        fh = logging.FileHandler('/data2/caimoyao/logs/api_service_ray.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._worker_logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._worker_logger.addHandler(sh)

        self._init_model()

    def _init_model(self):
        import torch
        import numpy as np
        from PIL import Image

        self.device = torch.device("cuda:0")

        sys.path.insert(0, self.config_dict["project_root"])
        from tools.inference.torch_inf import ECInferencer, build_model

        model, img_size, task = build_model(
            self.config_dict["config_path"],
            self.config_dict["checkpoint_path"],
            self.device,
        )
        self.inferencer = ECInferencer(
            model=model, task=task, device=self.device,
            size=img_size, thresh=self.config_dict.get("thresh", 0.5),
            square_size=1920,  # 拉伸到 1920×1920 正方形，与训练(2048×2048 拉伸)分布一致，消除框偏移
            # compile_mode="reduce-overhead",
        )

        updater = None
        rag_cache = self.config_dict.get("rag_cache")
        if rag_cache and os.path.exists(rag_cache):
            try:
                from streaming_cluster_updater import StreamingClusterUpdater
                updater = StreamingClusterUpdater(rag_db_path=rag_cache, device=self.device)
                self._worker_logger.info(f"[GPU-{self.worker_idx}] RAG 知识库已加载")
            except Exception as e:
                self._worker_logger.info(f"[GPU-{self.worker_idx}] RAG 知识库未加载: {e}")
        self.updater = updater

        num_streams = self.config_dict.get("num_streams", 4)
        self.num_streams = num_streams

        from PIL import Image as _PIL
        _warm_img = _PIL.new("RGB", (2048, 2048))

        # ONNX 路径：ORT CUDA EP 使用独立的显存管理，torch.cuda 无法直接测量
        # 因此用估算值：1080x1920 输入约 200MB/帧（ViT attention 占大头）
        if self.inferencer.use_onnx:
            self._worker_logger.info("[GPU-%d] ONNX 路径：使用估算显存", self.worker_idx)
            self.inferencer.infer(_warm_img)
            # 单帧约 200MB，加安全系数
            self.per_frame_mem_mb = 200.0
        else:
            torch.cuda.reset_peak_memory_stats()
            # warmup 首次调用会触发 torch.compile 编译（耗时数十秒，属正常现象）
            self._worker_logger.info(f"[GPU-{self.worker_idx}] warmup 开始（torch.compile 首次调用将编译模型，较慢）...")
            _ = self.inferencer.infer(_warm_img)
            torch.cuda.synchronize()
            _base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            # GPU tensor 路径 warmup
            _warm_tensor = torch.randint(0, 256, (3, 2048, 2048), dtype=torch.uint8, device=self.device)
            _ = self.inferencer.infer_tensor(_warm_tensor, (2048, 2048))
            torch.cuda.synchronize()
            self.per_frame_mem_mb = (torch.cuda.max_memory_allocated() - _base) / 1024 ** 2
            del _warm_tensor

        del _warm_img
        self._worker_logger.info(f"[GPU-{self.worker_idx}] 模型加载完成, per_frame_mem={self.per_frame_mem_mb:.1f}MB")

    def _prefetch_batches(self, tensor_nhwc, total_frames, infer_batch_size,
                          orig_size, prefetch_queue, h2d_stream):
        """后台预取线程：逐批完成 H2D 传输，与主线程的模型推理重叠。

        使用独立 CUDA stream（h2d_stream）做 permute+contiguous+cuda，
        主线程在推理上一批时，本线程已经在传输下一批。

        队列协议：
        - (batch_start, batch_gpu)  : 一个已传输完成的 batch
        - ("ERROR", exception)       : 传输过程出错
        - None                       : 全部传输完成
        """
        import torch
        try:
            for batch_start in range(0, total_frames, infer_batch_size):
                if self.shutdown_event:
                    break
                batch_end = min(batch_start + infer_batch_size, total_frames)
                batch_nhwc = tensor_nhwc[batch_start:batch_end]

                # H2D + 维度转换 (N, H, W, 3) -> (N, 3, H, W)，独立 stream 异步传输
                with torch.cuda.stream(h2d_stream):
                    batch_gpu = (
                        batch_nhwc.permute(0, 3, 1, 2)
                        .contiguous()
                        .cuda(self.device, non_blocking=True)
                    )
                # 放入就绪队列（主线程推理完上一批后立即取走）
                prefetch_queue.put((batch_start, batch_gpu))
        except Exception as e:
            prefetch_queue.put(("ERROR", e))
        finally:
            prefetch_queue.put(None)

    def infer(self, preprocessed_data: dict) -> dict:
        """GPU 推理：双缓冲流水线，预取线程提前 H2D 传输下一批，与推理重叠。"""
        import torch
        import supervision as sv
        import numpy as np
        from trackers import CBIoUTracker

        task_id = preprocessed_data["task_id"]
        video_path = preprocessed_data["video_path"]
        total_frames = preprocessed_data["total_frames"]
        self._worker_logger.info(f"[GPU-{self.worker_idx}] 任务 {task_id}: 开始推理, {total_frames}帧")

        # 从 ObjectKeeper 获取帧数据（避免通过 Task 返回值传递 frames_ref）
        try:
            keeper = ray.get_actor("ObjectKeeper", namespace="production")
            frames_ref = keeper.get_ref.remote(task_id)
            frames_np = ray.get(frames_ref)
            del frames_ref
            
            if frames_np is None:
                self._worker_logger.warning(f"[GPU-{self.worker_idx}] 任务 {task_id}: ObjectKeeper 中数据已过期或未注册，尝试从磁盘读取")
                from video_utils import read_video_frames
                if not video_path or not os.path.exists(video_path):
                    raise ValueError(f"视频文件不存在且 ObjectKeeper 无缓存: {video_path}")
                frames_np = read_video_frames(video_path)
        except Exception:
            self._worker_logger.error(f"[GPU-{self.worker_idx}] 任务 {task_id}: 从 ObjectKeeper 获取帧数据失败")
            return {"task_id": task_id, "video_path": video_path, "error": "ObjectKeeper 获取失败"}

        _t0 = time.perf_counter()

        try:
            _free, _total = torch.cuda.mem_get_info(self.device)
            # 双缓冲流水线：GPU 上最多同时有 1 个推理中 batch + 2 个预取 batch
            # 因此有效显存预算除以 3（流水线深度）
            pipeline_depth = 3
            safety_ratio = 0.60 / pipeline_depth
            # 动态计算推理批大小：可用显存 / 每帧显存估算
            _configured_batch = self.config_dict.get("infer_batch_size", 8)
            _max_batch = max(
                1, int(_free * safety_ratio / (self.per_frame_mem_mb * 1024 ** 2))
            )
            infer_batch_size = min(_configured_batch, _max_batch, total_frames)

            self._worker_logger.info(
                f"[GPU-{self.worker_idx}] 推理批大小: {infer_batch_size} "
                f"(每帧约 {self.per_frame_mem_mb:.1f}MB, 流水线深度 {pipeline_depth})"
            )

            frame_data = [None] * total_frames
            orig_size = (frames_np.shape[2], frames_np.shape[1])  # (W, H)

            # 直接将 NumPy 数组转为 PyTorch Tensor View（零拷贝）
            if not frames_np.flags.writeable:
                frames_np = frames_np.copy()
            tensor_nhwc = torch.from_numpy(frames_np)

            # ══════════════════════════════════════════════════════════
            # 双缓冲流水线：预取线程负责 H2D，主线程负责推理
            #   ┌──────────┐   H2D(下一批)   ┌──────────┐
            #   │ 预取线程  │ ───────────────▶ │  主线程   │
            #   │ (传输)    │ ◀─────────────── │ (推理)    │
            #   └──────────┘   队列(maxsize=2) └──────────┘
            # 推理 batch N 期间，预取线程已在传输 batch N+1
            # ══════════════════════════════════════════════════════════
            h2d_stream = torch.cuda.Stream(device=self.device)
            prefetch_queue = queue.Queue(maxsize=2)  # 双缓冲

            prefetch_thread = threading.Thread(
                target=self._prefetch_batches,
                args=(tensor_nhwc, total_frames, infer_batch_size,
                      orig_size, prefetch_queue, h2d_stream),
                daemon=True,
                name=f"prefetch-gpu{self.worker_idx}",
            )
            prefetch_thread.start()

            # 主循环：从队列取已传输好的 batch 推理
            while True:
                if self.shutdown_event:
                    self._worker_logger.info(
                        f"[GPU-{self.worker_idx}] task {task_id}: 收到停止信号，退出推理"
                    )
                    break

                item = prefetch_queue.get()
                if item is None:
                    break  # 预取线程全部完成
                if isinstance(item, tuple) and item[0] == "ERROR":
                    raise item[1]

                batch_start, batch_gpu = item
                batch_frames = min(infer_batch_size, total_frames - batch_start)

                # 等待本批 H2D 完成（通常早已就绪，几乎不阻塞）
                h2d_stream.synchronize()

                # 批量推理：GPU 变换 + model forward 一次处理 N 帧
                batch_orig_sizes = [orig_size] * batch_frames
                batch_results = self.inferencer.infer_batch(batch_gpu, batch_orig_sizes)

                # 分发结果到 frame_data 对应位置
                for j in range(batch_frames):
                    frame_idx = batch_start + j
                    frame_data[frame_idx] = {
                        "frame_idx": frame_idx,
                        "results": batch_results[j],
                        "frame_name": f"{frame_idx:06d}.jpg",
                    }

                # 清理本批次 GPU buffer（预取线程已开始下一批传输）
                del batch_gpu

            # 等待预取线程退出，确保资源清理
            prefetch_thread.join(timeout=5)
            h2d_stream.synchronize()
            del h2d_stream, prefetch_queue

            torch.cuda.synchronize()

            del tensor_nhwc

            infer_ms = (time.perf_counter() - _t0) * 1000
            self._worker_logger.info(f"[GPU-{self.worker_idx}] 任务 {task_id}: {total_frames} frames "
                                     f"推理完成, 耗时: {infer_ms:.0f}ms, "
                                     f"平均: {infer_ms / max(total_frames, 1):.1f}ms/帧")

            try:
                _t_track = time.perf_counter()
                tracker = CBIoUTracker()
                for fd in frame_data:
                    if fd is None:
                        # 帧无检测结果（None 占位）→ 用空检测更新 tracker，保持时间连续性
                        empty_det = sv.Detections(
                            xyxy=np.empty((0, 4)),
                            confidence=np.empty((0,)),
                            class_id=np.empty((0,), dtype=int),
                        )
                        tracker.update(empty_det)
                        continue
                    results = fd["results"]
                    track_results = [r for r in results if r.label == 1]
                    if not track_results:
                        empty_det = sv.Detections(
                            xyxy=np.empty((0, 4)),
                            confidence=np.empty((0,)),
                            class_id=np.empty((0,), dtype=int),
                        )
                        tracker.update(empty_det)
                        continue
                    # 统一 float32：ByteTrack 内部将框转 float32 处理，
                    # 若用 float64 坐标做精确 key 匹配会全部失败（track_id=-1）
                    xyxy = np.array([r.box for r in track_results]).astype(np.float32)
                    conf = np.array([r.score for r in track_results])
                    cls = np.array([r.label for r in track_results])
                    detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
                    tracked = tracker.update(detections)

                    box_to_tid = {}
                    for tb, tid in zip(tracked.xyxy, tracked.tracker_id):
                        box_to_tid[tuple(np.asarray(tb, dtype=np.float32))] = int(tid)
                    for r in track_results:
                        r.track_id = box_to_tid.get(
                            tuple(np.asarray(r.box, dtype=np.float32)), -1)
                self._worker_logger.info(f"[GPU-{self.worker_idx}] 跟踪总耗时: {(time.perf_counter()-_t_track)*1000:.1f}ms / {len(frame_data)}帧")
            except Exception as e:
                self._worker_logger.warning(f"[GPU-{self.worker_idx}] 跟踪失败（跳过）: {e}")

            from streaming_cluster_updater import process_video_incremental
            output_parent = Path(self.config_dict["output_dir"]) / "temp" / task_id
            summary = process_video_incremental(
                frame_data=frame_data,
                output_root=output_parent,
                draw_func=None,
                updater=self.updater,
                gpu_id=0,
                save_output=False,
                frames_np=frames_np,  # 传递 numpy 引用，下游按需转换为 PIL
            )

            # 🔴 关键：只提取需要的字段，释放 summary 可能持有的 frames_np/frame_data 间接引用
            summary_lightweight = {
                "output_dir": summary.get("output_dir", str(output_parent)),
            }
            del summary, frame_data

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            return {
                "task_id": task_id,
                "video_path": video_path,
                "summary": summary_lightweight,
                "total_frames": total_frames,
            }

        except torch.cuda.OutOfMemoryError as e:
            self._worker_logger.error(f"[GPU-{self.worker_idx}] OOM: {e}")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return {"task_id": task_id, "error": f"OOM: {e}"}
        except Exception as e:
            import traceback
            self._worker_logger.error(
                f"[GPU-{self.worker_idx}] 任务 {task_id} 失败: {e}\n"
                f"{traceback.format_exc()}"
            )
            return {"task_id": task_id, "error": str(e)}
        finally:
            # 彻底清理局部大变量引用，防止循环引用导致 GC 延迟
            try:
                del tensor_nhwc
            except (NameError, UnboundLocalError):
                pass
            try:
                del frame_data
            except (NameError, UnboundLocalError):
                pass
            try:
                del frames_np
            except (NameError, UnboundLocalError):
                pass
            try:
                del tracker
            except (NameError, UnboundLocalError):
                pass

            # 释放 ObjectKeeper 中的帧引用（infer 是最后一个使用 frames_np 的阶段，
            # 无论成功失败都必须释放，否则引用会一直持有到 TTL 过期导致内存泄漏）
            # try:
            #     keeper = ray.get_actor("ObjectKeeper", namespace="production")
            #     _released = releaseray.get(keeper.release_ref.remote(task_id))
            #     if _released:
            #         self._worker_logger.info(
            #             f"[GPU-{self.worker_idx}] 任务 {task_id}: ✅ 已释放 ObjectKeeper 引用")
            # except Exception as _e:
            #     self._worker_logger.warning(
            #         f"[GPU-{self.worker_idx}] 任务 {task_id}: 释放 ObjectKeeper 引用失败: {_e}")

            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def stop(self):
        self.shutdown_event = True

    def is_stopped(self):
        return self.shutdown_event


# ============================================================
# 流水线阶段 3: 后处理 (CPU, stateless)
# ============================================================

@ray.remote(num_cpus=1)
def api_postprocess(infer_results: dict, task_data: dict) -> dict:
    """CPU 后处理：构建最终 result_data，并释放 ObjectKeeper 中的帧数据引用。"""
    post_logger = get_worker_logger("api_service_ray.postprocess")
    task_id = infer_results.get("task_id", "")
    try:
        post_logger.info(f"[Postprocess] Task {task_id}: 开始后处理")
        if "error" in infer_results:
            post_logger.error(f"[Postprocess] task {task_id}: 推理阶段失败: {infer_results['error']}")
            return {"task_id": task_id, "status": "failed", "error": infer_results["error"]}

        video_path = infer_results.get("video_path", "")
        summary = infer_results.get("summary", {})

        result_data = {
            "task_id": task_id,
            "status": "completed",
            "summary": {
                "output_dir": summary.get("output_dir", ""),
                "video_path": video_path,
                "total_frames": infer_results.get("total_frames", 0),
            },
        }
        post_logger.info(f"[Postprocess] task {task_id}: 结果构建完成")
        return result_data
    except Exception as e:
        post_logger.error(f"[Postprocess] 构建结果失败: {e}")
        return {"task_id": task_data.get("task_id", ""), "status": "failed", "error": str(e)}
    # finally:
    #     # 防御性释放：infer 的 finally 已释放过，这里重复调用无害（返回 False 不报错）。
    #     # 用 finally 确保任何返回路径（含 error 提前 return）都执行释放。
    #     try:
    #         keeper = ray.get_actor("ObjectKeeper", namespace="production")
    #         _released = ray.get(keeper.release_ref.remote(task_id))
    #         if _released:
    #             post_logger.info(f"[Postprocess] Task {task_id}: ✅ 已释放 ObjectKeeper 引用")
    #     except Exception as _e:
    #         post_logger.warning(f"[Postprocess] Task {task_id}: 释放 ObjectKeeper 引用失败: {_e}")


# ============================================================
# 调度器 Actor (lightweight, 无 GPU)
# ============================================================

@ray.remote
class ApiSchedulerActor:
    def __init__(self, worker_idx: int, gpu_actor, config_dict: dict):
        self.worker_idx = worker_idx
        self.gpu_actor = gpu_actor
        self.config_dict = config_dict
        self.shutdown_event = False

        self._worker_logger = logging.getLogger(f"api_service_ray.scheduler_{worker_idx}")
        self._worker_logger.setLevel(logging.INFO)
        self._worker_logger.propagate = False
        fh = logging.FileHandler('/data2/caimoyao/logs/api_service_ray.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._worker_logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._worker_logger.addHandler(sh)

        self._init_kafka()
        self._offset_recovered = False  # 启动时是否已完成分区消费起点校准

    def _init_kafka(self):
        from confluent_kafka import Consumer, Producer

        def on_assign(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] ✅ 分配到分区: {p_ids}")

        def on_revoke(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] ❌ 被回收分区: {p_ids}")

        def on_lost(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] ⚠️ 丢失分区: {p_ids}")

        self.kafka_consumer = Consumer({
            "bootstrap.servers": self.config_dict["kafka_bootstrap_servers"],
            "group.id": self.config_dict["kafka_consumer_group"],
            "group.instance.id": f"ray-api-scheduler-{self.worker_idx}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": self.config_dict.get("kafka_max_poll_interval_ms", 1800000),
            "session.timeout.ms": self.config_dict.get("kafka_session_timeout_ms", 60000),
            "heartbeat.interval.ms": self.config_dict.get("kafka_heartbeat_interval_ms", 3000),
            "partition.assignment.strategy": "cooperative-sticky",
            "broker.address.family": "v4",
            "security.protocol": "PLAINTEXT",
            "fetch.min.bytes": 1,
            "queued.min.messages": 1,
        })
        self.kafka_consumer.subscribe(
            [self.config_dict["kafka_task_topic"]],
            on_assign=on_assign,
            on_revoke=on_revoke,
            on_lost=on_lost,
        )
        self._worker_logger.info(f"[Scheduler-{self.worker_idx}] Kafka Consumer 已连接")

        self.kafka_producer = Producer({
            "bootstrap.servers": self.config_dict["kafka_bootstrap_servers"],
            "broker.address.family": "v4",
            "security.protocol": "PLAINTEXT",
            "compression.type": "lz4",
            "message.max.bytes": 104857600,
            "queue.buffering.max.messages": 100000,
        })
        self._worker_logger.info(f"[Scheduler-{self.worker_idx}] Kafka Producer 已连接")

    def _recover_offsets(self):
        """启动时显式校准各分区消费起点（与 check_partitions.py 逻辑一致）。

        - 读取各分区已提交 offset（committed）与 watermark（low/high）
        - committed 有效（>= low）→ 从 committed 继续消费（Kafka 默认行为，无需 seek）
        - committed 无效/越界（-1001 或 < low）→ seek 到 low（日志最早可用位置），
          即继续消费 broker 上仍存在的"未消费"部分，避免从头重读已被清理的消息
        """
        from confluent_kafka import TopicPartition

        topic = self.config_dict["kafka_task_topic"]
        try:
            assignment = self.kafka_consumer.assignment()
        except Exception as e:
            self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] 分区校准: 获取 assignment 失败: {e}")
            return
        if not assignment:
            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] 分区校准: 暂无分配分区，等待下次重试")
            return

        try:
            committed = self.kafka_consumer.committed(assignment, timeout=10)
        except Exception as e:
            self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] 分区校准: 查询 committed offset 失败: {e}")
            return

        for tp, cp in zip(assignment, committed):
            try:
                low, high = self.kafka_consumer.get_watermark_offsets(tp, timeout=10)
                committed_off = cp.offset if cp else -1001
                if committed_off < 0 or committed_off < low:
                    # 已提交 offset 已不存在（被 retention 清理）或从未消费过：
                    # 从日志最早可用位置继续，即只消费"未消费"部分
                    self.kafka_consumer.seek(TopicPartition(tp.topic, tp.partition, low))
                    self._worker_logger.info(
                        f"[Scheduler-{self.worker_idx}] 📍 {topic}[{tp.partition}] 起点校准: "
                        f"committed={committed_off}(无效/越界) low={low} high={high} → 从 low={low} 继续"
                    )
                else:
                    self._worker_logger.info(
                        f"[Scheduler-{self.worker_idx}] 📍 {topic}[{tp.partition}] 起点校准: "
                        f"committed={committed_off} low={low} high={high} → 从 committed 继续 (未消费 {max(0, high - committed_off)})"
                    )
            except Exception as e:
                self._worker_logger.warning(
                    f"[Scheduler-{self.worker_idx}] 📍 {topic}[{tp.partition}] 起点校准失败: {e}"
                )

    def run(self):
        import json
        from pathlib import Path

        # pipeline state: task_id -> {stage, futures, task_data, msg, start_time}
        pipeline = {}
        disk_full_backoff_until = 0.0
        # task_timeout = 3600  # 30 分钟超时

        try:
            while not self.shutdown_event:
                # 启动后首次分配完成时，校准各分区消费起点（只做一次，不重复 seek）
                if not self._offset_recovered:
                    if self.kafka_consumer.assignment():
                        self._recover_offsets()
                        self._offset_recovered = True

                # 1. 检查各流水线阶段完成情况
                completed = []
                now = time.time()
                for task_id, state in list(pipeline.items()):
                    # 超时检查
                    # if now - state.get('start_time', 0) > task_timeout:
                    #     self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] task {task_id}: 超时，强制清理")
                    #     # self.kafka_consumer.commit(state['msg'], asynchronous=False)
                    #     completed.append(task_id)
                    #     continue

                    stage = state['stage']

                    if stage == 'preprocess':
                        ready, _ = ray.wait([state['preprocess_future']], timeout=0)
                        if ready:
                            try:
                                preprocessed = ray.get(ready[0])
                            except Exception as _e:
                                self._worker_logger.error(f"[Scheduler-{self.worker_idx}] task {task_id}: 预处理异常: {_e}")
                                _release_keeper_ref(task_id)  # 报错也释放引用
                                completed.append(task_id)
                                continue
                            if 'error' in preprocessed:
                                self._worker_logger.error(f"[Scheduler-{self.worker_idx}] task {task_id}: 预处理失败: {preprocessed['error']}")
                                _release_keeper_ref(task_id)  # 失败也释放引用
                                is_disk_full = "disk is full" in str(preprocessed.get('error', ''))
                                if is_disk_full:
                                    disk_full_backoff_until = time.time() + 30  # 退避 30s
                                    self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] 检测到磁盘已满, 暂停轮询 {30}s")
                                completed.append(task_id)
                            else:
                                infer_future = self.gpu_actor.infer.remote(preprocessed)
                                del preprocessed  # 释放元数据 dict（frames_np 已由 ObjectKeeper 管理）
                                state['infer_future'] = infer_future
                                state['stage'] = 'infer'
                                self._worker_logger.info(f"[Scheduler-{self.worker_idx}] task {task_id}: 进入推理阶段")

                    elif stage == 'infer':
                        ready, _ = ray.wait([state['infer_future']], timeout=0)
                        if ready:
                            infer_results = ray.get(ready[0])
                            post_future = api_postprocess.remote(infer_results, state['task_data'])
                            del infer_results  # 释放可能含 ObjectRef 的 dict
                            state['postprocess_future'] = post_future
                            state['stage'] = 'postprocess'
                            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] task {task_id}: 进入后处理阶段")

                    elif stage == 'postprocess':
                        ready, _ = ray.wait([state['postprocess_future']], timeout=0)
                        if ready:
                            result_data = ray.get(ready[0])
                            self._send_result(result_data, task_id)
                            # 提交 offset：扩容/缩容触发 cooperative-sticky rebalance 时，
                            # 分区所有权可能正被移交，commit 会抛 CommitFailedError。
                            # 此时结果已发送到 result topic，消息最多被重复消费一次
                            # (at-least-once)，不应因此 crash 掉 scheduler。
                            try:
                                self.kafka_consumer.commit(state['msg'], asynchronous=False)
                            except Exception as _ce:
                                self._worker_logger.warning(
                                    f"[Scheduler-{self.worker_idx}] task {task_id}: 提交 offset 失败(分区可能已移交): {_ce}")
                            completed.append(task_id)

                # 清理已完成的任务
                for tid in completed:
                    del pipeline[tid]

                # 2. 从 Kafka 轮询新任务（流水线未满 + 无正在预处理的任务 + 非退避状态）
                if len(pipeline) < config.PIPELINE_DEPTH and time.time() >= disk_full_backoff_until:
                    has_preprocess = any(state['stage'] == 'preprocess' for state in pipeline.values())
                    if not has_preprocess:
                        try:
                            msg = self.kafka_consumer.poll(timeout=0.1)
                        except Exception as e:
                            self._worker_logger.error(f"[Scheduler-{self.worker_idx}] Kafka poll 异常: {e}")
                            time.sleep(0.5)
                            continue

                        if msg is None:
                            time.sleep(0.05)
                            continue

                        if msg.error():
                            self._worker_logger.error(f"[Scheduler-{self.worker_idx}] Kafka poll error: {msg.error()}")
                            continue

                        task_data = json.loads(msg.value().decode('utf-8'))
                        task_id = task_data.get("task_id", "")
                        self._worker_logger.info(f"[Scheduler-{self.worker_idx}] 收到任务: {task_id}")

                        pre_future = api_preprocess.remote(task_data, self.config_dict)
                        pipeline[task_id] = {
                            'stage': 'preprocess',
                            'preprocess_future': pre_future,
                            'infer_future': None,
                            'postprocess_future': None,
                            'task_data': task_data,
                            'msg': msg,
                            'start_time': time.time(),
                        }
                        self._worker_logger.info(f"[Scheduler-{self.worker_idx}] task {task_id}: 进入预处理阶段")

                time.sleep(0.01)
        finally:
            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] 正在关闭 Kafka 连接...")
            # 主动停止时释放所有仍在流水线中的任务引用，避免 object store 占用
            for _tid in list(pipeline.keys()):
                _release_keeper_ref(_tid)
                self._worker_logger.info(f"[Scheduler-{self.worker_idx}] 主动停止, 释放引用: {_tid}")
            try:
                self.kafka_consumer.close()
            except Exception as e:
                self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] Consumer 关闭异常: {e}")
            try:
                self.kafka_producer.close()
            except Exception as e:
                self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] Producer 关闭异常: {e}")
            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] scheduler 退出")

    def _send_result(self, result_data: dict, task_id: str):
        import json
        result_bytes = json.dumps(result_data, ensure_ascii=False).encode('utf-8')
        send_success = False
        for attempt in range(3):
            try:
                self.kafka_producer.produce(
                    self.config_dict["kafka_result_topic"],
                    key=task_id.encode('utf-8'),
                    value=result_bytes,
                )
                self.kafka_producer.flush(timeout=30)
                send_success = True
                break
            except Exception as e:
                self._worker_logger.error(f"[Scheduler-{self.worker_idx}] task {task_id}: Kafka 发送失败: {e}")
                time.sleep(2 ** attempt)

        if send_success:
            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] 任务 {task_id} 完成")
        else:
            self._worker_logger.error(f"[Scheduler-{self.worker_idx}] task {task_id}: Kafka 发送失败")

    def stop(self):
        self.shutdown_event = True

    def is_stopped(self):
        return self.shutdown_event


# ============================================================
# 辅助组件
# ============================================================

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
            "security.protocol": "PLAINTEXT",
            "socket.timeout.ms": 10000,
        }

        results = {}

        # 1. 使用临时 Consumer 查 committed 和 watermark，设置 group.id 但不 subscribe，绝不触发 Rebalance
        for topic, group in zip(self.topics, self.consumer_groups):
            consumer_conf = kafka_conf.copy()
            # 设置 group.id 仅用于只读查询 committed offset
            consumer_conf["group.id"] = group
            consumer_conf["enable.auto.commit"] = False

            c = None
            try:
                c = Consumer(consumer_conf)
                
                # 1. 获取 Topic 元数据与分区数
                md = c.list_topics(topic, timeout=10)
                if topic not in md.topics or md.topics[topic].error:
                    results[topic] = {"lag": -1, "error": f"topic {topic} 不存在或不可用"}
                    continue

                partitions_meta = md.topics[topic].partitions
                tps = [TopicPartition(topic, p) for p in partitions_meta.keys()]

                # 2. 查询真实消费者组对这些分区的 committed offset
                committed_tps = c.committed(tps, timeout=10)
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
                    c.close()  # 确保 100% 释放 C 句柄与 socket，绝不泄漏

        return results

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
    total_workers = 0
    for gpu_id in config.GPU_IDS:
        free_mb = gpu_free.get(gpu_id, 0)
        if free_mb <= 11 * 1024:
            continue
        n_workers = min(config.MAX_TOTAL_WORKERS, free_mb // 10240)
        total_workers += n_workers
    return min(total_workers, max_total)


def _release_keeper_ref(task_id: str):
    """释放 ObjectKeeper 中 task_id 对应的引用，避免 object store 内存占用。

    在任务报错、超时、主动停止等非正常完成路径下调用，
    确保 jpeg/numpy 帧数据不会一直挂在 object store 中。
    """
    try:
        keeper = ray.get_actor("ObjectKeeper", namespace="production")
        ray.get(keeper.release_ref.remote(task_id))
    except Exception:
        pass


def _get_ray_object_store_used_gb():
    """获取 Ray object store 真实已使用内存 (GB)。

    ⚠️ 不能用 cluster_resources() - available_resources() 计算！
    object_store_memory 是静态资源量，available 不随真实占用变化，恒等于总量，
    算出来永远为 0（历史上出现 '已用 0.0GB' 的原因）。
    改用 `ray memory --stats-only` 的 Plasma usage 聚合值（与 ray memory 输出一致）。
    """
    try:
        gcs_address = ray.get_runtime_context().gcs_address
        ray_bin = os.path.join(os.path.dirname(sys.executable), "ray")
        env = dict(os.environ)
        env["RAY_ADDRESS"] = gcs_address
        _result = subprocess.run(
            [ray_bin, "memory", "--stats-only"],
            capture_output=True, text=True, timeout=20, env=env,
        )
        if _result.returncode != 0:
            logger.error(f"ray memory --stats-only 失败: {_result.stderr[-500:]}")
            return 0.0
        m = re.search(r"Plasma memory usage\s+(\d+)\s+MiB", _result.stdout)
        if not m:
            logger.error(f"解析 ray memory 输出失败: {_result.stdout[-500:]}")
            return 0.0
        return int(m.group(1)) / 1024.0
    except Exception as e:
        logger.error(f"获取 Ray object store 内存失败: {e}")
        return 0.0


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("   EdgeCrafter Kafka 推理服务 (Ray 流水线并行)")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.GPU_IDS))
    print(f"已设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # # 强行停止所有残留 Ray 进程
    # subprocess.run(["ray", "stop", "--force"], capture_output=True)
    # logger.info("🧹 已停止所有残留 Ray 进程")

    # # 清理上次 Ray 进程的残留数据
    # import glob as _glob
    # for _old_session in _glob.glob("/data2/caimoyao/ray/session_*"):
    #     import shutil
    #     shutil.rmtree(_old_session, ignore_errors=True)
    #     logger.info(f"🧹 已清理残留 Ray session: {_old_session}")
    # spill_dir = "/data2/caimoyao/ray_spill"
    # if os.path.exists(spill_dir):
    #     import shutil
    #     shutil.rmtree(spill_dir, ignore_errors=True)
    #     logger.info(f"🧹 已清理 Ray spill 目录: {spill_dir}")

    ray.init(
        address="auto",
        runtime_env={"env_vars": {
            "PYTHONPATH": os.getcwd(),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.GPU_IDS)),
            "LD_LIBRARY_PATH": f"{TRT_PATH}/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib",
        }},
    )
    logger.info(f"✅ 已连接到 Ray 集群, GPU可用: {ray.available_resources().get('GPU', 0)}")

    config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    cfg_dict = {
        "project_root": str(config.PROJECT_ROOT),
        "config_path": str(config.CONFIG_PATH),
        "checkpoint_path": str(config.CHECKPOINT_PATH),
        "thresh": config.CONF_THRESHOLD,
        "num_streams": config.NUM_STREAMS,
        "output_dir": str(config.OUTPUT_DIR),
        "rag_cache": str(config.RAG_CACHE) if config.RAG_CACHE else None,
        "rag_threshold": config.RAG_THRESHOLD,
        "kafka_bootstrap_servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "kafka_task_topic": config.KAFKA_TASK_TOPIC,
        "kafka_result_topic": config.KAFKA_RESULT_TOPIC,
        "kafka_consumer_group": config.KAFKA_CONSUMER_GROUP,
        "kafka_max_poll_interval_ms": config.KAFKA_MAX_POLL_INTERVAL_MS,
        "kafka_session_timeout_ms": config.KAFKA_SESSION_TIMEOUT_MS,
        "kafka_heartbeat_interval_ms": config.KAFKA_HEARTBEAT_INTERVAL_MS,
        "pipeline_depth": config.PIPELINE_DEPTH,
    }

    init_topics()

    ensure_kafka_partitions(config.MIN_TOTAL_WORKERS)

    # 每个 Worker = 1 个 GPU Actor + 1 个 Scheduler Actor
    gpu_actors = []
    scheduler_actors = []

    gpu_fraction = 1.0 / config.WORKERS_PER_GPU
    gpu_resource = {config.RESOURCE_GPU: 1.0}  # 锁定到伪节点，每个 Worker 占用 1 单位资源

    for w_idx in range(config.MIN_TOTAL_WORKERS):
        gpu_actor = ApiGPUInferActor.options(
            num_gpus=gpu_fraction,
            resources=gpu_resource
        ).remote(w_idx, cfg_dict)
        gpu_actors.append(gpu_actor)
        scheduler = ApiSchedulerActor.remote(w_idx, gpu_actor, cfg_dict)
        scheduler_actors.append(scheduler)
    logger.info(
        f"🚀 已创建 {config.MIN_TOTAL_WORKERS} 个 Worker "
        f"(GPU+Scheduler 配对, 每个 Worker 占 {gpu_fraction:.2f} GPU, "
        f"resource={config.RESOURCE_GPU})"
    )

    for scheduler in scheduler_actors:
        scheduler.run.remote()

    lag_monitor = KafkaLagMonitor.remote(
        config.KAFKA_BOOTSTRAP_SERVERS,
        [config.KAFKA_TASK_TOPIC],
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
                topic_lag = lag_info.get(config.KAFKA_TASK_TOPIC, {})
                lag = topic_lag.get("lag", 0)

                if lag < 0:
                    logger.warning(f"⚠️ Kafka 积压监控获取失败: {topic_lag.get('error', '未知错误')}")
                    continue

                partitions = topic_lag.get("partitions", 0)
                max_partition_lag = topic_lag.get("max_partition_lag", 0)
                partition_lags = topic_lag.get("partition_lags", {})
                logger.info(f"📊 Kafka 积压检测: {config.KAFKA_TASK_TOPIC} total_lag={lag}, partitions={partitions}, max_partition_lag={max_partition_lag}, 当前 workers={len(scheduler_actors)}")
                if partition_lags:
                    logger.info(f"📊 各分区积压: {partition_lags}")

                num_workers = len(scheduler_actors)
                lag_per_worker = lag / num_workers if num_workers > 0 else 0
                need_scale_up = False

                if num_workers == 0 and lag > 0:
                    need_scale_up = True
                    logger.info(f"📈 无消费者但 topic 有积压 (lag={lag}), 需要立即添加消费者")
                elif lag > 0 and lag_per_worker > config.SCALE_UP_THRESHOLD and num_workers < config.MAX_TOTAL_WORKERS:
                    need_scale_up = True
                    logger.info(f"📈 单消费者积压超过阈值 (lag={lag}, workers={num_workers}, lag/worker={lag_per_worker:.1f} > {config.SCALE_UP_THRESHOLD})")

                # === 背压检查：object store 内存超限则禁止扩容 ===
                obj_store_used_gb = _get_ray_object_store_used_gb()
                backpressure = obj_store_used_gb > config.BACKPRESSURE_MEMORY_GB
                if backpressure and need_scale_up:
                    logger.warning(f"🔴 背压触发: object_store 已用 {obj_store_used_gb:.1f}GB > {config.BACKPRESSURE_MEMORY_GB}GB, 取消扩容")
                    need_scale_up = False
                # elif obj_store_used_gb > 0:
                logger.info(f"📊 Ray object store 内存: 已用 {obj_store_used_gb:.1f}GB / 阈值 {config.BACKPRESSURE_MEMORY_GB}GB")

                if need_scale_up:
                    gpu_free = _get_gpu_free_memory()
                    max_possible = _calculate_max_workers(gpu_free, config.MAX_TOTAL_WORKERS)

                    if max_possible > 0:
                        new_workers = 1
                        logger.info(f"📈 扩展 1 个 worker")

                        ensure_kafka_partitions(num_workers + new_workers)

                        w_idx = num_workers
                        gpu_actor = ApiGPUInferActor.options(
                            num_gpus=gpu_fraction,
                            resources=gpu_resource
                        ).remote(w_idx, cfg_dict)
                        gpu_actors.append(gpu_actor)
                        scheduler = ApiSchedulerActor.remote(w_idx, gpu_actor, cfg_dict)
                        scheduler_actors.append(scheduler)
                        logger.info(f"🚀 新增 Worker-{w_idx} (GPU+Scheduler, resource={config.RESOURCE_GPU})")
                        scheduler.run.remote()

                        last_scale_time = current_time

                elif backpressure and len(scheduler_actors) > 0:
                    # 背压强制缩减：object store 内存超限，缩减 1 个 worker 释放内存
                    remove_count = 1
                    logger.warning(f"🔴 背压触发: object_store 已用 {obj_store_used_gb:.1f}GB > {config.BACKPRESSURE_MEMORY_GB}GB, 强制收缩 1 个 worker")

                    for _ in range(remove_count):
                        if len(scheduler_actors) == 0:
                            break
                        scheduler_to_remove = scheduler_actors.pop()
                        gpu_to_remove = gpu_actors.pop()

                        # 通知 stop 并等待返回
                        try:
                            ray.get(scheduler_to_remove.stop.remote(), timeout=5)
                            ray.get(gpu_to_remove.stop.remote(), timeout=5)
                        except Exception:
                            pass

                        # 核心：强行 Kill 掉 Actor 进程以释放 GPU 显存与私有堆内存
                        ray.kill(scheduler_to_remove)
                        ray.kill(gpu_to_remove)
                        time.sleep(2)
                        logger.info(f"⏹️ 背压: 已强行销毁 Worker Actor 对并释放显存/内存")

                    last_scale_time = current_time

                elif lag_per_worker < config.SCALE_DOWN_THRESHOLD and len(scheduler_actors) > 0:
                    remove_count = 1
                    logger.info(f"📉 平均每 worker 积压低于阈值 (lag_per_worker={lag_per_worker:.1f} < {config.SCALE_DOWN_THRESHOLD}), 收缩 1 个 worker")

                    for _ in range(remove_count):
                        if len(scheduler_actors) == 0:
                            break
                        scheduler_to_remove = scheduler_actors.pop()
                        gpu_to_remove = gpu_actors.pop()

                        # 通知 stop 并等待返回
                        try:
                            ray.get(scheduler_to_remove.stop.remote(), timeout=5)
                            ray.get(gpu_to_remove.stop.remote(), timeout=5)
                        except Exception:
                            pass

                        # 核心：强行 Kill 掉 Actor 进程以释放 GPU 显存与私有堆内存
                        ray.kill(scheduler_to_remove)
                        ray.kill(gpu_to_remove)
                        time.sleep(2)
                        logger.info(f"⏹️ 已强行销毁 Worker Actor 对并释放显存/内存")

                    last_scale_time = current_time

            except Exception as e:
                logger.error(f"动态扩缩容失败: {e}")

    except KeyboardInterrupt:
        logger.info("🛑 收到停止信号，正在关闭服务...")
    finally:
        logger.info(f"📋 停止 {len(scheduler_actors)} 个 Scheduler Actor...")
        try:
            ray.get([s.stop.remote() for s in scheduler_actors], timeout=10)
        except Exception:
            pass
        for s in scheduler_actors:
            ray.kill(s)

        logger.info(f"📋 停止 {len(gpu_actors)} 个 GPU Actor...")
        try:
            ray.get([g.stop.remote() for g in gpu_actors], timeout=10)
        except Exception:
            pass
        for g in gpu_actors:
            ray.kill(g)

        time.sleep(3)
        ray.shutdown()
        logger.info("✅ 服务已关闭")


if __name__ == "__main__":
    main()
