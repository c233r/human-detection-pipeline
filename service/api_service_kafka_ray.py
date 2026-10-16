"""

EdgeCrafter 推理服务 (Ray版本)
基于 Kafka 消息队列 + Ray Actor 流水线分布式推理（方案B：流水线并行）
"""

import os
import sys
import json
import time
import signal
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



class Config:
    PROJECT_ROOT = Path("/data2/caimoyao/EdgeCrafter-main/ecdetseg")
    CONFIG_PATH = PROJECT_ROOT / "configs/ecdet/ecdet_l.yml"
    CHECKPOINT_PATH = PROJECT_ROOT / "models/last.pth"

    CONF_THRESHOLD = 0.5
    CLASSES = ["person", "face", "ball"]

    OUTPUT_DIR = Path("/data2/caimoyao/api_outputs_test")

    GPU_IDS = [7]
    # WORKERS_PER_GPU = 4
    NUM_STREAMS = 4

    RAG_CACHE = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/rag_db.pkl"
    RAG_THRESHOLD = 0.75

    SAVE_OUTPUT = False

    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_TASK_TOPIC = "video_tasks_test"
    KAFKA_RESULT_TOPIC = "video_results_test"
    KAFKA_CONSUMER_GROUP = "video_task_consumers_test"

    KAFKA_MAX_POLL_INTERVAL_MS = 14400000
    KAFKA_SESSION_TIMEOUT_MS = 60000
    KAFKA_HEARTBEAT_INTERVAL_MS = 3000

    RAY_OBJECT_STORE_MEMORY = None  # 设为 None 时自动计算

    PIPELINE_DEPTH = 3

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

    LAG_CHECK_INTERVAL = 1800
    SCALE_UP_THRESHOLD = 30
    SCALE_DOWN_THRESHOLD = 3
    MAX_TOTAL_WORKERS = 5
    MIN_TOTAL_WORKERS = 2


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
        from video_utils import read_video_frames

        task_id = task_data.get("task_id", "")
        video_path = task_data.get("video_path", "")

        if not video_path or not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        frames_np = read_video_frames(video_path)
        total_frames = len(frames_np)
        logger.info(f"[Preprocess] Task {task_id}: 读取视频 {video_path}, 共 {total_frames} 帧")
        frames_ref = ray.put(frames_np)

        # 注册到 ObjectKeeper 进行保活
        try:
            keeper = ray.get_actor("ObjectKeeper", namespace="production")
            keeper.register_ref.remote(task_id, frames_ref, ttl=1800)
            logger.info(f"[Preprocess] Task {task_id}: frames_ref 已注册到 ObjectKeeper (TTL=1800s)")
        except ValueError:
            logger.warning(f"[Preprocess] Task {task_id}: ObjectKeeper 未启动，跳过保活注册")
        except Exception as e:
            logger.error(f"[Preprocess] Task {task_id}: 注册 ObjectKeeper 失败: {e}")

        return {
            "task_id": task_id,
            "video_path": video_path,
            "total_frames": total_frames,
            "frames_ref": frames_ref,
        }
    except Exception as e:
        logger.error(f"[Preprocess] 预处理失败 task {task_data.get('task_id', '')}: {e}")
        return {"task_id": task_data.get("task_id", ""), "error": str(e)}


# ============================================================
# 流水线阶段 2: GPU 推理 (stateful, 持有 PyTorch 模型)
# ============================================================

@ray.remote(num_gpus=1 / 10)
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
        torch.cuda.reset_peak_memory_stats()
        _ = self.inferencer.infer(_warm_img)
        torch.cuda.synchronize()
        _base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # GPU tensor 路径 warmup
        _warm_tensor = torch.randint(0, 256, (3, 2048, 2048), dtype=torch.uint8, device=self.device)
        _ = self.inferencer.infer_tensor(_warm_tensor, (2048, 2048))
        torch.cuda.synchronize()
        self.per_frame_mem_mb = (torch.cuda.max_memory_allocated() - _base) / 1024 ** 2
        del _warm_img, _warm_tensor
        self._worker_logger.info(f"[GPU-{self.worker_idx}] 模型加载完成, per_frame_mem={self.per_frame_mem_mb:.1f}MB")

    def infer(self, preprocessed_data: dict) -> dict:
        """GPU 推理：零拷贝流水线，直接将 numpy 转为 PyTorch Tensor View，按帧 permute+异步 H2D。"""
        import torch
        import itertools
        import supervision as sv
        import numpy as np
        from trackers import CBIoUTracker

        task_id = preprocessed_data["task_id"]
        video_path = preprocessed_data["video_path"]
        total_frames = preprocessed_data["total_frames"]
        frames_ref = preprocessed_data["frames_ref"]
        self._worker_logger.info(f"[GPU-{self.worker_idx}] 任务 {task_id}: 开始推理, {total_frames}帧")

        frames_np = ray.get(frames_ref)

        _t0 = time.perf_counter()

        try:
            _free, _total = torch.cuda.mem_get_info(self.device)
            safety_ratio = 0.60
            _max_streams = max(
                1, int(_free * safety_ratio / (self.per_frame_mem_mb * 1024 ** 2))
            )
            n_streams = min(self.num_streams, _max_streams, 8)

            streams = [torch.cuda.Stream() for _ in range(n_streams)]
            stream_cycle = itertools.cycle(streams)

            frame_data = [None] * total_frames
            orig_size = (frames_np.shape[2], frames_np.shape[1])  # (W, H)

            # 直接将 NumPy 数组转为 PyTorch Tensor View（零拷贝）
            # 先确保数组可写，避免 PyTorch 警告
            if not frames_np.flags.writeable:
                frames_np = frames_np.copy()
            tensor_nhwc = torch.from_numpy(frames_np)

            for i in range(total_frames):
                if self.shutdown_event:
                    self._worker_logger.info(
                        f"[GPU-{self.worker_idx}] task {task_id}: 收到停止信号，退出推理"
                    )
                    break

                s = next(stream_cycle)
                with torch.cuda.stream(s):
                    # 关键优化：按帧做维度转换 permute(2,0,1) 并异步推送到 GPU
                    # 避免对全量数据执行 ascontiguousarray() 和 pin_memory() 导致产生几 GB 的私有副本
                    frame_gpu = (
                        tensor_nhwc[i].permute(2, 0, 1).cuda(self.device, non_blocking=True)
                    )

                    results = self.inferencer.infer_tensor(frame_gpu, orig_size)

                    # 关键优化：不要在这里生成全量 PIL Image 对象！
                    # 仅保存帧索引或者元数据；如果第三阶段需要图像，直接让第三阶段拿着 frames_np[i] 去转换
                    frame_data[i] = {
                        "frame_idx": i,
                        "results": results,
                        "frame_name": f"{i:06d}.jpg",
                    }

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
                    xyxy = np.array([r.box for r in track_results])
                    conf = np.array([r.score for r in track_results])
                    cls = np.array([r.label for r in track_results])
                    detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
                    tracked = tracker.update(detections)

                    box_to_tid = {}
                    for tb, tid in zip(tracked.xyxy, tracked.tracker_id):
                        box_to_tid[tuple(tb)] = int(tid)
                    for r in track_results:
                        r.track_id = box_to_tid.get(tuple(r.box), -1)
                self._worker_logger.info(f"[GPU-{self.worker_idx}] 跟踪总耗时: {(time.perf_counter()-_t_track)*1000:.1f}ms / {len(frame_data)}帧")
            except Exception as e:
                self._worker_logger.warning(f"[GPU-{self.worker_idx}] 跟踪失败（跳过）: {e}")

            from streaming_cluster_updater import process_video_incremental
            output_parent = Path(self.config_dict["output_dir"]) / "temp_test" / task_id
            summary = process_video_incremental(
                frame_data=frame_data,
                output_root=output_parent,
                draw_func=None,
                updater=self.updater,
                gpu_id=0,
                save_output=False,
                frames_np=frames_np,  # 传递 numpy 引用，下游按需转换为 PIL
            )

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            return {
            "task_id": task_id,
            "video_path": video_path,
            "summary": summary,
            "frames_ref": preprocessed_data.get("frames_ref"),
            "total_frames": total_frames,
        }

        except torch.cuda.OutOfMemoryError as e:
            self._worker_logger.error(f"[GPU-{self.worker_idx}] OOM: {e}")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return {"task_id": task_id, "error": f"OOM: {e}"}
        except Exception as e:
            self._worker_logger.error(f"[GPU-{self.worker_idx}] 任务 {task_id} 失败: {e}")
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
    """CPU 后处理：构建最终 result_data。"""
    try:
        task_id = infer_results.get("task_id", "")
        logger.info(f"[Postprocess] Task {task_id}: 开始后处理")
        if "error" in infer_results:
            logger.error(f"[Postprocess] task {task_id}: 推理阶段失败: {infer_results['error']}")
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
        logger.info(f"[Postprocess] task {task_id}: 结果构建完成")
        return result_data
    except Exception as e:
        logger.error(f"[Postprocess] 构建结果失败: {e}")
        return {"task_id": task_data.get("task_id", ""), "status": "failed", "error": str(e)}


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

    def run(self):
        import json
        from pathlib import Path


        # pipeline state: task_id -> {stage, futures, task_data}
        pipeline = {}
        disk_full_backoff_until = 0.0

        try:
            while not self.shutdown_event:
                # 1. 检查各流水线阶段完成情况
                completed = []
                for task_id, state in list(pipeline.items()):
                    stage = state['stage']

                    if stage == 'preprocess':
                        ready, _ = ray.wait([state['preprocess_future']], timeout=0)
                        if ready:
                            preprocessed = ray.get(ready[0])
                            if 'error' in preprocessed:
                                self._worker_logger.error(f"[Scheduler-{self.worker_idx}] task {task_id}: 预处理失败: {preprocessed['error']}")
                                is_disk_full = "disk is full" in str(preprocessed.get('error', ''))
                                if is_disk_full:
                                    disk_full_backoff_until = time.time() + 30  # 退避 30s
                                    self._worker_logger.warning(f"[Scheduler-{self.worker_idx}] 检测到磁盘已满, 暂停轮询 {30}s")
                                completed.append(task_id)
                            else:
                                infer_future = self.gpu_actor.infer.remote(preprocessed)
                                state['infer_future'] = infer_future
                                state['stage'] = 'infer'
                                self._worker_logger.info(f"[Scheduler-{self.worker_idx}] task {task_id}: 进入推理阶段")

                    elif stage == 'infer':
                        ready, _ = ray.wait([state['infer_future']], timeout=0)
                        if ready:
                            infer_results = ray.get(ready[0])
                            post_future = api_postprocess.remote(infer_results, state['task_data'])
                            state['postprocess_future'] = post_future
                            state['stage'] = 'postprocess'
                            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] task {task_id}: 进入后处理阶段")

                    elif stage == 'postprocess':
                        ready, _ = ray.wait([state['postprocess_future']], timeout=0)
                        if ready:
                            result_data = ray.get(ready[0])
                            self._send_result(result_data, task_id)
                            self.kafka_consumer.commit(asynchronous=False)  # 后处理完成后提交offset
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
                        }
                        self._worker_logger.info(f"[Scheduler-{self.worker_idx}] task {task_id}: 进入预处理阶段")

                time.sleep(0.01)
        finally:
            self._worker_logger.info(f"[Scheduler-{self.worker_idx}] 正在关闭 Kafka 连接...")
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
        from confluent_kafka.admin import AdminClient

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
            # 加一个随机前缀，避免与正式消费者的 group.instance.id 冲突，仅用于只读查询
            consumer_conf["group.id"] = group
            consumer_conf["enable.auto.commit"] = False

            c = None
            try:
                c = Consumer(consumer_conf)
                
                # 获取 Topic 元数据与分区数
                md = c.list_topics(topic, timeout=10)
                if topic not in md.topics or md.topics[topic].error:
                    results[topic] = {"lag": -1, "error": f"topic {topic} 不存在或不可用"}
                    continue

                partitions_meta = md.topics[topic].partitions
                tps = [TopicPartition(topic, p) for p in partitions_meta.keys()]

                # 查询真实消费者组对这些分区的 committed offset（不会 trigger join group）
                committed_tps = c.committed(tps, timeout=10)
                committed_map = {tp.partition: tp.offset for tp in committed_tps}

                total_lag = 0
                partition_lags = {}
                max_partition_lag = 0

                # 遍历每个分区查 watermark (low, high)
                for p_id in partitions_meta.keys():
                    low, high = c.get_watermark_offsets(TopicPartition(topic, p_id), timeout=10)
                    committed_val = committed_map.get(p_id, 0)
                    
                    # 如果 offset < 0 (例如 OFFSET_INVALID -1001)，说明该分区从未提交过，从 low 开始算
                    if committed_val < 0:
                        committed_val = low

                    lag = max(0, high - committed_val)
                    total_lag += lag
                    partition_lags[p_id] = lag
                    if lag > max_partition_lag:
                        max_partition_lag = lag

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
        runtime_env={"env_vars": {"PYTHONPATH": os.getcwd(), "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.GPU_IDS)), "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/targets/x86_64-linux/lib"}},
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

    for w_idx in range(config.MIN_TOTAL_WORKERS):
        gpu_actor = ApiGPUInferActor.remote(w_idx, cfg_dict)
        gpu_actors.append(gpu_actor)
        scheduler = ApiSchedulerActor.remote(w_idx, gpu_actor, cfg_dict)
        scheduler_actors.append(scheduler)
    logger.info(f"🚀 已创建 {config.MIN_TOTAL_WORKERS} 个 Worker (GPU+Scheduler 配对)")

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
                partitions = topic_lag.get("partitions", 0)
                max_partition_lag = topic_lag.get("max_partition_lag", 0)
                partition_lags = topic_lag.get("partition_lags", {})
                logger.info(f"📊 Kafka 积压检测: {config.KAFKA_TASK_TOPIC} total_lag={lag}, partitions={partitions}, max_partition_lag={max_partition_lag}, 当前 workers={len(scheduler_actors)}")
                if partition_lags:
                    logger.info(f"📊 各分区积压: {partition_lags}")

                num_workers = len(scheduler_actors)
                lag_per_worker = lag / num_workers if num_workers > 0 else float('inf')
                need_scale_up = False

                if num_workers == 0 and lag > 0:
                    need_scale_up = True
                    logger.info(f"📈 无消费者但 topic 有积压 (lag={lag}), 需要立即添加消费者")
                elif lag_per_worker > config.SCALE_UP_THRESHOLD and num_workers < config.MAX_TOTAL_WORKERS:
                    need_scale_up = True
                    logger.info(f"📈 单消费者积压超过阈值 (lag={lag}, workers={num_workers}, lag/worker={lag_per_worker:.1f} > {config.SCALE_UP_THRESHOLD})")

                if need_scale_up:
                    gpu_free = _get_gpu_free_memory()
                    max_possible = _calculate_max_workers(gpu_free, config.MAX_TOTAL_WORKERS)

                    if max_possible > num_workers:
                        new_workers = 1
                        logger.info(f"📈 扩展 1 个 worker")

                        ensure_kafka_partitions(num_workers + new_workers)

                        w_idx = num_workers
                        gpu_actor = ApiGPUInferActor.remote(w_idx, cfg_dict)
                        gpu_actors.append(gpu_actor)
                        scheduler = ApiSchedulerActor.remote(w_idx, gpu_actor, cfg_dict)
                        scheduler_actors.append(scheduler)
                        logger.info(f"🚀 新增 Worker-{w_idx} (GPU+Scheduler)")
                        scheduler.run.remote()

                        last_scale_time = current_time

                elif max_partition_lag <= config.SCALE_DOWN_THRESHOLD and len(scheduler_actors) > 0:
                    remove_count = 1
                    logger.info(f"📉 所有分区积压低于阈值 (max_partition_lag={max_partition_lag} <= {config.SCALE_DOWN_THRESHOLD}), 收缩 1 个 worker")

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
