"""
HoldRope 关键点检测服务 (流水线架构)
基于 Kafka + Ray Pipeline 分布式推理
将预处理、GPU 推理、后处理拆分为独立流水线阶段，实现 CPU/GPU 并行

流程:
  video_results ──→ kps_preprocess(CPU) ──→ KpsGPUInferActor(GPU) ──→ kps_postprocess(CPU) ──→ hold_rope_keypoints
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
from pathlib import Path
from typing import Dict

import ray
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
from video_utils import read_video_frames


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/monitor_hold_rope_ray.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("hold_rope_monitor_onnx")


class Config:
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_CONSUME_TOPIC = "video_results"
    KAFKA_PRODUCE_TOPIC = "hold_rope_keypoints"
    KAFKA_FINAL_TOPIC = "final_videos"
    KAFKA_CONSUMER_GROUP = "hold_rope_consumers"

    TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")
    GPU_IDS = [6]
    WORKERS_PER_GPU = 4  # 每张 GPU 最多 2 个 Worker (monitor + hand_crop 共用 GPU 4)
    RESOURCE_GPU = "gpu_4_node"  # 伪节点资源标签，锁定 GPU 4

    RAY_OBJECT_STORE_MEMORY = 16 * 1024 ** 3

    LAG_CHECK_INTERVAL = 600
    SCALE_UP_THRESHOLD = 5
    SCALE_DOWN_THRESHOLD = 1
    MAX_TOTAL_WORKERS = 4
    MIN_TOTAL_WORKERS = 2

    PIPELINE_DEPTH = 3


config = Config()


# ==================== 初始化 Topic ====================
def init_topics():
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
    })
    topics_needed = [config.KAFKA_CONSUME_TOPIC, config.KAFKA_PRODUCE_TOPIC, config.KAFKA_FINAL_TOPIC]
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


def _collect_crops(labels, frame_bgr, height, width):
    crops = []
    offset_list = []
    label_indices = []

    for label_idx, label in enumerate(labels):
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
        crop_x1 = max(0, int(cx - new_w / 2))
        crop_y1 = max(0, int(cy - new_h / 2))
        crop_x2 = min(width, int(cx + new_w / 2))
        crop_y2 = min(height, int(cy + new_h / 2))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        cropped = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.size == 0:
            continue

        crops.append(cropped)
        offset_list.append((crop_x1, crop_y1))
        label_indices.append(label_idx)

    return crops, offset_list, label_indices


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


# ==================== 流水线 Task：CPU 预处理 ====================

@ray.remote(num_cpus=2)
def kps_preprocess(task_data: dict) -> dict:
    """CPU 预处理：读取视频帧、验证文件路径"""
    task_id = task_data.get("task_id", "")
    video_path_str = task_data.get("summary", {}).get("video_path", "")
    print(f"[Stage-Preprocess] task {task_id}: START, video={video_path_str}", flush=True)

    if not task_id:
        print(f"[Stage-Preprocess] task {task_id}: END (failed - missing task_id)", flush=True)
        return {"task_id": task_id, "status": "failed", "error": "missing task_id", "fail_sent": False}

    _t0 = time.perf_counter()

    task_dir = config.TEMP_DIR / task_id
    labels_dir = task_dir / "labels"

    if not labels_dir.exists():
        print(f"[Stage-Preprocess] task {task_id}: END (failed - labels dir not found)", flush=True)
        return {"task_id": task_id, "status": "failed", "error": "labels dir not found", "fail_sent": False}

    if not video_path_str:
        print(f"[Stage-Preprocess] task {task_id}: END (failed - video_path not found)", flush=True)
        return {"task_id": task_id, "status": "failed", "error": "video_path not found", "fail_sent": False}

    # === 从 ObjectKeeper 获取 ref（不再依赖 Kafka 传递 frames_ref）===
    total_frames_from_msg = task_data.get("summary", {}).get("total_frames")
    frames_source = "ref"
    frames_np = None
    n_frames = total_frames_from_msg

    try:
        keeper = ray.get_actor("ObjectKeeper", namespace="production")
        frames_np = ray.get(keeper.get_ref.remote(task_id))
        if frames_np is not None:
            print(f"[Stage-Preprocess] task {task_id}: ✅ 从 ObjectKeeper 获取 ref 成功, shape={frames_np.shape}", flush=True)
        else:
            print(f"[Stage-Preprocess] task {task_id}: ⚠️ ObjectKeeper 中无 ref", flush=True)
    except Exception as e:
        print(f"[Stage-Preprocess] task {task_id}: 从 ObjectKeeper 获取失败: {e}", flush=True)
    
    # 回退到磁盘读取
    if frames_np is None:
        frames_source = "disk"
        video_path = Path(video_path_str)
        print(f"[Stage-Preprocess] task {task_id}: 回退到磁盘读取: {video_path}", flush=True)
        if not video_path.exists():
            print(f"[Stage-Preprocess] task {task_id}: END (failed - video not found)", flush=True)
            return {"task_id": task_id, "status": "failed", "error": f"video not found: {video_path}", "fail_sent": False}
        frames_np = read_video_frames(str(video_path))
        n_frames = len(frames_np)
        print(f"[Stage-Preprocess] task {task_id}: 磁盘读取完成, 共 {n_frames} 帧", flush=True)
        # 磁盘读取的也注册到 ObjectKeeper，以便 GPU Actor 后续获取
        try:
            keeper = ray.get_actor("ObjectKeeper", namespace="production")
            frames_ref = ray.put(frames_np)
            # ⚠️ 同步等待注册完成：数据需先传输到 keeper 所在节点，否则下游 GPU Actor 会查不到
            _t_reg = time.perf_counter()
            ok = ray.get(keeper.register_ref.remote(task_id, frames_ref, ttl=1800))
            print(f"[Stage-Preprocess] task {task_id}: ✅ 已将磁盘读取的帧数据注册到 ObjectKeeper "
                  f"(shape={frames_np.shape}, ok={ok}, {(time.perf_counter()-_t_reg)*1000:.0f}ms)", flush=True)
        except Exception as e:
            print(f"[Stage-Preprocess] task {task_id}: ❌ 注册 ObjectKeeper 失败: {e}", flush=True)

    # 如果 n_frames 仍未知，通过 ref 获取
    if n_frames is None:
        n_frames = len(frames_np)

    label_files = sorted(labels_dir.glob("*.json"))
    dt = (time.perf_counter() - _t0) * 1000
    print(f"[Stage-Preprocess] task {task_id}: END (OK), frames={n_frames}, labels={len(label_files)}, source={frames_source}, time={dt:.0f}ms", flush=True)

    # 清理本地 frames_np 引用，避免返回巨形数组
    del frames_np
    import gc
    gc.collect()

    return {
        "task_id": task_id,
        "video_path": video_path_str,
        "frames_source": frames_source,
        "label_files": [str(lf) for lf in label_files],
        "labels_dir": str(labels_dir),
        "n_frames": n_frames,
        "fail_sent": False,
    }


# ==================== GPU 推理 Actor ====================

@ray.remote
class KpsGPUInferActor:
    def __init__(self, worker_idx: int):
        self.worker_idx = worker_idx
        self._logger = logging.getLogger(f"hold_rope_monitor_onnx.ray_{worker_idx}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        fh = logging.FileHandler('/data2/caimoyao/logs/monitor_hold_rope_ray.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._logger.addHandler(sh)

        self._init_model()

    def _init_model(self):
        self._logger.info(f"[KpsGPU-{self.worker_idx}] 加载模型...")

        # 统一 CUDA 上下文：先通过 PyTorch 初始化，确保 PyTorch 和 ORT 共享同一个上下文
        import torch as _torch
        _torch.cuda.set_device(0)
        _torch.empty((1,), device='cuda:0')  # 触发 CUDA 上下文创建
        self._logger.info(f"[KpsGPU-{self.worker_idx}] CUDA 上下文已初始化 (cuda:{0})")

        from inference_tensorRT_multi_rtmpose_l_stream_ray import RTMPoseModelStream
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "weights", "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
        )
        self.model = RTMPoseModelStream(model_path, device_id=0)
        self._logger.info(f"[KpsGPU-{self.worker_idx}] 模型加载完成")

    def infer(self, preprocessed_data: dict) -> dict:
        """GPU 推理：对预处理的帧数据进行关键点检测"""
        from inference_tensorRT_multi_rtmpose_l_stream_ray import batch_preprocess_images, batch_postprocess

        task_id = preprocessed_data["task_id"]
        frames_source = preprocessed_data.get("frames_source", "?")
        self._logger.info(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: START (frames_source={frames_source})")
        _t0 = time.perf_counter()

        # 从 ObjectKeeper 获取帧数据（避免通过 Task 返回值传递巨形数组）
        try:
            keeper = ray.get_actor("ObjectKeeper", namespace="production")
            frames_np = ray.get(keeper.get_ref.remote(task_id))
        except Exception:
            frames_np = None
        if frames_np is None:
            # fallback: 从磁盘读取帧数据（ObjectKeeper 中无 ref 或已过期）
            video_path_str = preprocessed_data["video_path"]
            self._logger.warning(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: ObjectKeeper 无 ref，回退到磁盘读取: {video_path_str}")
            from video_utils import read_video_frames
            frames_np = read_video_frames(video_path_str)
            self._logger.info(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: 磁盘读取完成, {len(frames_np)} 帧")
        if frames_np is None:
            self._logger.error(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: 无法获取帧数据")
            return {"task_id": task_id, "status": "failed", "error": "no frame data", "fail_sent": False}

        label_file_strs = preprocessed_data["label_files"]
        labels_dir = Path(preprocessed_data["labels_dir"])
        video_path_str = preprocessed_data["video_path"]
        n_frames = preprocessed_data["n_frames"]

        label_files = [Path(lf) for lf in label_file_strs]

        has_error = False
        fail_sent = False
        processed_count = 0
        pending = None

        def flush_pending():
            nonlocal pending, processed_count, has_error
            if pending is None:
                return
            cpu_out, pi, off, idxs, lbs, lf, actual_n = pending
            fid = int(Path(lf).stem)
            try:
                outputs = self.model.wait(cpu_out, frame_idx=fid)
                outputs = [o[:actual_n] for o in outputs]
                all_kps = batch_postprocess(outputs, pi, off)
                for kps, lidx in zip(all_kps, idxs):
                    lbs[lidx]["keypoints"] = kps
                with open(lf, 'w', encoding='utf-8') as f:
                    json.dump(lbs, f, ensure_ascii=False, indent=2)
                processed_count += 1
            except Exception as e:
                self._logger.error(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: 后处理/保存标签文件 {lf} 失败: {e}")
                has_error = True
            pending = None

        try:
            for label_file in label_files:
                frame_idx = int(label_file.stem)
                if frame_idx >= n_frames:
                    self._logger.warning(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: 帧索引 {frame_idx} 超出范围 ({n_frames})，跳过")
                    continue

                frame_bgr = None
                frame_np = None
                frame_np = frames_np[frame_idx]
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                height, width = frame_bgr.shape[:2]

                try:
                    with open(label_file, 'r') as f:
                        labels = json.load(f)

                    crops, offsets, indices = _collect_crops(labels, frame_bgr, height, width)

                    if not crops:
                        with open(label_file, 'w', encoding='utf-8') as f:
                            json.dump(labels, f, ensure_ascii=False, indent=2)
                        processed_count += 1
                        del crops, offsets, indices, labels
                        continue

                    batch, actual_n, pad_info = batch_preprocess_images(
                        crops, input_size=self.model.input_size,
                        pinned_buffer=self.model._input_pinned,
                    )

                    flush_pending()

                    cpu_outputs = self.model.submit(batch, frame_idx=frame_idx)
                    pending = (cpu_outputs, pad_info, offsets, indices, labels, label_file, actual_n)

                except Exception as e:
                    self._logger.error(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: 处理标签文件 {label_file} 失败: {e}")
                    has_error = True
                    _release_keeper_ref(task_id, self._logger)
                finally:
                    if frame_bgr is not None:
                        del frame_bgr
                    if frame_np is not None:
                        del frame_np

            flush_pending()
        finally:
            # 确保无论是否发生异常，pending 内的大 Tensor 都能被清空
            if pending is not None:
                del pending
                pending = None
            # 清理可能逃逸的循环变量
            for v in ['crops', 'offsets', 'indices', 'labels', 'label_file']:
                try:
                    del v
                except NameError:
                    pass
            del frames_np, label_files
            import gc
            gc.collect()

        dt = (time.perf_counter() - _t0) * 1000
        self._logger.info(f"[Stage-Infer] [KpsGPU-{self.worker_idx}] task {task_id}: END (processed={processed_count}, error={has_error}, time={dt:.0f}ms)")

        return {
            "task_id": task_id,
            "status": "completed" if not has_error else "partial_error",
            "processed_labels": processed_count,
            "video_path": video_path_str,
            "output_labels": str(labels_dir),
            "fail_sent": fail_sent,
        }


# ==================== 流水线 Task：CPU 后处理 ====================

@ray.remote(num_cpus=1)
def kps_postprocess(infer_results: dict, task_data: dict) -> dict:
    """CPU 后处理：构建最终结果，发送错误消息"""
    result = infer_results
    task_id = result.get("task_id", "")
    print(f"[Stage-Postprocess] task {task_id}: START", flush=True)
    _t0 = time.perf_counter()

    if result.get("status") != "completed":
        dt = (time.perf_counter() - _t0) * 1000
        print(f"[Stage-Postprocess] task {task_id}: END (status={result.get('status')}, time={dt:.0f}ms)", flush=True)
        return result

    dt = (time.perf_counter() - _t0) * 1000
    print(f"[Stage-Postprocess] task {task_id}: END (OK, time={dt:.0f}ms)", flush=True)
    return result


# ==================== 调度 Actor ====================

@ray.remote
class KpsSchedulerActor:
    def __init__(self, worker_idx: int, gpu_actor):
        self.worker_idx = worker_idx
        self.gpu_actor = gpu_actor
        self.shutdown_event = False

        self._logger = logging.getLogger(f"hold_rope_monitor_onnx.scheduler_{worker_idx}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        fh = logging.FileHandler('/data2/caimoyao/logs/monitor_hold_rope_ray.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._logger.addHandler(sh)

        self._init_kafka()

    def _init_kafka(self):
        from confluent_kafka import Consumer, Producer

        def on_assign(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._logger.info(f"[Scheduler-{self.worker_idx}] ✅ 分配到分区: {p_ids}")

        def on_revoke(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._logger.info(f"[Scheduler-{self.worker_idx}] ❌ 被回收分区: {p_ids}")

        def on_lost(consumer, partitions):
            p_ids = [p.partition for p in partitions]
            self._logger.warning(f"[Scheduler-{self.worker_idx}] ⚠️ 丢失分区: {p_ids}")

        self.kafka_consumer = Consumer({
            "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": config.KAFKA_CONSUMER_GROUP,
            "group.instance.id": f"monitor-ray-{self.worker_idx}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": 14400000,
            "session.timeout.ms": 180000,
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
        self._logger.info(f"[Scheduler-{self.worker_idx}] Consumer 已连接")

        self.kafka_producer = Producer({
            "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
            "broker.address.family": "v4",
        })
        self._logger.info(f"[Scheduler-{self.worker_idx}] Producer 已连接")

    def run(self):
        pending = {}  # task_id -> (postprocess_ref, task_data, msg, created_time)
        pipeline_depth = config.PIPELINE_DEPTH
        # task_timeout = 1800  # 单任务最长等待 30 分钟，超时则清理

        try:
            while not self.shutdown_event:
                # 流水线满时跳过拉取，专心等待已完成任务
                if len(pending) >= pipeline_depth:
                    time.sleep(0.05)
                else:
                    try:
                        msg = self.kafka_consumer.poll(timeout=0.5)
                    except Exception as e:
                        self._logger.error(f"[Scheduler-{self.worker_idx}] Kafka poll 异常: {e}")
                        time.sleep(1)
                        continue

                    if msg is None:
                        time.sleep(0.05)
                    elif msg.error():
                        self._logger.error(f"[Scheduler-{self.worker_idx}] poll error: {msg.error()}")
                        time.sleep(1)
                    else:
                        try:
                            task_data = json.loads(msg.value().decode('utf-8'))
                        except Exception as e:
                            self._logger.error(f"[Scheduler-{self.worker_idx}] 消息解析失败: {e}")
                            continue

                        task_id = task_data.get("task_id", "")
                        status = task_data.get("status", "")

                        if status != "completed":
                            self._logger.debug(f"[Scheduler-{self.worker_idx}] 跳过非 completed 消息: {task_id}")
                            self.kafka_consumer.commit(msg)  # 提交该消息的 offset
                            continue

                        self._logger.info(f"[Scheduler-{self.worker_idx}] 收到任务: {task_id}")

                        if len(pending) >= pipeline_depth:
                            self._logger.debug(f"[Scheduler-{self.worker_idx}] 流水线已满 ({len(pending)}), 跳过 {task_id}")
                            self.kafka_consumer.commit(msg)  # 提交 offset 避免消息丢失
                        else:
                            # 启动流水线：preprocess → infer → postprocess
                            pre_ref = kps_preprocess.remote(task_data)
                            infer_ref = self.gpu_actor.infer.remote(pre_ref)
                            post_ref = kps_postprocess.remote(infer_ref, task_data)
                            pending[task_id] = (post_ref, task_data, msg, time.time())  # 保存 msg + 创建时间用于超时检测
                            self._logger.info(f"[Scheduler-{self.worker_idx}] 启动流水线: {task_id}")

                # 收集已完成的任务结果
                if pending:
                    ready_ids = []
                    expired_ids = []
                    now = time.time()
                    for tid, (ref, tdata, msg, created_time) in pending.items():
                        # 超时检测：超过 task_timeout 秒的任务，强制清理
                        # if now - created_time > task_timeout:
                        #     expired_ids.append(tid)
                        #     self._logger.warning(f"[Scheduler-{self.worker_idx}] task {tid} 超时 ({task_timeout}s), 强制清理")
                        #     _release_keeper_ref(tid, self._logger)
                        #     try:
                        #         self.kafka_consumer.commit(msg)
                        #     except Exception:
                        #         pass
                        #     continue
                        ready, _ = ray.wait([ref], timeout=0)
                        if ready:
                            ready_ids.append(tid)

                    # 清理超时任务
                    for tid in expired_ids:
                        pending.pop(tid, None)

                    for tid in ready_ids:
                        ref, tdata, msg, _ = pending.pop(tid)
                        try:
                            result = ray.get(ref)
                            
                            if result.get("status") == "completed":
                                self.kafka_producer.produce(
                                    config.KAFKA_PRODUCE_TOPIC,
                                    key=tid.encode('utf-8'),
                                    value=json.dumps(result).encode('utf-8'),
                                )
                                self.kafka_producer.flush(timeout=10)
                                self.kafka_consumer.commit(msg)  # 提交该消息的 offset
                                self._logger.info(f"[Scheduler-{self.worker_idx}] 结果已发送: {tid} → {config.KAFKA_PRODUCE_TOPIC}")
                            else:
                                # 失败消息
                                fail_msg = {
                                    "task_id": tid,
                                    "error": result.get("error", f"status={result.get('status')}"),
                                    "failed_at": "monitor_hold_rope",
                                    "status": "failed",
                                }
                                if result.get("fail_sent"):
                                    self._logger.info(f"[Scheduler-{self.worker_idx}] task {tid}: 失败消息已由内层处理发送")
                                else:
                                    video_path = tdata.get("video_path", tdata.get("summary", {}).get("video_path", ""))
                                    fail_msg["video_path"] = video_path
                                    self.kafka_producer.produce(
                                        config.KAFKA_FINAL_TOPIC,
                                        key=tid.encode('utf-8'),
                                        value=json.dumps(fail_msg).encode('utf-8'),
                                    )
                                    self.kafka_producer.flush(timeout=10)
                                    self._logger.info(f"[Scheduler-{self.worker_idx}] 失败消息已发送: {tid}")
                                # 失败消息也提交 offset，避免重复消费
                                self.kafka_consumer.commit(msg)
                        except Exception as e:
                            self._logger.error(f"[Scheduler-{self.worker_idx}] task {tid}: 收集结果异常: {e}")
                            _release_keeper_ref(tid, self._logger)  # 报错也释放 ObjectKeeper 引用
                            # 即使失败也要 commit，避免无限卡住
                            try:
                                self.kafka_consumer.commit(msg)
                            except Exception:
                                pass
                        finally:
                            # 强行释放对 Ray ObjectRef 的句柄绑定
                            del ref

        finally:
            self._logger.info(f"[Scheduler-{self.worker_idx}] 正在关闭 Kafka 连接...")
            # 主动停止时释放所有仍挂起任务的 ObjectKeeper 引用，避免 object store 占用
            for _tid in list(pending.keys()):
                _release_keeper_ref(_tid, self._logger)
                self._logger.info(f"[Scheduler-{self.worker_idx}] 主动停止, 释放引用: {_tid}")
            try:
                self.kafka_consumer.close()
            except Exception as e:
                self._logger.warning(f"[Scheduler-{self.worker_idx}] Consumer 关闭异常: {e}")
            try:
                self.kafka_producer.close()
            except Exception as e:
                self._logger.warning(f"[Scheduler-{self.worker_idx}] Producer 关闭异常: {e}")
            self._logger.info(f"[Scheduler-{self.worker_idx}] scheduler 退出")

    def stop(self):
        self.shutdown_event = True


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
                # 特意指定 group 参数以查询指定消费组的 offset
                # 注意：c.committed 在某些版本可能不支持指定 group，
                # 但这里我们主要目的是监控，如果 c.committed 不行，可以 fallback 到 AdminClient
                try:
                    # 对于 confluent-kafka-python，Consumer.committed 只能查当前 group.id 的
                    # 所以我们需要把 group.id 设置为我们要查询的 group
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
    WORKER_MEM_GB = 10.0
    WORKER_MEM_MB = int(WORKER_MEM_GB * 1024)

    total_workers = 0
    for gpu_id in config.GPU_IDS:
        free_mb = gpu_free.get(gpu_id, 0)
        if free_mb < WORKER_MEM_MB:
            continue
        n_workers = min(free_mb // WORKER_MEM_MB, config.WORKERS_PER_GPU)
        total_workers += n_workers
    return min(total_workers, max_total)


def main():
    print("=" * 60)
    print("   HoldRope 关键点检测服务 (Ray 流水线)")
    print("=" * 60)
    print(f"🌐 Kafka: {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"📥 消费: {config.KAFKA_CONSUME_TOPIC}")
    print(f"📤 产出: {config.KAFKA_PRODUCE_TOPIC}")
    print(f"👥 消费者组: {config.KAFKA_CONSUMER_GROUP}")
    print(f"🎮 GPU: {config.GPU_IDS}")
    print(f"⚙️ Pipeline Depth: {config.PIPELINE_DEPTH}")
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

    # 创建 GPU Actor 和 Scheduler Actor 配对
    gpu_actors = []
    schedulers = []

    gpu_fraction = 1.0 / config.WORKERS_PER_GPU
    gpu_resource = {config.RESOURCE_GPU: 1.0}  # 锁定到伪节点，每个 Worker 占用 1 单位资源

    for w_idx in range(config.MIN_TOTAL_WORKERS):
        gpu_actor = KpsGPUInferActor.options(
            num_gpus=gpu_fraction,
            resources=gpu_resource
        ).remote(w_idx)
        gpu_actors.append(gpu_actor)
        scheduler = KpsSchedulerActor.remote(w_idx, gpu_actor)
        schedulers.append(scheduler)
        scheduler.run.remote()
        logger.info(
            f"🚀 Worker-{w_idx}: GPU Actor ({gpu_fraction:.2f} GPU, "
            f"resource={config.RESOURCE_GPU}) + Scheduler Actor 已创建"
        )
    logger.info(f"🚀 共创建 {config.MIN_TOTAL_WORKERS} 个 Worker")

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
                logger.info(f"📊 Kafka 积压检测: {config.KAFKA_CONSUME_TOPIC} total_lag={lag}, partitions={partitions}, max_partition_lag={max_partition_lag}, 当前 workers={len(schedulers)}")
                if partition_lags:
                    logger.info(f"📊 各分区积压: {partition_lags}")

                num_workers = len(schedulers)
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

                    print(f"gpu_free={gpu_free}, GPU{config.GPU_IDS[0]}空闲={gpu_free.get(config.GPU_IDS[0], 0)/1024:.2f}GB, max_possible={max_possible}")

                    if max_possible > 0:
                        logger.info(f"📈 扩展 1 个 worker")

                        ensure_kafka_partitions(num_workers + 1)

                        w_idx = num_workers
                        gpu_a = KpsGPUInferActor.options(
                            num_gpus=gpu_fraction,
                            resources=gpu_resource
                        ).remote(w_idx)
                        sch = KpsSchedulerActor.remote(w_idx, gpu_a)
                        gpu_actors.append(gpu_a)
                        schedulers.append(sch)
                        sch.run.remote()
                        logger.info(f"🚀 新增 Worker-{w_idx}: GPU Actor + Scheduler Actor, resource={config.RESOURCE_GPU}")

                        last_scale_time = current_time

                elif lag_per_worker < config.SCALE_DOWN_THRESHOLD and len(schedulers) > config.MIN_TOTAL_WORKERS:
                    logger.info(f"📉 平均每 worker 积压低于阈值 (lag_per_worker={lag_per_worker:.1f} < {config.SCALE_DOWN_THRESHOLD}), 收缩 1 个 worker")

                    if len(schedulers) == 0:
                        pass
                    else:
                        sch = schedulers.pop()
                        gpu_a = gpu_actors.pop()

                        # 1. 通知 Scheduler 停止循环
                        try:
                            ray.get(sch.stop.remote(), timeout=5)
                        except Exception:
                            pass

                        # 2. 强制杀掉 Ray Actor 进程，彻底释放显存和物理内存！
                        ray.kill(sch)
                        ray.kill(gpu_a)
                        logger.info("⏹️ 已经强制销毁并释放 1 个 Worker 资源")

                    last_scale_time = current_time

            except Exception as e:
                logger.error(f"动态扩缩容失败: {e}")

    except KeyboardInterrupt:
        logger.info("🛑 收到停止信号，正在关闭服务...")
    finally:
        logger.info(f"📋 停止 {len(schedulers)} 个 Scheduler...")
        try:
            ray.get([s.stop.remote() for s in schedulers], timeout=10)
        except Exception:
            pass
        for s in schedulers:
            ray.kill(s)

        logger.info(f"📋 停止 {len(gpu_actors)} 个 GPU Actor...")
        for g in gpu_actors:
            ray.kill(g)

        logger.info("📋 停止 LagMonitor...")
        ray.kill(lag_monitor)

        time.sleep(3)
        ray.shutdown()
        logger.info("✅ 服务已关闭")


if __name__ == "__main__":
    main()
