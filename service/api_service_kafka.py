"""
EdgeCrafter 推理服务
基于 Kafka 消息队列，自动监控 topic 并执行视频推理

功能特性:
- Kafka 消息队列消费 (confluent-kafka)
- GPU 并行推理
- 自动清理临时文件
"""

import os
import sys
import json
import time
import signal
from pathlib import Path
from typing import Dict, Any
import multiprocessing as mp
import logging
import subprocess

from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewPartitions, NewTopic

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/api_service.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
class Config:
    """服务配置类"""

    PROJECT_ROOT = Path("/data2/caimoyao/EdgeCrafter-main/ecdetseg")
    CONFIG_PATH = PROJECT_ROOT / "configs/ecdet/ecdet_l.yml"
    CHECKPOINT_PATH = PROJECT_ROOT / "models/last.pth"

    CONF_THRESHOLD = 0.5
    CLASSES = ["person", "face", "ball"]

    OUTPUT_DIR = Path("/data2/caimoyao/api_outputs")

    GPU_IDS = [6]
    WORKERS_PER_GPU = 4
    NUM_STREAMS = 4

    RAG_CACHE = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/rag_db.pkl"
    RAG_THRESHOLD = 0.75

    SAVE_OUTPUT = False

    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_TASK_TOPIC = "video_tasks"
    KAFKA_RESULT_TOPIC = "video_results"
    KAFKA_FINAL_TOPIC = "final_videos"
    KAFKA_CONSUMER_GROUP = "video_task_consumers"

    KAFKA_MAX_POLL_INTERVAL_MS = 14400000
    KAFKA_SESSION_TIMEOUT_MS = 1800000
    KAFKA_HEARTBEAT_INTERVAL_MS = 3000

config = Config()

mp_ctx = mp.get_context("spawn")
gpu_processes = []


# ==================== 分区管理 ====================
def init_topics():
    """初始化所需的 topic（不存在则创建），不检查分区数。"""
    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
        "security.protocol": "PLAINTEXT",
    })
    topics_to_create = []

    for topic_name in [config.KAFKA_TASK_TOPIC, config.KAFKA_RESULT_TOPIC, config.KAFKA_FINAL_TOPIC]:
        try:
            metadata = admin.list_topics(timeout=10)
            if topic_name not in metadata.topics:
                topics_to_create.append(topic_name)
        except Exception:
            topics_to_create.append(topic_name)

    if not topics_to_create:
        logger.info("所有 topic 已存在，无需创建")
        return

    new_topics = [
        NewTopic(t, num_partitions=1, replication_factor=1) for t in topics_to_create
    ]
    fs = admin.create_topics(new_topics)
    for t in topics_to_create:
        try:
            fs[t].result(timeout=10)
            logger.info(f"✅ Topic {t} 已创建")
        except Exception as e:
            logger.warning(f"创建 Topic {t} 失败: {e}")


def ensure_kafka_partitions(total_consumers: int):
    """检查 topic 分区数，不足时扩展（不负责创建 topic）。"""
    topic = config.KAFKA_TASK_TOPIC
    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
        "security.protocol": "PLAINTEXT",
    })

    # 获取当前分区数
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


# ==================== GPU Worker Process ====================
def gpu_worker_process(
    gpu_id: int,
    w_idx: int,
    config_dict: dict,
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import numpy as np
    from PIL import Image

    device = torch.device("cuda:0")

    _worker_logger = logging.getLogger(f"api_service.gpu_{gpu_id}_{w_idx}")
    _worker_logger.setLevel(logging.INFO)
    _worker_logger.propagate = False
    _worker_logger.handlers.clear()
    _fh = logging.FileHandler('/data2/caimoyao/logs/api_service_01.log', encoding='utf-8')
    _fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _worker_logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _worker_logger.addHandler(_sh)

    sys.path.insert(0, config_dict["project_root"])
    from tools.inference.torch_inf import (
        ECInferencer, build_model, draw, Result, IMAGE_EXTS,
    )

    model, img_size, task = build_model(
        config_dict["config_path"],
        config_dict["checkpoint_path"],
        device,
    )
    inferencer = ECInferencer(
        model=model, task=task, device=device,
        size=img_size, thresh=config_dict.get("thresh", 0.5),
        square_size=1280,
    )
    # _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] 模型加载完成，img_size={img_size}")

    # ── 初始化 RAG ────────────────────────────────────────────────
    updater = None
    rag_cache = config_dict.get("rag_cache_path")
    if rag_cache and os.path.exists(rag_cache):
        try:
            from streaming_cluster_updater import StreamingClusterUpdater
            updater = StreamingClusterUpdater(
                rag_db_path=rag_cache,
                device=device,
            )
            _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] RAG 知识库已加载: "
                        f"{updater.rag_db.total_clusters} clusters, "
                        f"{updater.rag_db.total_samples} samples, "
                        f"threshold={updater.rag_db.decision_threshold:.2f}")
        except Exception as e:
            _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] RAG 知识库未加载: "
                        f"path={rag_cache}, "
                        f"exists={os.path.exists(rag_cache) if rag_cache else 'N/A'}")

    num_streams = config_dict.get("num_streams", 4)
    per_frame_mem_mb = None
    safety_ratio = 0.60

    # ── 预热 + 测量每帧显存 ─────────────────────────────────────
    from PIL import Image as _PIL
    _warm_img = _PIL.new("RGB", (2048, 2048))
    torch.cuda.reset_peak_memory_stats()
    _ = inferencer.infer(_warm_img)
    torch.cuda.synchronize()
    _base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _ = inferencer.infer(_warm_img)
    torch.cuda.synchronize()
    per_frame_mem_mb = (torch.cuda.max_memory_allocated() - _base) / 1024**2
    # _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] per-frame memory ≈ {per_frame_mem_mb:.0f} MB")
    del _warm_img

    # ── 分区分配回调 ────────────────────────────────────────────
    assigned_partitions = []
    need_rejoin = False

    def on_assign(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        assigned_partitions[:] = p_ids
        _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] ✅ 分配到分区: {p_ids}")
        if not p_ids:
            _worker_logger.warning(f"[GPU-{gpu_id}-{w_idx}] ⚠️ 分配到空分区, 将重新加入消费组")
            nonlocal need_rejoin
            need_rejoin = True

    def on_revoke(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] ❌ 被回收分区: {p_ids}")
        assigned_partitions[:] = []

    def on_lost(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        _worker_logger.warning(f"[GPU-{gpu_id}-{w_idx}] ⚠️ 丢失分区: {p_ids}")
        assigned_partitions[:] = []
        nonlocal need_rejoin
        need_rejoin = True

    def _create_kafka_consumer():
        c = Consumer({
            "bootstrap.servers": config_dict["kafka_bootstrap_servers"],
            "group.id": config_dict["kafka_consumer_group"],
            "group.instance.id": f"gpu-{gpu_id}-worker-{w_idx}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": config_dict.get("kafka_max_poll_interval_ms", 1800000),
            "session.timeout.ms": config_dict.get("kafka_session_timeout_ms", 60000),
            "heartbeat.interval.ms": config_dict.get("kafka_heartbeat_interval_ms", 3000),
            "partition.assignment.strategy": "cooperative-sticky",
            "broker.address.family": "v4",
            "security.protocol": "PLAINTEXT",
            "fetch.min.bytes": 1,
            "queued.min.messages": 1,
        })
        c.subscribe(
            [config_dict["kafka_task_topic"]],
            on_assign=on_assign,
            on_revoke=on_revoke,
            on_lost=on_lost,
        )
        return c

    # ── 初始化 Kafka Consumer ────────────────────────────────────
    kafka_consumer = _create_kafka_consumer()
    _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] Kafka Consumer 已连接: topic={config_dict['kafka_task_topic']}, instance=gpu-{gpu_id}-worker-{w_idx}")

    # ── 初始化 Kafka Producer ────────────────────────────────────
    kafka_producer = Producer({
        "bootstrap.servers": config_dict["kafka_bootstrap_servers"],
        "broker.address.family": "v4",
        "security.protocol": "PLAINTEXT",
        "compression.type": "lz4",
        "message.max.bytes": 104857600,
        "queue.buffering.max.messages": 100000,
    })
    _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] Kafka Producer 已连接")

    shutdown_event = mp.Event()

    def handle_worker_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] 收到停止信号 ({signal_name})")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_worker_signal)
    signal.signal(signal.SIGTERM, handle_worker_signal)

    while not shutdown_event.is_set():
        try:
            msg = kafka_consumer.poll(timeout=0.5)
        except Exception as e:
            _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] Kafka poll 异常: {e}")
            time.sleep(1)
            continue

        if msg is None:
            if need_rejoin:
                _worker_logger.warning(f"[GPU-{gpu_id}-{w_idx}] 执行重新加入消费组...")
                kafka_consumer.close()
                kafka_consumer = _create_kafka_consumer()
                need_rejoin = False
            time.sleep(0.05)
            continue

        if msg.error():
            _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] Kafka poll error: {msg.error()}")
            continue

        task_data = json.loads(msg.value().decode('utf-8'))
        task_id = task_data.get("task_id", "")
        _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] 收到任务: {task_id}")

        # 立即 commit offset，避免长时间处理过程中崩溃/重平衡导致重复消费
        kafka_consumer.commit(asynchronous=False)
        conf_threshold = task_data.get("conf_threshold", 0.5)
        save_output = task_data.get("save_output", False)

        _free, _total = torch.cuda.mem_get_info(device)
        if _free < 8 * 1024**3:
            _worker_logger.warning(f"[GPU-{gpu_id}-{w_idx}] task {task_id} 显存不足 "
                         f"(free={_free/1024**3:.1f}GB < 8GB), "
                         f"该 GPU 标记死亡，进程退出...")
            kafka_consumer.close()
            kafka_producer.close()
            os._exit(1)

        try:
            video_path = task_data.get("video_path", "")
            if not video_path or not Path(video_path).exists():
                _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: 视频文件不存在 {video_path}")
                kafka_consumer.commit(asynchronous=True)
                continue
            sys.path.insert(0, config_dict["project_root"])
            from video_utils import read_video_frames

            CHUNK_SIZE = 200

            _t_read = time.perf_counter()
            frames_np, total_frames, total_chunks = read_video_frames(
                video_path, chunk_size=CHUNK_SIZE, chunk_index=0
            )
            n = len(frames_np)
            _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: "
                        f"读取 chunk 0/{total_chunks}, {n} 帧, "
                        f"总计 {total_frames} 帧, "
                        f"耗时: {(time.perf_counter() - _t_read) * 1000:.0f}ms")

            _free, _total = torch.cuda.mem_get_info(device)
            # 双缓冲流水线：GPU 上最多同时有 1 个推理中 batch + 2 个预取 batch，
            # 因此有效显存预算除以流水线深度
            pipeline_depth = 3
            _configured_batch = config_dict.get("infer_batch_size", 8)
            _max_batch = max(
                1, int(_free * (safety_ratio / pipeline_depth) / (per_frame_mem_mb * 1024 ** 2))
            )
            infer_batch_size = min(_configured_batch, _max_batch, 8)
            _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: "
                        f"free={_free/1024**3:.1f}GB, 推理批大小={infer_batch_size} "
                        f"(每帧约 {per_frame_mem_mb:.1f}MB, 流水线深度 {pipeline_depth})")

            import queue as _queue
            import threading as _threading

            frame_data: list[dict] = [None] * total_frames
            _t_infer = time.perf_counter()

            for chunk_idx in range(total_chunks):
                if shutdown_event.is_set():
                    _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: 收到停止信号，退出推理")
                    break

                _t_chunk = time.perf_counter()
                frames_chunk, _, _ = read_video_frames(
                    video_path, chunk_size=CHUNK_SIZE, chunk_index=chunk_idx
                )
                chunk_start = chunk_idx * CHUNK_SIZE
                chunk_frames = len(frames_chunk)

                # 零拷贝 numpy (N,H,W,3) → tensor
                if not frames_chunk.flags.writeable:
                    frames_chunk = frames_chunk.copy()
                tensor_nhwc = torch.from_numpy(frames_chunk)
                orig_size = (frames_chunk.shape[2], frames_chunk.shape[1])  # (W, H)

                # ══════════════════════════════════════════════════════════
                # 双缓冲流水线：预取线程负责 H2D，主线程负责批量推理
                #   ┌──────────┐   H2D(下一批)   ┌──────────┐
                #   │ 预取线程  │ ───────────────▶ │  主线程   │
                #   │ (传输)    │ ◀─────────────── │ (推理)    │
                #   └──────────┘   队列(maxsize=2) └──────────┘
                # 推理 batch N 期间，预取线程已在传输 batch N+1
                # ══════════════════════════════════════════════════════════
                h2d_stream = torch.cuda.Stream(device=device)
                prefetch_queue = _queue.Queue(maxsize=2)  # 双缓冲

                def _prefetch_batches(_tensor, _total, _bs, _q, _h2d_stream):
                    try:
                        for _b_start in range(0, _total, _bs):
                            if shutdown_event.is_set():
                                break
                            _b_end = min(_b_start + _bs, _total)
                            _batch_nhwc = _tensor[_b_start:_b_end]
                            # H2D + 维度转换 (N,H,W,3) -> (N,3,H,W)，独立 stream 异步传输
                            with torch.cuda.stream(_h2d_stream):
                                _batch_gpu = (
                                    _batch_nhwc.permute(0, 3, 1, 2)
                                    .contiguous()
                                    .cuda(device, non_blocking=True)
                                )
                            # 放入就绪队列（主线程推理完上一批后立即取走）
                            _q.put((_b_start, _batch_gpu))
                    except Exception as _e:
                        _q.put(("ERROR", _e))
                    finally:
                        _q.put(None)

                prefetch_thread = _threading.Thread(
                    target=_prefetch_batches,
                    args=(tensor_nhwc, chunk_frames, infer_batch_size,
                          prefetch_queue, h2d_stream),
                    daemon=True,
                    name=f"prefetch-gpu{gpu_id}-{w_idx}-chunk{chunk_idx}",
                )
                prefetch_thread.start()

                # 主循环：从队列取已传输好的 batch 推理
                while True:
                    if shutdown_event.is_set():
                        break
                    item = prefetch_queue.get()
                    if item is None:
                        break  # 预取线程全部完成
                    if isinstance(item, tuple) and item[0] == "ERROR":
                        raise item[1]

                    batch_start, batch_gpu = item
                    batch_frames = min(infer_batch_size, chunk_frames - batch_start)

                    # 等待本批 H2D 完成（通常早已就绪，几乎不阻塞）
                    h2d_stream.synchronize()

                    # 批量推理：GPU 变换 + model forward 一次处理 N 帧
                    batch_orig_sizes = [orig_size] * batch_frames
                    batch_results = inferencer.infer_batch(batch_gpu, batch_orig_sizes)

                    # 分发结果到 frame_data 对应位置
                    for j in range(batch_frames):
                        frame_idx = chunk_start + batch_start + j
                        frame_data[frame_idx] = {
                            "image": Image.fromarray(frames_chunk[batch_start + j]),
                            "results": batch_results[j],
                            "frame_name": f"{frame_idx:06d}.jpg",
                        }

                    # 清理本批次 GPU buffer（预取线程已开始下一批传输）
                    del batch_gpu

                # 等待预取线程退出，确保资源清理
                prefetch_thread.join(timeout=5)
                h2d_stream.synchronize()
                del h2d_stream, prefetch_queue, tensor_nhwc

                torch.cuda.synchronize()
                del frames_chunk
                _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: "
                            f"chunk {chunk_idx}/{total_chunks} 推理完成, "
                            f"耗时: {(time.perf_counter() - _t_chunk) * 1000:.0f}ms")

            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - _t_infer) * 1000
            _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] 任务 {task_id}: {total_frames} frames "
                        f"推理完成, 耗时: {infer_ms:.0f}ms, "
                        f"平均: {infer_ms / max(total_frames, 1):.1f}ms/帧")

            try:
                import supervision as sv
                from trackers import CBIoUTracker
                tracker = CBIoUTracker()
                _t0 = time.perf_counter()
                for fd in frame_data:
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
                _t3 = time.perf_counter()
                _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] 跟踪总耗时: {(_t3-_t0)*1000:.1f}ms / {len(frame_data)}帧")
            except Exception as e:
                _worker_logger.warning(f"[GPU-{gpu_id}-{w_idx}] 跟踪失败（跳过）: {e}")

            from streaming_cluster_updater import process_video_incremental
            output_parent = Path(config_dict["output_dir"]) / "temp" / task_id
            summary = process_video_incremental(
                frame_data=frame_data,
                output_root=output_parent,
                draw_func=draw,
                updater=updater,
                gpu_id=gpu_id,
                save_output=save_output,
            )

            # 发送完成结果
            result_data = {
                "task_id": task_id,
                "status": "completed",
                "summary": {
                    "output_dir": summary.get("output_dir", ""),
                    "video_path": video_path,
                },
            }

            result_bytes = json.dumps(result_data, ensure_ascii=False).encode('utf-8')
            msg_size_kb = len(result_bytes) / 1024
            _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: 结果消息大小={msg_size_kb:.1f}KB")

            send_success = False
            for attempt in range(3):
                try:
                    kafka_producer.produce(
                        config_dict["kafka_result_topic"],
                        key=task_id.encode('utf-8'),
                        value=result_bytes,
                    )
                    kafka_producer.flush(timeout=30)
                    _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: Kafka 结果发送成功 (尝试 {attempt+1})")
                    send_success = True
                    break
                except Exception as e:
                    _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: Kafka 发送失败 (尝试 {attempt+1}/3): {e}")
                    time.sleep(2 ** attempt)

            if not send_success:
                _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: Kafka 发送失败，已达最大重试次数，保存结果到本地")
                local_result_path = Path(config_dict["output_dir"]) / "temp" / task_id / "result.json"
                local_result_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: 结果已保存到本地: {local_result_path}")
                continue

            _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] 任务 {task_id} 完成, "
                        f"{summary['processed_frames']} frames, "
                        f"{summary['total_detections']} detections, "
                        f"keep={summary.get('frames_kept', 0)}, "
                        f"warn={summary.get('frames_warned', 0)}")

        except torch.cuda.OutOfMemoryError as e:
            _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] 任务 {task_id} OOM: {e}, "
                         f"该 GPU 标记死亡，进程退出...")
            _free_now, _total = torch.cuda.mem_get_info(device)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            kafka_consumer.close()
            kafka_producer.close()
            os._exit(1)

        except Exception as e:
            _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] 任务 {task_id} 失败: {e}")
            # 发送失败消息到 final_videos，由验证消费者重新路由
            try:
                fail_msg = {
                    "task_id": task_id,
                    "video_path": task_data.get("video_path", ""),
                    "error": str(e),
                    "failed_at": "api_service",
                    "status": "failed",
                }
                kafka_producer.produce(
                    config.KAFKA_FINAL_TOPIC,
                    key=task_id.encode('utf-8'),
                    value=json.dumps(fail_msg).encode('utf-8'),
                )
                kafka_producer.flush(timeout=10)
                _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: 失败消息已发送到 {config.KAFKA_FINAL_TOPIC}")
            except Exception as pe:
                _worker_logger.error(f"[GPU-{gpu_id}-{w_idx}] task {task_id}: 发送失败消息到 final_videos 失败: {pe}")

        finally:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    kafka_consumer.close()
    kafka_producer.close()
    _worker_logger.info(f"[GPU-{gpu_id}-{w_idx}] worker 退出")


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("   EdgeCrafter Kafka 推理服务")
    print("=" * 60)
    print(f"📂 项目路径: {config.PROJECT_ROOT}")
    print(f"📂 模型路径: {config.CHECKPOINT_PATH}")
    print(f"📂 输出路径: {config.OUTPUT_DIR}")
    print(f"🌐 Kafka: {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"📤 任务 Topic: {config.KAFKA_TASK_TOPIC}")
    print(f"📥 结果 Topic: {config.KAFKA_RESULT_TOPIC}")
    print(f"👥 消费者组: {config.KAFKA_CONSUMER_GROUP}")
    print("=" * 60)

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
    }

    _result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    _gpu_free = {}
    for line in _result.stdout.strip().split("\n"):
        parts = line.split(", ")
        if len(parts) == 2:
            _gpu_free[int(parts[0])] = int(parts[1])

    # 计算总消费者数，确保 topic 分区足够
    total_workers = 0
    gpu_worker_counts = {}
    for gpu_id in config.GPU_IDS:
        free_mb = _gpu_free.get(gpu_id, 0)

        # 空闲显存不超过 11GB 的 GPU 跳过，不创建进程
        if free_mb <= 11 * 1024:
            logger.info(f"⏭️ GPU {gpu_id}: 空闲显存 {free_mb}MB (≤11GB)，跳过")
            continue

        n_workers = max(1, min(config.WORKERS_PER_GPU, free_mb // 10240))
        gpu_worker_counts[gpu_id] = n_workers
        total_workers += n_workers
        logger.info(f"🔍 GPU {gpu_id}: 空闲 {free_mb}MB → {n_workers} 个 worker")

    # 初始化 topic + 确保分区足够
    init_topics()
    ensure_kafka_partitions(total_workers)

    for gpu_id, n_workers in gpu_worker_counts.items():
        for w_idx in range(n_workers):
            p = mp_ctx.Process(
                target=gpu_worker_process,
                args=(gpu_id, w_idx, cfg_dict),
                daemon=True,
                name=f"gpu-{gpu_id}-worker-{w_idx}",
            )
            p.start()
            gpu_processes.append(p)
            logger.info(f"🚀 GPU {gpu_id} worker-{w_idx} 已启动 (PID={p.pid})")

    logger.info(f"🚀 已启动 {len(gpu_processes)} 个 GPU worker")
    logger.info("✅ 服务启动成功！开始监控 Kafka topic...")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)

    shutdown_event = mp.Event()

    def handle_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.info(f"🛑 收到停止信号 ({signal_name})，正在关闭服务...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not shutdown_event.is_set():
        time.sleep(1)
        alive_workers = [p for p in gpu_processes if p.is_alive()]
        if not alive_workers:
            logger.critical("🚨 所有 GPU worker 均已退出！")
            break

    logger.info(f"📋 开始停止 {len(gpu_processes)} 个 GPU worker...")
    terminated_count = 0
    killed_count = 0

    for p in gpu_processes:
        if p.is_alive():
            logger.info(f"⏹️ 发送终止信号给进程 {p.pid} ({p.name})...")
            p.terminate()
            terminated_count += 1

    logger.info(f"⏳ 等待 {terminated_count} 个进程退出 (最多3秒)...")
    for p in gpu_processes:
        if p.is_alive():
            p.join(timeout=3)

    for p in gpu_processes:
        if p.is_alive():
            logger.warning(f"⚠️ 进程 {p.pid} ({p.name}) 超时未退出，强制终止...")
            p.kill()
            p.join(timeout=1)
            killed_count += 1

    logger.info(f"✅ 已停止 {terminated_count} 个进程，强制杀死 {killed_count} 个进程")
    logger.info("🛑 服务已关闭")


if __name__ == "__main__":
    main()
