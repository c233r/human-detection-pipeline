"""
HoldRope 关键点检测服务
基于 Kafka 消息队列，自动监控 video_results topic 并执行关键点检测

流程:
  video_tasks ──→ GPU推理(api_service) ──→ video_results ──→ 本服务(关键点) ──→ hold_rope_keypoints
"""

import os
import sys
import json
import time
import signal
from pathlib import Path
from typing import Dict, List
import multiprocessing as mp
import logging

from confluent_kafka import Consumer, Producer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic, NewPartitions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from video_utils import read_video_frames

mp.set_start_method('spawn', force=True)

from inference_tensorRT_multi_rtmpose_l import RTMPoseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/monitor_hold_rope.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("hold_rope_monitor_onnx")


# ==================== 配置 ====================
class Config:
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_CONSUME_TOPIC = "video_results"       # 消费：上游推理完成的结果
    KAFKA_PRODUCE_TOPIC = "hold_rope_keypoints"  # 产出：关键点检测结果
    KAFKA_CONSUMER_GROUP = "hold_rope_consumers"

    TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")

    GPU_IDS = [1, 5]

config = Config()


# ==================== 初始化 Topic ====================
def init_topics():
    """确保所需的 topic 存在。"""
    admin = AdminClient({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
    })
    topics_needed = [config.KAFKA_CONSUME_TOPIC, config.KAFKA_PRODUCE_TOPIC]
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
    """检查 consume topic 分区数，不足时扩展到指定数量。"""
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


# ==================== 关键点处理（核心逻辑） ====================
def process_task(task_data: Dict, model, gpu_id: int, worker_logger) -> Dict:
    """处理单个任务：读取帧 → 关键点检测 → 返回结果。"""
    task_id = task_data.get("task_id", "")
    if not task_id:
        return {"task_id": task_id, "status": "failed", "error": "missing task_id"}

    task_dir = config.TEMP_DIR / task_id
    labels_dir = task_dir / "labels"

    if not labels_dir.exists():
        worker_logger.warning(f"[GPU-{gpu_id}] task {task_id}: labels 目录不存在")
        return {"task_id": task_id, "status": "failed", "error": "labels dir not found"}

    # 从 msg 中获取视频路径
    video_path_str = task_data.get("summary", {}).get("video_path", "")
    if not video_path_str:
        worker_logger.warning(f"[GPU-{gpu_id}] task {task_id}: 未找到视频路径")
        return {"task_id": task_id, "status": "failed", "error": "video_path not found"}

    video_path = Path(video_path_str)
    if not video_path.exists():
        worker_logger.warning(f"[GPU-{gpu_id}] task {task_id}: 视频文件不存在 {video_path}")
        return {"task_id": task_id, "status": "failed", "error": "video not found"}

    try:
        frames_np = read_video_frames(str(video_path))
    except Exception as e:
        worker_logger.error(f"[GPU-{gpu_id}] task {task_id}: 读取视频失败 {e}")
        return {"task_id": task_id, "status": "failed", "error": str(e)}

    n_frames = len(frames_np)
    worker_logger.info(f"[GPU-{gpu_id}] task {task_id}: 读取 {n_frames} 帧")

    label_files = sorted(labels_dir.glob("*.json"))
    worker_logger.info(f"[GPU-{gpu_id}] task {task_id}: 找到 {len(label_files)} 个标签文件")

    has_error = False
    processed_count = 0

    for label_file in label_files:
        frame_idx = int(label_file.stem)
        if frame_idx >= n_frames:
            worker_logger.warning(f"[GPU-{gpu_id}] 帧索引 {frame_idx} 超出范围 ({n_frames})，跳过")
            continue

        frame_np = frames_np[frame_idx]
        import cv2
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

                cropped = frame_bgr[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                try:
                    keypoints = model.inference(cropped)
                    orig_kps = [
                        [float(kp['x']) + x1, float(kp['y']) + y1, float(kp['score'])]
                        for kp in keypoints
                    ]
                    label["keypoints"] = orig_kps
                except Exception as e:
                    worker_logger.error(f"[GPU-{gpu_id}] 关键点检测失败 {label_file}: {e}")
                    has_error = True

            with open(label_file, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
            processed_count += 1

        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}] 处理标签文件 {label_file} 失败: {e}")
            has_error = True

    worker_logger.info(f"[GPU-{gpu_id}] task {task_id} 处理完成, "
                       f"已处理标签文件: {processed_count}, 错误: {has_error}")

    return {
        "task_id": task_id,
        "status": "completed" if not has_error else "partial_error",
        "processed_labels": processed_count,
        "video_path": str(video_path),
        "output_labels": str(labels_dir),
    }


# ==================== GPU Worker ====================
def gpu_worker(gpu_id: int, worker_idx: int):
    """每个 GPU worker 独立消费 Kafka 并执行关键点检测。"""
    worker_logger = logging.getLogger(f"hold_rope_monitor_onnx.gpu_{gpu_id}_{worker_idx}")
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False
    fh = logging.FileHandler('/data2/caimoyao/logs/monitor_hold_rope_onnx.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    worker_logger.addHandler(fh)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] 加载模型...")
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "weights", "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
    )
    model = RTMPoseModel(model_path, device_id=0)
    worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] 模型加载完成")

    assigned_partitions = []
    need_rejoin = False

    def on_assign(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        assigned_partitions[:] = p_ids
        worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] ✅ 分配到分区: {p_ids}")
        if not p_ids:
            worker_logger.warning(f"[GPU-{gpu_id}-{worker_idx}] ⚠️ 分配到空分区, 将重新加入消费组")
            global need_rejoin
            need_rejoin = True

    def on_revoke(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] ❌ 被回收分区: {p_ids}")
        assigned_partitions[:] = []

    def on_lost(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        worker_logger.warning(f"[GPU-{gpu_id}-{worker_idx}] ⚠️ 丢失分区: {p_ids}")
        assigned_partitions[:] = []
        global need_rejoin
        need_rejoin = True

    def _create_consumer():
        c = Consumer({
            "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": config.KAFKA_CONSUMER_GROUP,
            "group.instance.id": f"monitor-gpu-{gpu_id}-worker-{worker_idx}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": 14400000,
            "session.timeout.ms": 180000,
            "heartbeat.interval.ms": 3000,
            "partition.assignment.strategy": "cooperative-sticky",
            "broker.address.family": "v4",
            "queued.min.messages": 1,
        })
        c.subscribe(
            [config.KAFKA_CONSUME_TOPIC],
            on_assign=on_assign,
            on_revoke=on_revoke,
            on_lost=on_lost,
        )
        return c

    consumer = _create_consumer()
    worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] Consumer 已连接: topic={config.KAFKA_CONSUME_TOPIC}, instance=monitor-gpu-{gpu_id}-worker-{worker_idx}")

    # Kafka Producer（产出 hold_rope_keypoints）
    producer = Producer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
    })
    worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] Producer 已连接")

    shutdown_event = mp.Event()

    def handle_worker_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] 收到停止信号 ({signal_name})")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_worker_signal)
    signal.signal(signal.SIGTERM, handle_worker_signal)

    while not shutdown_event.is_set():
        try:
            msg = consumer.poll(timeout=0.5)
        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}-{worker_idx}] poll 异常: {e}, 重建 consumer")
            consumer.close()
            consumer = _create_consumer()
            time.sleep(1)
            continue

        if msg is None:
            if need_rejoin:
                worker_logger.warning(f"[GPU-{gpu_id}-{worker_idx}] 执行重新加入消费组...")
                consumer.close()
                consumer = _create_consumer()
                need_rejoin = False
            time.sleep(0.05)
            continue

        if msg.error():
            err_code = msg.error().code()
            worker_logger.error(f"[GPU-{gpu_id}-{worker_idx}] poll error: {msg.error()}")
            if err_code == KafkaError._MAX_POLL_EXCEEDED:
                worker_logger.warning(f"[GPU-{gpu_id}-{worker_idx}] poll 超时被踢出组, 重建 consumer")
                consumer.close()
                consumer = _create_consumer()
            elif err_code == KafkaError._TIMED_OUT:
                time.sleep(1)
            else:
                time.sleep(1)
            continue

        try:
            task_data = json.loads(msg.value().decode('utf-8'))
        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}-{worker_idx}] 消息解析失败: {e}")
            continue

        task_id = task_data.get("task_id", "")
        status = task_data.get("status", "")

        if status != "completed":
            worker_logger.debug(f"[GPU-{gpu_id}-{worker_idx}] 跳过非 completed 消息: {task_id} (status={status})")
            consumer.commit()
            continue

        worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] 收到任务: {task_id}")

        result = process_task(task_data, model, gpu_id, worker_logger)

        try:
            producer.produce(
                config.KAFKA_PRODUCE_TOPIC,
                key=task_id.encode('utf-8'),
                value=json.dumps(result).encode('utf-8'),
            )
            producer.flush(timeout=10)
            worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] 结果已发送: {task_id} → {config.KAFKA_PRODUCE_TOPIC}")
        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}-{worker_idx}] 发送结果失败: {e}")

        if result.get("status") != "completed":
            worker_logger.warning(f"[GPU-{gpu_id}-{worker_idx}] task {task_id}: 处理失败 (status={result.get('status')}), 不提交 offset, 等待重试")
            continue

        for commit_attempt in range(3):
            try:
                consumer.commit(asynchronous=False)
                worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] task {task_id}: offset 提交成功")
                break
            except Exception as e:
                worker_logger.error(f"[GPU-{gpu_id}-{worker_idx}] task {task_id}: offset 提交失败 ({commit_attempt + 1}/3): {e}")
                time.sleep(2)
        else:
            worker_logger.error(f"[GPU-{gpu_id}-{worker_idx}] task {task_id}: offset 提交失败，可能会被重复消费")

        worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] 任务完成: {task_id}")

    consumer.close()
    producer.close()


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("   HoldRope 关键点检测服务 (Kafka)")
    print("=" * 60)
    print(f"🌐 Kafka: {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"📥 消费: {config.KAFKA_CONSUME_TOPIC}")
    print(f"📤 产出: {config.KAFKA_PRODUCE_TOPIC}")
    print(f"👥 消费者组: {config.KAFKA_CONSUMER_GROUP}")
    print(f"🎮 GPU: {config.GPU_IDS}")
    print("=" * 60)

    init_topics()

    import subprocess
    _result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    _gpu_free = {}
    for line in _result.stdout.strip().split("\n"):
        parts = line.split(", ")
        if len(parts) == 2:
            _gpu_free[int(parts[0])] = int(parts[1])

    WORKER_MEM_GB = 1.5
    WORKER_MEM_MB = int(WORKER_MEM_GB * 1024)
    MAX_WORKERS_PER_GPU = 4

    total_workers = 0
    gpu_worker_counts = {}
    for gpu_id in config.GPU_IDS:
        free_mb = _gpu_free.get(gpu_id, 0)
        if free_mb < WORKER_MEM_MB:
            logger.warning(f"⚠️ GPU {gpu_id}: 空闲 {free_mb}MB < {WORKER_MEM_GB}GB，跳过")
            continue
        n_workers = min(free_mb // WORKER_MEM_MB, MAX_WORKERS_PER_GPU)
        gpu_worker_counts[gpu_id] = n_workers
        total_workers += n_workers
        logger.info(f"🔍 GPU {gpu_id}: 空闲 {free_mb}MB → {n_workers} 个 worker (上限 {MAX_WORKERS_PER_GPU})")

    ensure_kafka_partitions(total_workers)

    mp_ctx = mp.get_context("spawn")
    processes = []

    for gpu_id, n_workers in gpu_worker_counts.items():
        for w_idx in range(n_workers):
            p = mp_ctx.Process(
                target=gpu_worker,
                args=(gpu_id, w_idx),
                daemon=True,
                name=f"holdrope-gpu-{gpu_id}-worker-{w_idx}",
            )
            p.start()
            processes.append(p)
            logger.info(f"🚀 GPU {gpu_id} worker-{w_idx} 已启动 (PID={p.pid})")

    logger.info(f"✅ 已启动 {len(processes)} 个 GPU worker")
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
        alive = [p for p in processes if p.is_alive()]
        if not alive:
            logger.critical("🚨 所有 GPU worker 均已退出！")
            break

    logger.info(f"📋 开始停止 {len(processes)} 个 GPU worker...")
    terminated_count = 0
    killed_count = 0

    for p in processes:
        if p.is_alive():
            logger.info(f"⏹️ 发送终止信号给进程 {p.pid} ({p.name})...")
            p.terminate()
            terminated_count += 1

    logger.info(f"⏳ 等待 {terminated_count} 个进程退出 (最多3秒)...")
    for p in processes:
        if p.is_alive():
            p.join(timeout=3)

    for p in processes:
        if p.is_alive():
            logger.warning(f"⚠️ 进程 {p.pid} ({p.name}) 超时未退出，强制终止...")
            p.kill()
            p.join(timeout=1)
            killed_count += 1

    logger.info(f"✅ 已停止 {terminated_count} 个进程，强制杀死 {killed_count} 个进程")
    logger.info("🛑 服务已关闭")


if __name__ == "__main__":
    main()
