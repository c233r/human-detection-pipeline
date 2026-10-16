import os
import json
import glob
import time
import math
import signal
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
from confluent_kafka import Consumer

INPUT_DIR = "/data2/caimoyao/api_outputs/temp"
OUTPUT_DIR = "/data2/caimoyao/api_outputs/hold_rope_results_01"

POSITIVE_THRESHOLD = 0.9
NEGATIVE_THRESHOLD = 0.2
MAX_DEPTH = 10
MIN_FRAMES = 5
KEYPOINT_SCORE_THRESHOLD = 3.0
MIN_KEYPOINTS_PASS = 6

KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
KAFKA_CONSUME_TOPIC = "final_videos"
KAFKA_CONSUMER_GROUP = "process_hold_rope_consumers"

mp.set_start_method('spawn', force=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/process_hold_rope.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("process_hold_rope")


def entropy(pos_count, total):
    if total == 0:
        return 0.0
    p = pos_count / total
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def split_neutral(items, task_id, track_id, depth=0):
    pos_count = sum(1 for it in items if it["data"]["hold_rope"])
    ratio = pos_count / len(items)
    
    if ratio >= POSITIVE_THRESHOLD:
        track_dir = os.path.join(OUTPUT_DIR, "positive", task_id, str(track_id))
        os.makedirs(track_dir, exist_ok=True)
        for it in items:
            with open(os.path.join(track_dir, it["frame"]), "w") as f:
                json.dump([it["data"]], f, indent=2)
        return len(items), 0
    elif ratio < NEGATIVE_THRESHOLD:
        track_dir = os.path.join(OUTPUT_DIR, "negative", task_id, str(track_id))
        os.makedirs(track_dir, exist_ok=True)
        for it in items:
            with open(os.path.join(track_dir, it["frame"]), "w") as f:
                json.dump([it["data"]], f, indent=2)
        return 0, len(items)
    
    if depth >= MAX_DEPTH or len(items) <= MIN_FRAMES:
        category = "positive" if ratio >= 0.5 else "negative"
        track_dir = os.path.join(OUTPUT_DIR, category, task_id, str(track_id))
        os.makedirs(track_dir, exist_ok=True)
        for it in items:
            with open(os.path.join(track_dir, it["frame"]), "w") as f:
                json.dump([it["data"]], f, indent=2)
        return (len(items), 0) if category == "positive" else (0, len(items))
    
    parent_entropy = entropy(pos_count, len(items))
    best_gain = -1.0
    best_split = len(items) // 2
    
    for i in range(1, len(items)):
        left_pos = sum(1 for it in items[:i] if it["data"]["hold_rope"])
        right_pos = pos_count - left_pos
        
        left_entropy = entropy(left_pos, i)
        right_entropy = entropy(right_pos, len(items) - i)
        
        weighted_entropy = (i / len(items)) * left_entropy + ((len(items) - i) / len(items)) * right_entropy
        info_gain = parent_entropy - weighted_entropy
        
        if info_gain > best_gain:
            best_gain = info_gain
            best_split = i
    
    pos_l, neg_l = split_neutral(items[:best_split], task_id, track_id, depth + 1)
    pos_r, neg_r = split_neutral(items[best_split:], task_id, track_id, depth + 1)
    return pos_l + pos_r, neg_l + neg_r


def filter_by_keypoints(item):
    keypoints = item.get("keypoints", [])
    low_conf_count = 0
    for kp in keypoints:
        if len(kp) >= 3 and kp[2] < KEYPOINT_SCORE_THRESHOLD:
            low_conf_count += 1
    return low_conf_count < MIN_KEYPOINTS_PASS


MIN_BOX_SIZE = 150

def filter_by_box_size(item):
    box = item.get("box", [0, 0, 0, 0])
    if len(box) >= 4:
        x1, y1, x2, y2 = map(float, box)
        orig_w = x2 - x1
        orig_h = y2 - y1
        if orig_h < MIN_BOX_SIZE or orig_w < MIN_BOX_SIZE:
            return False
    return True


def process_video_folder(task_id):
    video_folder = os.path.join(INPUT_DIR, task_id)
    labels_dir = os.path.join(video_folder, "labels")
    
    if not os.path.exists(labels_dir):
        return None, None

    track_stats = defaultdict(lambda: {"total": 0, "positive": 0})
    track_data = defaultdict(list)
    
    json_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))

    for json_path in json_files:
        frame_name = os.path.basename(json_path)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                for item in data:
                    track_id = item.get("track_id")
                    if track_id == -1 or track_id is None:
                        continue
                    if "hold_rope" in item:
                        if not filter_by_box_size(item):
                            continue
                        if not filter_by_keypoints(item):
                            continue
                        track_stats[track_id]["total"] += 1
                        if item["hold_rope"]:
                            track_stats[track_id]["positive"] += 1
                        track_data[track_id].append({
                            "frame": frame_name,
                            "data": item
                        })
        except:
            continue

    if not track_stats:
        return None, None

    results = []
    track_categories = {}
    
    for track_id, stats in track_stats.items():
        if stats["total"] > 0:
            ratio = stats["positive"] / stats["total"]
            if ratio >= POSITIVE_THRESHOLD:
                category = "positive"
            elif ratio < NEGATIVE_THRESHOLD:
                category = "negative"
            else:
                category = "neutral"
            track_categories[track_id] = category
            results.append({
                "video_name": task_id,
                "track_id": track_id,
                "ratio": ratio,
                "category": category
            })

    split_track_counts = {"positive": 0, "negative": 0}
    
    for track_id, items in track_data.items():
        if track_id not in track_categories:
            continue
        category = track_categories[track_id]
        
        if category == "positive":
            track_dir = os.path.join(OUTPUT_DIR, "positive", task_id, str(track_id))
            os.makedirs(track_dir, exist_ok=True)
            for item in items:
                output_path = os.path.join(track_dir, item["frame"])
                with open(output_path, "w") as f:
                    json.dump([item["data"]], f, indent=2)
            split_track_counts["positive"] += len(items)
            
        elif category == "negative":
            track_dir = os.path.join(OUTPUT_DIR, "negative", task_id, str(track_id))
            os.makedirs(track_dir, exist_ok=True)
            for item in items:
                output_path = os.path.join(track_dir, item["frame"])
                with open(output_path, "w") as f:
                    json.dump([item["data"]], f, indent=2)
            split_track_counts["negative"] += len(items)
            
        else:
            pos_frames, neg_frames = split_neutral(items, task_id, track_id)
            split_track_counts["positive"] += pos_frames
            split_track_counts["negative"] += neg_frames

    return results, split_track_counts


def worker_process(worker_idx):
    worker_logger = logging.getLogger(f"process_hold_rope.worker_{worker_idx}")
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False
    fh = logging.FileHandler('/data2/caimoyao/logs/process_hold_rope.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    worker_logger.addHandler(fh)

    kafka_consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_CONSUMER_GROUP,
        "group.instance.id": f"process-holdrope-worker-{worker_idx}",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.poll.interval.ms": 1800000,
        "session.timeout.ms": 180000,
        "heartbeat.interval.ms": 3000,
        "broker.address.family": "v4",
    })
    kafka_consumer.subscribe([KAFKA_CONSUME_TOPIC])
    worker_logger.info(f"[Worker-{worker_idx}] Kafka Consumer 已连接")

    shutdown_event = mp.Event()

    def handle_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        worker_logger.info(f"[Worker-{worker_idx}] 收到停止信号 ({signal_name})")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not shutdown_event.is_set():
        try:
            msg = kafka_consumer.poll(timeout=1.0)
        except Exception as e:
            worker_logger.error(f"[Worker-{worker_idx}] Kafka poll 异常: {e}")
            time.sleep(1)
            continue

        if msg is None:
            continue

        if msg.error():
            worker_logger.error(f"[Worker-{worker_idx}] Kafka poll error: {msg.error()}")
            continue

        try:
            task_data = json.loads(msg.value().decode('utf-8'))
        except Exception as e:
            worker_logger.error(f"[Worker-{worker_idx}] 消息解析失败: {e}")
            continue

        task_id = task_data.get("task_id", "")
        status = task_data.get("status", "")

        if status != "completed":
            worker_logger.debug(f"[Worker-{worker_idx}] 跳过非 completed 消息: {task_id}")
            kafka_consumer.commit()
            continue

        worker_logger.info(f"[Worker-{worker_idx}] 收到任务: {task_id}")

        try:
            results, split_counts = process_video_folder(task_id)
            if results:
                worker_logger.info(f"[Worker-{worker_idx}] 任务 {task_id} 处理完成, 正帧: {split_counts['positive']}, 负帧: {split_counts['negative']}")
            else:
                worker_logger.warning(f"[Worker-{worker_idx}] 任务 {task_id}: 无有效数据")
        except Exception as e:
            worker_logger.error(f"[Worker-{worker_idx}] 任务 {task_id} 处理失败: {e}")
            continue

        for commit_attempt in range(3):
            try:
                kafka_consumer.commit(asynchronous=False)
                worker_logger.info(f"[Worker-{worker_idx}] task {task_id}: offset 提交成功")
                break
            except Exception as e:
                worker_logger.error(f"[Worker-{worker_idx}] task {task_id}: offset 提交失败 ({commit_attempt + 1}/3): {e}")
                time.sleep(2)
        else:
            worker_logger.error(f"[Worker-{worker_idx}] task {task_id}: offset 提交失败，可能会被重复消费")

        worker_logger.info(f"[Worker-{worker_idx}] 任务完成: {task_id}")

    kafka_consumer.close()
    worker_logger.info(f"[Worker-{worker_idx}] 已关闭")


def ensure_kafka_partitions(total_consumers):
    from confluent_kafka.admin import AdminClient, NewPartitions
    topic = KAFKA_CONSUME_TOPIC
    admin = AdminClient({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
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

    logger.info(f"Topic {topic} 当前分区数: {current_partitions}, 消费者数: {total_consumers}")

    if current_partitions >= total_consumers:
        return

    logger.info(f"分区数不足，扩展至 {total_consumers} 个分区")
    fs = admin.create_partitions([NewPartitions(topic, total_consumers)])
    fs[topic].result(timeout=10)
    logger.info(f"✅ Topic {topic} 分区已扩展至 {total_consumers}")


def main():
    print("=" * 60)
    print("   Process HoldRope 分类结果处理服务 (Kafka)")
    print("=" * 60)
    print(f"🌐 Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"📥 消费: {KAFKA_CONSUME_TOPIC}")
    print(f"👥 消费者组: {KAFKA_CONSUMER_GROUP}")
    print(f"📂 输入目录: {INPUT_DIR}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(os.path.join(OUTPUT_DIR, "positive"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "negative"), exist_ok=True)

    NUM_WORKERS = 8
    ensure_kafka_partitions(NUM_WORKERS)

    mp_ctx = mp.get_context("spawn")
    processes = []

    for w_idx in range(NUM_WORKERS):
        p = mp_ctx.Process(
            target=worker_process,
            args=(w_idx,),
            daemon=True,
            name=f"process-holdrope-worker-{w_idx}",
        )
        p.start()
        processes.append(p)
        logger.info(f"🚀 Worker-{w_idx} 已启动 (PID={p.pid})")

    logger.info(f"✅ 已启动 {len(processes)} 个 Worker")
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
            logger.critical("🚨 所有 Worker 均已退出！")
            break

    logger.info(f"📋 开始停止 {len(processes)} 个 Worker...")
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
