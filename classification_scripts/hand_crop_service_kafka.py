"""
HandCrop 手部裁剪分类服务（最终阶段）
基于 Kafka 消息队列，消费 hold_rope_keypoints topic 并执行手部裁剪分类
结果直接写入 label 文件，不再产出新 topic
"""

import os
import sys
import json
import time
import signal
import math
import cv2
import numpy as np
import multiprocessing as mp
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic, NewPartitions

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import torch

from utils.inference import inference_model, init_model
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from core.datasets.compose import Compose

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data2/caimoyao/logs/hand_crop.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("hand_crop_classify")


# ==================== 配置 ====================
class Config:
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
    KAFKA_CONSUME_TOPIC = "hold_rope_keypoints"
    KAFKA_PRODUCE_TOPIC = "final_videos"
    KAFKA_CONSUMER_GROUP = "hand_crop_consumers"

    TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")

    # 分类模型配置
    CLASSES_MAP = '/data2/caimoyao/Jump_rope/classification/datas/annotations.txt'
    CONFIG_PATH = '/data2/caimoyao/Jump_rope/classification/logs/ConvNeXt/2026-07-16-11-48-44/convnext_base_fix.py'
    GPU_IDS = [1, 5]
    CONFIDENCE_THRESHOLD = 0.8

    # 手部裁剪参数
    HAND_SHOULDER_RATIO = 0.8
    MIN_HAND_AREA = 1800
    KEYPOINT_SCORE_THRESHOLD = 0.3
    BATCH_CHUNK_SIZE = 100


config = Config()


# ==================== Kafka 初始化 ====================
def init_topics():
    """确保所需的 topic 存在。"""
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


# ==================== GPU Worker ====================
def gpu_worker(gpu_id: int, worker_idx: int):
    """每个 GPU worker 独立消费 Kafka 并执行手部裁剪分类。"""
    worker_logger = logging.getLogger(f"hand_crop.gpu_{gpu_id}_{worker_idx}")
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False
    fh = logging.FileHandler('/data2/caimoyao/logs/hand_crop.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    worker_logger.addHandler(fh)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    # 加载模型
    worker_logger.info(f"[GPU-{gpu_id}] 开始加载分类模型...")
    
    t0 = time.time()
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 1/5] 加载类别信息: {config.CLASSES_MAP}")
    classes_names, label_names = get_info(config.CLASSES_MAP)
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 1/5] 类别信息加载完成: {len(classes_names)} 个类别, 耗时 {time.time()-t0:.2f}s")
    
    # t1 = time.time()
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 2/5] 加载模型配置: {config.CONFIG_PATH}")
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config.CONFIG_PATH)
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 2/5] 模型配置加载完成, 耗时 {time.time()-t1:.2f}s")
    
    # t2 = time.time()
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 3/5] 构建网络模型...")
    net = BuildNet(model_cfg)
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 3/5] 网络模型构建完成, 参数数量: {sum(p.numel() for p in net.parameters()):,}, 耗时 {time.time()-t2:.2f}s")
    
    # t3 = time.time()
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 4/5] 加载权重文件: {data_cfg.get('test', {}).get('ckpt', 'unknown')}")
    model = init_model(net, data_cfg, device=device, mode='eval')
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 4/5] 权重加载完成, 耗时 {time.time()-t3:.2f}s")
    
    # t4 = time.time()
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 5/5] 模型移动到 GPU {gpu_id}...")
    model.to(device)
    torch.cuda.synchronize()
    # worker_logger.info(f"[GPU-{gpu_id}] [Step 5/5] 模型移动完成, 显存占用: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB, 耗时 {time.time()-t4:.2f}s")
    
    worker_logger.info(f"[GPU-{gpu_id}] 分类模型加载完成, 总耗时 {time.time()-t0:.2f}s")

    assigned_partitions = []
    need_rejoin = False

    def on_assign(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        assigned_partitions[:] = p_ids
        if p_ids:
            worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] ✅ 分配到分区: {p_ids}")
        else:
            worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] ⏳ 暂未分配到分区 (worker 多于分区数，等待中...)")

    def on_revoke(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] ❌ 被回收分区: {p_ids}")
        assigned_partitions[:] = []

    def on_lost(consumer, partitions):
        p_ids = [p.partition for p in partitions]
        worker_logger.warning(f"[GPU-{gpu_id}-{worker_idx}] ⚠️ 丢失分区: {p_ids}")
        assigned_partitions[:] = []
        nonlocal need_rejoin
        need_rejoin = True

    def _create_kafka_consumer():
        c = Consumer({
            "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": config.KAFKA_CONSUMER_GROUP,
            "group.instance.id": f"handcrop-gpu-{gpu_id}-worker-{worker_idx}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": 14400000,
            "session.timeout.ms": 100000,
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

    kafka_consumer = _create_kafka_consumer()
    worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] Kafka Consumer 已连接: {config.KAFKA_CONSUME_TOPIC}, instance=handcrop-gpu-{gpu_id}-worker-{worker_idx}")

    # ── 初始化 Kafka Producer（产出 final_videos） ──────────────
    kafka_producer = Producer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
    })
    worker_logger.info(f"[GPU-{gpu_id}-{worker_idx}] Kafka Producer 已连接: {config.KAFKA_PRODUCE_TOPIC}")

    shutdown_event = mp.Event()

    def handle_worker_signal(signum, frame):
        signal_name = signal.Signals(signum).name
        worker_logger.info(f"[GPU-{gpu_id}] 收到停止信号 ({signal_name})")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_worker_signal)
    signal.signal(signal.SIGTERM, handle_worker_signal)

    while not shutdown_event.is_set():
        # 延迟 seek：分配到分区后，在主循环中再执行 seek，避免元数据未就绪

        try:
            msg = kafka_consumer.poll(timeout=0.5)
        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}] Kafka poll 异常: {e}")
            time.sleep(1)
            continue

        if msg is None:
            if need_rejoin:
                worker_logger.warning(f"[GPU-{gpu_id}-{worker_idx}] 执行重新加入消费组...")
                kafka_consumer.close()
                kafka_consumer = _create_kafka_consumer()
                need_rejoin = False
            time.sleep(0.05)
            continue

        if msg.error():
            worker_logger.error(f"[GPU-{gpu_id}] Kafka poll error: {msg.error()}")
            continue

        try:
            task_data = json.loads(msg.value().decode('utf-8'))
        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}] 消息解析失败: {e}")
            continue

        task_id = task_data.get("task_id", "")
        status = task_data.get("status", "")

        if status != "completed":
            worker_logger.debug(f"[GPU-{gpu_id}] 跳过非 completed 消息: {task_id}")
            kafka_consumer.commit() # 如果没有完成，那就重新加入到原队列中重新消费
            continue

        worker_logger.info(f"[GPU-{gpu_id}] 收到任务: {task_id}")

        # 从 msg 中获取路径
        output_labels = task_data.get("output_labels", "")
        video_path_str = task_data.get("video_path", "")
        if not output_labels:
            worker_logger.error(f"[GPU-{gpu_id}] 任务 {task_id}: output_labels 为空")
            continue

        try:
            process_task(task_id, output_labels, video_path_str, model, val_pipeline,
                         classes_names, label_names, gpu_id, worker_logger)
            worker_logger.info(f"[GPU-{gpu_id}] 任务 {task_id} 处理成功")
        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}] 任务 {task_id} 处理失败: {e}")
            continue

        # 发送结果到 final_videos
        try:
            result_msg = {
                "task_id": task_id,
                "video_path": video_path_str,
                "output_labels": output_labels,
                "status": "completed",
            }
            kafka_producer.produce(
                config.KAFKA_PRODUCE_TOPIC,
                key=task_id.encode('utf-8'),
                value=json.dumps(result_msg).encode('utf-8'),
            )
            kafka_producer.flush(timeout=10)
            worker_logger.info(f"[GPU-{gpu_id}] 结果已发送 → {config.KAFKA_PRODUCE_TOPIC}: {task_id}")
        except Exception as e:
            worker_logger.error(f"[GPU-{gpu_id}] 发送结果到 {config.KAFKA_PRODUCE_TOPIC} 失败: {e}")
            continue

        for commit_attempt in range(3):
            try:
                kafka_consumer.commit(asynchronous=False)
                worker_logger.info(f"[GPU-{gpu_id}] task {task_id}: offset 提交成功")
                break
            except Exception as e:
                worker_logger.error(f"[GPU-{gpu_id}] task {task_id}: offset 提交失败 ({commit_attempt + 1}/3): {e}")
                time.sleep(2)
        else:
            worker_logger.error(f"[GPU-{gpu_id}] task {task_id}: offset 提交失败，可能会被重复消费")

        worker_logger.info(f"[GPU-{gpu_id}] 任务完成: {task_id}")

    kafka_consumer.close()
    kafka_producer.close()


# ==================== 任务处理核心逻辑 ====================
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
    if ls[2] < config.KEYPOINT_SCORE_THRESHOLD or rs[2] < config.KEYPOINT_SCORE_THRESHOLD:
        return None
    shoulder_dist = math.sqrt((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2)
    if shoulder_dist < 1:
        return None

    def _crop(kp):
        hx, hy, hs = kp
        if hs < config.KEYPOINT_SCORE_THRESHOLD:
            return None
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


def _classify_hands_batch(model, val_pipeline, classes_names, label_names,
                          hand_images: list[np.ndarray], gpu_id: int, worker_logger):
    if not hand_images:
        return []

    all_results = []
    chunk_size = config.BATCH_CHUNK_SIZE
    n_chunks = (len(hand_images) + chunk_size - 1) // chunk_size
    _batch_t0 = time.perf_counter()

    for ci in range(n_chunks):
        chunk = hand_images[ci * chunk_size:(ci + 1) * chunk_size]
        device = next(model.parameters()).device
        tensors = []

        pipeline = Compose(
            [t for t in val_pipeline if t.get('type') != 'LoadImageFromFile']
        )

        for img in chunk:
            padded = _pad_square(img)
            data = dict(img=padded, filename=None)
            t = pipeline(data)['img']
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(device)

        with torch.no_grad():
            scores = model(batch, return_loss=False)
            pred_scores, pred_labels = torch.max(scores, axis=1)

        chunk_results = list(zip(pred_labels.cpu().tolist(), pred_scores.cpu().tolist()))
        all_results.extend(chunk_results)

    worker_logger.info(f"[GPU-{gpu_id}] batch 分类 {len(hand_images)} 张手图, "
                       f"{n_chunks} chunk, "
                       f"总耗时: {(time.perf_counter() - _batch_t0) * 1000:.0f}ms")
    return all_results


def process_task(task_id: str, output_labels: str, video_path_str: str,
                 model, val_pipeline, classes_names, label_names,
                 gpu_id: int, worker_logger):
    """处理单个任务：读取帧 → 裁剪手图 → 分类 → 更新标签。"""
    labels_dir = Path(output_labels)
    if not labels_dir.exists():
        worker_logger.warning(f"[GPU-{gpu_id}] 任务 {task_id}: labels 目录不存在")
        return {"task_id": task_id, "status": "failed", "error": "labels dir not found"}

    video_path = Path(video_path_str) if video_path_str else None
    if video_path and not video_path.exists():
        worker_logger.warning(f"[GPU-{gpu_id}] 任务 {task_id}: 视频文件不存在")
        return {"task_id": task_id, "status": "failed", "error": "video not found"}

    label_files = sorted(labels_dir.glob("*.json"))
    worker_logger.info(f"[GPU-{gpu_id}] 任务 {task_id}: {len(label_files)} 帧")
    _task_t0 = time.perf_counter()

    # 读取视频帧
    if video_path:
        from video_utils import read_video_frames
        frames_rgb, indices = read_video_frames(str(video_path), return_indices=True)
        n_total = indices[-1] + 1 if indices else 0
        frames_bgr = {}
        for i, idx in enumerate(indices):
            frames_bgr[idx] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        del frames_rgb
        all_indices = sorted({int(lf.stem) for lf in label_files if int(lf.stem) < n_total})
        _t_read = time.perf_counter()
        worker_logger.info(f"[GPU-{gpu_id}] 任务 {task_id}: 读取 {len(indices)} 帧, "
                           f"含标签{len(all_indices)}帧, "
                           f"耗时: {(_t_read - _task_t0) * 1000:.0f}ms")
    else:
        frames_bgr = {}

    # Phase 1（CPU 并行）：裁剪所有手图
    all_hands = []
    all_mappings = []
    frame_labels_map = {}
    total_persons = 0
    total_hold = 0

    def _proc_one(lf):
        fidx = int(lf.stem)
        if fidx not in frames_bgr:
            return None, False
        img = frames_bgr[fidx]
        try:
            with open(lf, 'r') as f:
                labs = json.load(f)
        except Exception:
            return None, False

        h, w = img.shape[:2]
        hands, mappings = [], []
        any_person = False
        for p in labs:
            plabel = p.get("label")
            if (int(plabel) if plabel is not None else -1) != 1:
                continue
            any_person = True
            kps = p.get("keypoints", [])
            if len(kps) < 25:
                continue
            ls_score, rs_score = kps[5][2], kps[6][2]
            if ls_score < config.KEYPOINT_SCORE_THRESHOLD or rs_score < config.KEYPOINT_SCORE_THRESHOLD:
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
                mappings.append((lf, p, "left"))
                hands.append(right)
                mappings.append((lf, p, "right"))
        return (lf, labs, hands, mappings), any_person

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_map = {pool.submit(_proc_one, lf): lf for lf in label_files}
        for fut in as_completed(fut_map):
            result, has_person = fut.result()
            if result is None:
                continue
            lf, labs, hands, mappings = result
            frame_labels_map[lf] = labs
            all_hands.extend(hands)
            all_mappings.extend(mappings)

    worker_logger.info(f"[GPU-{gpu_id}] 任务 {task_id}: 预处理完成, {len(all_hands)} 张手图")

    # Phase 2（GPU）：batch 分类所有手图
    if all_hands:
        batch_results = _classify_hands_batch(model, val_pipeline, classes_names,
                                              label_names, all_hands, gpu_id, worker_logger)
        for (label_file, person, side), (label, score) in zip(all_mappings, batch_results):
            if side == "left":
                person["hand_left_hold"] = (label == 0)
                person["hand_left_score"] = float(score)
            else:
                person["hand_right_hold"] = (label == 0)
                person["hand_right_score"] = float(score)

    # Phase 3（CPU）：写回 JSON
    for label_file, labels in frame_labels_map.items():
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

        with open(label_file, 'w') as f:
            json.dump(labels, f, indent=2)
        total_persons += frame_persons
        total_hold += frame_hold

    worker_logger.info(f"[GPU-{gpu_id}] 任务 {task_id}: {total_persons} 人体, {total_hold} 握绳, "
                       f"总耗时: {(time.perf_counter() - _task_t0) * 1000:.0f}ms")

    return {
        "task_id": task_id,
        "status": "completed",
        "total_persons": total_persons,
        "total_hold": total_hold,
        "video_path": video_path_str,
        "output_labels": output_labels,
    }


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("   HandCrop 手部裁剪分类服务 (Kafka)")
    print("=" * 60)
    print(f"🌐 Kafka: {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"📥 消费: {config.KAFKA_CONSUME_TOPIC}")
    print(f" 消费者组: {config.KAFKA_CONSUMER_GROUP}")
    print(f"🎮 GPU: {config.GPU_IDS}")
    print("=" * 60)

    init_topics()

    mp_ctx = mp.get_context("spawn")
    processes = []

    # 查显存，计算每个 GPU 开几个 worker
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

    WORKER_MEM_MB = 4 * 1024  # 每个 worker 约 4GB
    MAX_WORKERS_PER_GPU = 4
    gpu_worker_counts = {}
    for gpu_id in config.GPU_IDS:
        free_mb = _gpu_free.get(gpu_id, 0)
        n_workers = min(free_mb // WORKER_MEM_MB, MAX_WORKERS_PER_GPU)
        # n_workers = max(n_workers, 1)  # 至少1个
        gpu_worker_counts[gpu_id] = n_workers
        logger.info(f"🔍 GPU {gpu_id}: 空闲 {free_mb}MB → {n_workers} 个 worker (上限 {MAX_WORKERS_PER_GPU})")

    # 重新确保分区数足够
    total_workers = sum(gpu_worker_counts.values())
    ensure_kafka_partitions(total_workers)

    for gpu_id in config.GPU_IDS:
        n_workers = gpu_worker_counts.get(gpu_id, 1)
        for w_idx in range(n_workers):
            p = mp_ctx.Process(
                target=gpu_worker,
                args=(gpu_id, w_idx),
                daemon=True,
                name=f"handcrop-gpu-{gpu_id}-{w_idx}",
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
    mp.set_start_method('spawn', force=True)
    main()