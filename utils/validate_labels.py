"""
验证脚本：扫描 api_outputs/temp 下的所有 labels JSON 文件，
检查完整性，并与 final_videos topic 中的 task 交叉验证。

要求：
  - 每个 entry 必须有: label, score, box, track_id
  - label == 1 的 entry 还必须有: keypoints, hold_rope, hand_left_hold, hand_right_hold,
    hand_left_score, hand_right_score

用法：
  python validate_labels.py [--topic-only] [--repair] [--task TASK_ID]
"""

import os
import sys
import json
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from confluent_kafka import Consumer, Producer, TopicPartition


TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")
KAFKA_BOOTSTRAP = "localhost:9094"
FINAL_TOPIC = "final_videos"
VIDEO_TASKS_TOPIC = "video_tasks"
VIDEO_RESULTS_TOPIC = "video_results"
HOLD_ROPE_TOPIC = "hold_rope_keypoints"
CONSUMER_GROUP = "validate_script_consumer"

LOG_FILE = Path("/data2/caimoyao/api_outputs/validate_repair.log")

# label==1 必需的字段
BASE_FIELDS = {"label", "score", "box", "track_id"}
REQUIRED_FIELDS_LABEL1 = {"keypoints", "hold_rope", "hand_left_hold", "hand_right_hold",
                          "hand_left_score", "hand_right_score"}


# ──────────────────────── 日志 ────────────────────────

def log_repair(task_id, topic, success):
    """追加写入修复日志。"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    status = "✅" if success else "❌"
    line = f"[{datetime.now().isoformat()}] {status} {task_id} → {topic}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


# ───────────────────────── 文件扫描 ─────────────────────────

def validate_one_entry(entry, idx, file_path):
    """验证单个 entry，返回错误列表。"""
    errors = []

    if not isinstance(entry, dict):
        return [f"  entry[{idx}]: 不是 dict，实际类型 {type(entry).__name__}"]

    # 基础字段
    for f in BASE_FIELDS:
        if f not in entry:
            errors.append(f"  entry[{idx}]: 缺少基础字段 '{f}'")

    if "label" in entry:
        try:
            label_val = int(entry["label"])
        except (ValueError, TypeError):
            errors.append(f"  entry[{idx}]: label 不是整数")
            label_val = None

        # label==1 的额外字段
        if label_val == 1:
            for f in REQUIRED_FIELDS_LABEL1:
                if f not in entry:
                    errors.append(f"  entry[{idx}]: label=1 但缺少字段 '{f}'")
    else:
        label_val = None

    return errors


def validate_labels_file(file_path):
    """验证单个 JSON 文件，返回 (is_valid, errors)。"""
    errors = []

    # 1. 文件是否可读 / 解析
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"  JSON 解析失败: {e}"]
    except Exception as e:
        return False, [f"  读取失败: {e}"]

    # 2. 必须是 list
    if not isinstance(data, list):
        return False, [f"  顶层不是 list，实际 {type(data).__name__}"]

    # 3. 逐条验证
    for idx, entry in enumerate(data):
        errors.extend(validate_one_entry(entry, idx, file_path))

    return len(errors) == 0, errors


def scan_dir(task_dir):
    """扫描一个 task 目录下的所有 labels JSON，返回统计信息和文件级详情。"""
    labels_dir = task_dir / "labels"
    if not labels_dir.exists():
        return {
            "task_id": task_dir.name,
            "status": "NO_LABELS_DIR",
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "files": [],
        }

    result = {
        "task_id": task_dir.name,
        "status": "OK",
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "files": [],
    }

    for f in sorted(labels_dir.glob("*.json")):
        result["total"] += 1
        valid, errors = validate_labels_file(f)
        if valid:
            result["valid"] += 1
        else:
            result["invalid"] += 1
        result["files"].append({
            "filename": f.name,
            "valid": valid,
            "errors": errors,
        })

    if result["invalid"] > 0:
        result["status"] = "INCOMPLETE"
    elif result["total"] == 0:
        result["status"] = "EMPTY"

    return result


# ──────────────────────── Kafka 扫描 ────────────────────────

def list_tasks_from_kafka():
    """从 final_videos topic 读取所有 task_id，使用 assign() 跳过 group 协调。"""
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": CONSUMER_GROUP,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "session.timeout.ms": 10000,
        "broker.address.family": "v4",
    })

    metadata = consumer.list_topics(FINAL_TOPIC, timeout=5)
    partitions = sorted(metadata.topics[FINAL_TOPIC].partitions.keys())
    tasks = {}

    for p in partitions:
        tp = TopicPartition(FINAL_TOPIC, p, 0)
        consumer.assign([tp])
        lo, hi = consumer.get_watermark_offsets(tp)
        if hi == 0:
            continue  # 分区无消息

        while True:
            msg = consumer.poll(timeout=0.5)
            if msg is None:
                break
            if msg.error():
                continue
            if msg.offset() >= hi:
                break
            try:
                data = json.loads(msg.value().decode("utf-8"))
                tid = data.get("task_id", "")
                if tid:
                    tasks[tid] = data
            except json.JSONDecodeError:
                pass

    consumer.close()
    print(f"  共发现 {len(tasks)} 个 task (topic '{FINAL_TOPIC}', {len(partitions)} 个分区)")
    return tasks


# ──────────────────────── 对比 ────────────────────────

def cross_validate(task_dir_result, kafka_tasks):
    """交叉验证：检查目录中的 task 是否都在 Kafka 中完整。"""
    task_id = task_dir_result["task_id"]
    if task_id not in kafka_tasks:
        return f"❌ task '{task_id}': 在 final_videos topic 中未找到"
    kmsg = kafka_tasks[task_id]
    if kmsg.get("status") != "completed":
        return f"⚠️  task '{task_id}': final_videos 中 status={kmsg.get('status')}"
    return None


# ──────────────────────── Repair 发送 ────────────────────────

def _repair_needed_topics(result):
    """分析一个 task 的错误类型，按优先级返回需要重新发送的 topic（只返回一个）。"""
    for f_info in result.get("files", []):
        for err in f_info.get("errors", []):
            if "缺少基础字段" in err:
                return [VIDEO_TASKS_TOPIC]
    for f_info in result.get("files", []):
        for err in f_info.get("errors", []):
            if "缺少字段 'keypoints'" in err:
                return [VIDEO_RESULTS_TOPIC]
    # for f_info in result.get("files", []):
    #     for err in f_info.get("errors", []):
    #         for hf in ["hold_rope", "hand_left_hold", "hand_right_hold",
    #                    "hand_left_score", "hand_right_score"]:
    #             if f"缺少字段 '{hf}'" in err:
    #                 return [HOLD_ROPE_TOPIC]
    return []


def _publish_repair(producer, task_id, topics, kafka_data, scan_result):
    """向指定 topics 发送修复消息，kafka_data 来自 final_videos 的消息。"""
    video_path = kafka_data.get("video_path", "") if kafka_data else ""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    for topic in topics:
        if topic == VIDEO_TASKS_TOPIC:
            msg = {
                "task_id": task_id,
                "video_path": video_path,
                "filename": os.path.basename(video_path) if video_path else "",
                "conf_threshold": 0.5,
                "created_at": ts,
            }
        elif topic == VIDEO_RESULTS_TOPIC:
            msg = {
                "task_id": task_id,
                "status": "completed",
                "summary": {
                    "output_dir": str(TEMP_DIR / task_id / "output"),
                    "video_path": video_path,
                },
            }
        elif topic == HOLD_ROPE_TOPIC:
            msg = {
                "task_id": task_id,
                "status": "completed",
                "processed_labels": scan_result.get("total", 0),
                "video_path": video_path,
                "output_labels": str(TEMP_DIR / task_id / "labels"),
            }
        else:
            continue

        try:
            producer.produce(
                topic,
                key=task_id.encode("utf-8"),
                value=json.dumps(msg, ensure_ascii=False).encode("utf-8"),
            )
            producer.flush(timeout=10)
            print(f"  ✅ 已发送 {task_id} → {topic}")
            log_repair(task_id, topic, True)
        except Exception as e:
            print(f"  ❌ 发送 {task_id} → {topic} 失败: {e}")
            log_repair(task_id, topic, False)
            return False
    return True


# ──────────────────────── 消费者模式 ────────────────────────

def consumer_loop():
    """持续消费 final_videos topic，验证并修复不完整的 task。"""
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": "validate_repair_consumer_01",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "session.timeout.ms": 30000,
        "broker.address.family": "v4",
    })
    consumer.subscribe([FINAL_TOPIC])

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "broker.address.family": "v4",
    })

    # failed_at → 对应重发 topic
    FAILED_ROUTE = {
        "api_service": VIDEO_TASKS_TOPIC,
        "monitor_hold_rope": VIDEO_TASKS_TOPIC,
        "hand_crop": VIDEO_RESULTS_TOPIC,
    }

    print(f"✅ 消费者已启动，监听 {FINAL_TOPIC} topic...")
    empty_polls = 0

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            empty_polls += 1
            if empty_polls % 60 == 0:
                print(f"  ⏳ 等待消息中... ({empty_polls}s)")
            continue
        if msg.error():
            print(f"  ❌ 消费错误: {msg.error()}")
            continue

        empty_polls = 0
        try:
            data = json.loads(msg.value().decode("utf-8"))
        except json.JSONDecodeError:
            print(f"  ⚠️ 消息解析失败，直接 commit")
            consumer.commit(asynchronous=False)
            continue

        task_id = data.get("task_id", "")
        status = data.get("status", "")
        if not task_id:
            consumer.commit(asynchronous=False)
            continue

        # ── status == "failed" → 直接路由到对应阶段 ──
        if status == "failed":
            failed_at = data.get("failed_at", "")
            target = FAILED_ROUTE.get(failed_at)
            if target:
                # 原样转发 msg.value()
                try:
                    producer.produce(
                        target,
                        key=task_id.encode("utf-8"),
                        value=msg.value(),
                    )
                    producer.flush(timeout=10)
                    print(f"  🔁 {task_id}: failed@{failed_at} → {target}")
                    log_repair(task_id, f"reroute→{target}", True)
                except Exception as e:
                    print(f"  ❌ {task_id}: 路由到 {target} 失败: {e}")
                    log_repair(task_id, f"reroute→{target}", False)
                    continue  # 不 commit，等待重试
            else:
                print(f"  ⚠️ {task_id}: unknown failed_at='{failed_at}', 跳过")
            consumer.commit(asynchronous=False)
            continue

        # ── status == "completed" → 验证并修复 ──
        if status == "completed":
            task_dir = TEMP_DIR / task_id
            labels_dir = task_dir / "labels"

            if not labels_dir.exists():
                log_repair(task_id, "SKIP_no_labels_dir", True)
                consumer.commit(asynchronous=False)
                continue

            r = scan_dir(task_dir)
            if r["status"] == "OK":
                consumer.commit(asynchronous=False)
                continue

            print(f"  🔧 {task_id}: {r['status']} ({r['invalid']}/{r['total']} 帧不完整)")
            topics = _repair_needed_topics(r)
            if not topics:
                consumer.commit(asynchronous=False)
                continue

            kdata = {k: data.get(k) for k in ("video_path", "output_labels", "status")}
            ok = _publish_repair(producer, task_id, topics, kdata, r)
            if ok:
                consumer.commit(asynchronous=False)
                print(f"  ✅ {task_id} 修复成功，offset 已提交")
            else:
                print(f"  ❌ {task_id} 修复失败，不提交 offset（等待重试）")
            continue

        # ── 其他 status → 直接跳过 ──
        print(f"  ⚠️ {task_id}: 未知 status='{status}', 跳过")
        consumer.commit(asynchronous=False)


# ──────────────────────── 主逻辑 ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="验证 labels 完整性")
    parser.add_argument("--topic-only", action="store_true",
                        help="只扫描 Kafka final_videos topic，不扫目录")
    parser.add_argument("--repair", action="store_true",
                        help="对不完整文件仅打印修复建议（不实际修改）")
    parser.add_argument("--task", type=str, default=None,
                        help="只验证指定 task_id")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 结果文件路径")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 个 task")
    parser.add_argument("--consumer", action="store_true",
                        help="消费者模式：持续监听 final_videos topic，自动修复")
    args = parser.parse_args()

    # ── 消费者模式 ──
    if args.consumer:
        consumer_loop()
        return

    # ── 扫描 Kafka（以此为 task 名单） ──
    kafka_tasks = list_tasks_from_kafka()

    if args.topic_only:
        print("\n========== Kafka final_videos topic 中的 task ==========")
        for tid, data in sorted(kafka_tasks.items()):
            print(f"  {tid:50s} status={data.get('status')}")
        return

    print(f"\n📁 在 {TEMP_DIR} 中查找 Kafka topic 中出现的 task...")
    if not TEMP_DIR.exists():
        print(f"❌ 目录不存在: {TEMP_DIR}")
        sys.exit(1)

    task_ids = [args.task] if args.task else list(kafka_tasks.keys())
    if args.limit:
        task_ids = task_ids[:args.limit]
    task_dirs = []
    for tid in task_ids:
        d = TEMP_DIR / tid
        if d.exists() and d.is_dir():
            task_dirs.append(d)
        else:
            print(f"⚠️  {tid:50s}  本地目录不存在，跳过")

    if not task_dirs:
        print("❌ 没有可验证的本地目录")
        sys.exit(1)


    print(f"  发现 {len(task_dirs)} 个 task 目录\n")

    summary = {"OK": 0, "INCOMPLETE": 0, "EMPTY": 0, "NO_LABELS_DIR": 0}
    all_ok = True
    all_results_lock = threading.Lock()
    all_results = {}
    done_count = {"val": 0}
    total = len(task_dirs)

    def _print_progress():
        pct = done_count["val"] / total * 100
        err = summary["INCOMPLETE"] + summary["EMPTY"] + summary["NO_LABELS_DIR"]
        print(f"  进度: {done_count['val']}/{total} ({pct:.0f}%)  错误: {err}")

    def _scan_one(td):
        """单个 task 扫描（线程 worker）。"""
        tid = td.name
        r = scan_dir(td)
        with all_results_lock:
            all_results[tid] = r
            summary[r["status"]] += 1
            if r["status"] != "OK":
                nonlocal all_ok
                all_ok = False
            done_count["val"] += 1
            if done_count["val"] % 100 == 0:
                _print_progress()

        # 不完整的 task 立即发送修复消息
        if args.repair and r["status"] == "INCOMPLETE":
            topics = _repair_needed_topics(r)
            if topics:
                kdata = kafka_tasks.get(tid, {})
                _publish_repair(_repair_producer, tid, topics, kdata, r)

    # 提前创建修复用的 Producer（线程安全）
    _repair_producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "broker.address.family": "v4",
    }) if args.repair else None

    workers = min(len(task_dirs), 8)
    print(f"  使用 {workers} 个线程并行扫描 {total} 个 task...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_scan_one, task_dirs))
    _print_progress()

    # ── 只打印"Kafka 中有但本地目录不存在"的 task ──
    local_ids = {r["task_id"] for r in all_results.values()}
    for tid in kafka_tasks:
        if tid not in local_ids:
            print(f"⚠️  {tid:50s}  本地目录不存在，Kafka topic 中有消费记录")

    # ── 构建 JSON（只保存不完整的 task） ──
    incomplete = {tid: r for tid, r in all_results.items() if r["status"] != "OK"}

    # ── 写入 JSON（单次写入全部不完整 task） ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wrap = {
            "scan_time": datetime.now().isoformat(),
            "kafka_topic": FINAL_TOPIC,
            "kafka_task_count": len(kafka_tasks),
            "validation_summary": summary,
            "incomplete_tasks": incomplete,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(wrap, f, ensure_ascii=False, indent=2)
        print(f"\n📄 不完整 task 结果已保存: {output_path}  ({len(incomplete)} 个)")

    # ── 摘要 ──
    print("\n" + "=" * 70)
    print(f"  扫描完成  完整: {summary['OK']}  不完整: {summary['INCOMPLETE']}  "
          f"空目录: {summary['EMPTY']}  无 labels: {summary['NO_LABELS_DIR']}")
    print(f"  Kafka 中 task 数: {len(kafka_tasks)}")
    print("=" * 70)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
