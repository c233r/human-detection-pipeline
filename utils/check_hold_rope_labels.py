"""
检查 temp 下每个 task 的 labels 中是否包含 hold_rope 关键字。
每个 task 随机抽 10 个 label JSON，全部含 hold_rope 则 success，否则 fail。
Fail 的 task 自动重新发送到 video_tasks 主题，按 client_example_kafka 格式组装配消息。
"""
import os
import json
import random
import time
from pathlib import Path

from confluent_kafka import Producer

TEMP_DIR = Path("/data2/caimoyao/api_outputs/temp")
SAMPLE_SIZE = 10

KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
KAFKA_TOPIC = "video_tasks"

# 视频源目录列表（按优先级顺序搜索 task_id.mp4）
VIDEO_DIRS = [
    Path("/data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos/videos"),
    Path("/data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos"),
]

CONF_THRESHOLD = 0.5


def find_video_path(task_id: str) -> str:
    """根据 task_id 查找视频文件路径。"""
    for video_dir in VIDEO_DIRS:
        candidates = [
            video_dir / f"{task_id}.mp4",
            video_dir / f"{task_id}.avi",
            video_dir / f"{task_id}.mov",
        ]
        for p in candidates:
            if p.exists():
                return str(p.resolve())
    return ""


def check_task(task_dir: Path) -> dict:
    """检查单个 task：随机采样 SAMPLE_SIZE 个 label 文件，验证是否包含 hold_rope"""
    labels_dir = task_dir / "labels"
    if not labels_dir.exists():
        return {"task_id": task_dir.name, "status": "fail", "reason": "labels dir not found"}

    json_files = sorted(labels_dir.glob("*.json"))
    if not json_files:
        return {"task_id": task_dir.name, "status": "fail", "reason": "no label files"}

    # 如果文件总数少于 SAMPLE_SIZE，全部检查
    sample = json_files if len(json_files) < SAMPLE_SIZE else random.sample(json_files, SAMPLE_SIZE)

    has_hold_rope_count = 0
    no_hold_rope_files = []

    for jf in sample:
        try:
            with open(jf, "r") as f:
                labels = json.load(f)
        except Exception:
            no_hold_rope_files.append(f"{jf.name}: parse error")
            continue

        found = False
        for item in labels:
            if isinstance(item, dict) and item.get("label") == 1:
                if "hold_rope" in item:
                    found = True
                    break

        if found:
            has_hold_rope_count += 1
        else:
            no_hold_rope_files.append(jf.name)

    status = "success" if has_hold_rope_count == len(sample) else "fail"

    return {
        "task_id": task_dir.name,
        "status": status,
        "total_labels": len(json_files),
        "sampled": len(sample),
        "with_hold_rope": has_hold_rope_count,
        "no_hold_rope_files": no_hold_rope_files,
    }


def send_to_video_tasks(producer: Producer, task_id: str):
    """按 client_example_kafka 格式发送任务到 video_tasks 主题。"""
    video_path = find_video_path(task_id)
    if not video_path:
        print(f"  ⚠️  {task_id}: 未找到视频文件，跳过重新发送")
        return

    message = {
        "task_id": task_id,
        "video_path": video_path,
        "filename": os.path.basename(video_path),
        "conf_threshold": CONF_THRESHOLD,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        producer.produce(
            KAFKA_TOPIC,
            key=task_id.encode("utf-8"),
            value=json.dumps(message).encode("utf-8"),
        )
        producer.flush(timeout=10)
        print(f"  📤  {task_id}: 已重新发送到 {KAFKA_TOPIC}")
    except Exception as e:
        print(f"  ❌  {task_id}: 发送失败 {e}")


def main():
    print("=" * 60)
    print("  Hold Rope 标签检测统计")
    print(f"  目录: {TEMP_DIR}")
    print(f"  采样: 每个 task 随机抽 {SAMPLE_SIZE} 个文件")
    print(f"  Fail 时自动发回 {KAFKA_TOPIC}")
    print("=" * 60)

    task_dirs = sorted([d for d in TEMP_DIR.iterdir() if d.is_dir()])
    if not task_dirs:
        print(f"❌ {TEMP_DIR} 下没有子目录")
        return

    print(f"\n共发现 {len(task_dirs)} 个 task\n")

    # 初始化 Kafka Producer
    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "broker.address.family": "v4",
    })

    results = []
    for task_dir in task_dirs:
        result = check_task(task_dir)
        results.append(result)
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_icon} {result['task_id']:30s}  [{result['status']:7s}]  "
              f"总文件: {result['total_labels']:4d}  抽检: {result['sampled']:2d}  "
              f"含hold_rope: {result['with_hold_rope']:2d}/{result['sampled']}")

        if result["status"] == "fail":
            send_to_video_tasks(producer, result["task_id"])

    producer.close()

    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = sum(1 for r in results if r["status"] == "fail")

    print("\n" + "=" * 60)
    print(f"  统计结果")
    print(f"  ✅ Success: {success_count} / {len(results)}")
    print(f"  ❌ Fail:    {fail_count} / {len(results)}")
    print(f"  📤 已重新发送 {fail_count} 个 fail task 到 {KAFKA_TOPIC}")
    print("=" * 60)


if __name__ == "__main__":
    main()
