import os
import json
import glob
import time
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count

INPUT_DIR = "/data2/caimoyao/api_outputs/temp"
OUTPUT_DIR = "/data2/caimoyao/api_outputs/hold_rope_results"

POSITIVE_THRESHOLD = 0.95
NEGATIVE_THRESHOLD = 0.3
MAX_DEPTH = 10
MIN_FRAMES = 5

def entropy(pos_count, total):
    if total == 0:
        return 0.0
    p = pos_count / total
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def split_neutral(items, task_id, track_id, depth=0):
    """按信息熵最大减少幅度递归拆分，深度限制为 MAX_DEPTH"""
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

def process_video_folder(video_folder):
    """单遍读取：分析统计 + 提取保存，避免重复IO"""
    task_id = os.path.basename(video_folder)
    labels_dir = os.path.join(video_folder, "labels")
    
    if not os.path.exists(labels_dir):
        return None

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
        return None

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



def main():
    start_time = time.time()
    video_folders = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]
    print(f"Total video folders: {len(video_folders)}")

    os.makedirs(os.path.join(OUTPUT_DIR, "positive"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "negative"), exist_ok=True)

    num_workers = 8
    print(f"Processing using {num_workers} workers...")

    all_results = []
    total_split = {"positive": 0, "negative": 0}
    with Pool(num_workers) as pool:
        for i, ret in enumerate(pool.imap_unordered(process_video_folder, video_folders)):
            if ret:
                results, split_counts = ret
                all_results.extend(results)
                total_split["positive"] += split_counts["positive"]
                total_split["negative"] += split_counts["negative"]
            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{len(video_folders)}...")

    stats = {
        "total_tracks": len(all_results),
        "positive": sum(1 for r in all_results if r["category"] == "positive"),
        "negative": sum(1 for r in all_results if r["category"] == "negative"),
        "neutral": sum(1 for r in all_results if r["category"] == "neutral")
    }

    print(f"\n=== Statistics (by track_id) ===")
    print(f"Total tracks: {stats['total_tracks']}")
    print(f"Positive (ratio >= {POSITIVE_THRESHOLD}): {stats['positive']}")
    print(f"Negative (ratio < {NEGATIVE_THRESHOLD}): {stats['negative']}")
    print(f"Neutral ({NEGATIVE_THRESHOLD} <= ratio < {POSITIVE_THRESHOLD}): {stats['neutral']} (recursively split into pos/neg by frames)")

    pos_ratio = stats["positive"] / stats["total_tracks"] * 100 if stats["total_tracks"] > 0 else 0
    neg_ratio = stats["negative"] / stats["total_tracks"] * 100 if stats["total_tracks"] > 0 else 0
    neu_ratio = stats["neutral"] / stats["total_tracks"] * 100 if stats["total_tracks"] > 0 else 0
    print(f"\n=== Percentages (tracks) ===")
    print(f"Positive: {pos_ratio:.1f}%")
    print(f"Negative: {neg_ratio:.1f}%")
    print(f"Neutral: {neu_ratio:.1f}%")
    
    print(f"\n=== Frames after split ===")
    print(f"Positive frames: {total_split['positive']}")
    print(f"Negative frames: {total_split['negative']}")
    print(f"Total frames: {total_split['positive'] + total_split['negative']}")

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\n=== Time ===")
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:.2f}")

if __name__ == "__main__":
    main()