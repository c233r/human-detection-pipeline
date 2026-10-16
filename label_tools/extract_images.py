import os
import json
import glob
import argparse
import cv2
from multiprocessing import Pool
import random

HOLD_ROPE_RESULTS_DIR = "/data2/caimoyao/api_outputs/hold_rope_results"
SAMPLING_RATE = 0.01
TARGET_FPS = 25.0
SKIP_SECONDS = 4.0
LABEL_START_SECONDS = 3.0
LOG_FILE = "/data2/caimoyao/api_outputs/extract_images.log"

def write_log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{msg}\n")

def get_mp4_filename(task_id):
    parts = task_id.split("_")
    mp4_name = "_".join(parts[:-1]) + ".mp4"
    return mp4_name

def process_track(category, task_id, track_id, mp4_dir):
    track_dir = os.path.join(HOLD_ROPE_RESULTS_DIR, category, task_id, str(track_id))
    json_files = sorted(glob.glob(os.path.join(track_dir, "*.json")))
    
    if not json_files:
        return 0
    
    num_samples = max(1, int(len(json_files) * SAMPLING_RATE))
    sampled_files = random.sample(json_files, num_samples)
    
    mp4_name = get_mp4_filename(task_id)
    mp4_path = os.path.join(mp4_dir, mp4_name)
    
    if not os.path.exists(mp4_path):
        return 0
    
    import decord
    decord.bridge.set_bridge('native')
    
    try:
        vr = decord.VideoReader(mp4_path)
        n_total = len(vr)
        video_fps = vr.get_avg_fps() if hasattr(vr, 'get_avg_fps') else 30.0
    except Exception as e:
        write_log(f"[!] Decode error: {mp4_name} - {e}")
        return 0
    
    frame_interval = max(1, int(round(video_fps / TARGET_FPS)))
    skip_frames = int(SKIP_SECONDS * video_fps)
    json_offset = int((SKIP_SECONDS - LABEL_START_SECONDS) * TARGET_FPS)
    
    frames_data = []
    abs_indices = []
    for json_path in sampled_files:
        rel_frame_num = int(os.path.splitext(os.path.basename(json_path))[0])
        adjusted_rel = rel_frame_num - json_offset
        if adjusted_rel < 0:
            continue
        
        abs_frame_num = skip_frames + adjusted_rel * frame_interval
        if abs_frame_num < n_total:
            abs_indices.append(abs_frame_num)
            frames_data.append((json_path, rel_frame_num))
    
    if not abs_indices:
        del vr
        return 0
    
    try:
        frames = vr.get_batch(abs_indices).asnumpy()
    except Exception:
        del vr
        return 0
    del vr
    
    output_dir = os.path.join(HOLD_ROPE_RESULTS_DIR, category, "images")
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for i, (json_path, rel_frame_num) in enumerate(frames_data):
        frame = frames[i]
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if not data:
                    continue
                box = data[0].get("box", [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, box)
            
            h, w = frame.shape[:2]
            
            orig_w = x2 - x1
            orig_h = y2 - y1
            pad_w = int(orig_w * 0.15)
            pad_h = int(orig_h * 0.15)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            if x2 > x1 and y2 > y1:
                
                if orig_h < 150 or orig_w < 150:
                    continue
                
                cropped = frame[y1:y2, x1:x2]
                cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                output_name = f"{task_id}_{track_id}_{rel_frame_num}.jpg"
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, cropped_bgr)
                count += 1
        except Exception:
            continue
    
    del frames
    return count


def process_task(args):
    category, task_id, mp4_dir = args
    task_dir = os.path.join(HOLD_ROPE_RESULTS_DIR, category, task_id)
    track_dirs = [f.path for f in os.scandir(task_dir) if f.is_dir()]
    
    mp4_name = get_mp4_filename(task_id)
    mp4_path = os.path.join(mp4_dir, mp4_name)
    n_total = 0
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    images_dir = os.path.join(HOLD_ROPE_RESULTS_DIR, category, "images")
    processed_tracks = set()
    if os.path.exists(images_dir):
        for fname in os.listdir(images_dir):
            if fname.endswith(".jpg") and fname.startswith(f"{task_id}_"):
                track_part = fname.rsplit("_", 2)[1]
                try:
                    processed_tracks.add(int(track_part))
                except ValueError:
                    continue
    
    pending_tracks = []
    for track_dir in track_dirs:
        track_id = int(os.path.basename(track_dir))
        if track_id not in processed_tracks:
            pending_tracks.append(track_id)
    
    skipped = len(track_dirs) - len(pending_tracks)
    if skipped > 0:
        print(f"  [{task_id}] Skipping {skipped} already processed tracks", flush=True)
    
    total_frames = 0
    for track_dir in track_dirs:
        total_frames += len(glob.glob(os.path.join(track_dir, "*.json")))
    
    total_extracted = 0
    for track_id in pending_tracks:
        extracted = process_track(category, task_id, track_id, mp4_dir)
        total_extracted += extracted
    
    return task_id, total_extracted, total_frames, n_total


def process_category(category, mp4_dir):
    category_dir = os.path.join(HOLD_ROPE_RESULTS_DIR, category)
    
    task_dirs = [f.path for f in os.scandir(category_dir) if f.is_dir() and not f.name == "images"]
    
    if not task_dirs:
        return 0
    
    task_ids = [os.path.basename(d) for d in task_dirs]
    task_args = [(category, tid, mp4_dir) for tid in task_ids]
    total_tasks = len(task_ids)
    
    num_workers = 32
    total_extracted = 0
    
    with Pool(num_workers) as pool:
        for idx, (tid, extracted, total_frames, n_total) in enumerate(pool.imap_unordered(process_task, task_args)):
            total_extracted += extracted
            if (idx + 1) % 10 == 0 or idx == total_tasks - 1:
                print(f"  [{tid}] video total: {n_total}, json frames: {total_frames}, extracted: {extracted}", flush=True)
                print(f"  Completed {idx + 1}/{total_tasks} tasks", flush=True)
    
    return total_extracted


def main():
    parser = argparse.ArgumentParser(description="Extract images from MP4 videos based on track_id results")
    parser.add_argument("--mp4_dir", required=True, help="Directory containing MP4 video files")
    args = parser.parse_args()
    
    mp4_dir = args.mp4_dir
    if not os.path.exists(mp4_dir):
        print(f"Error: MP4 directory '{mp4_dir}' does not exist")
        return

    # print(f"\nProcessing positive samples...")
    # pos_count = process_category("positive", mp4_dir)
    # print(f"Extracted {pos_count} images from positive samples")
    
    print(f"Processing negative samples...")
    neg_count = process_category("negative", mp4_dir)
    print(f"Extracted {neg_count} images from negative samples")
    
    print(f"\nTotal extracted: {neg_count + pos_count} images")


if __name__ == "__main__":
    main()