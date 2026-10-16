#!/usr/bin/env python3
"""将标签 JSON 中的检测框绘制到视频上，输出标注后的视频。

用法:
    python draw_labels_on_video.py <video_path> <labels_dir> [--output OUTPUT] [--fps FPS] [--skip SECONDS]
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path


def draw_labels(video_path: str, labels_dir: str, output: str = None,
                fps: float = 25.0, skip_seconds: float = 3.0):
    """读取视频和标签，绘制标注框后输出视频。"""
    video_path = Path(video_path)
    labels_dir = Path(labels_dir)

    if not video_path.exists():
        print(f"[错误] 视频文件不存在: {video_path}")
        return
    if not labels_dir.exists():
        print(f"[错误] 标签目录不存在: {labels_dir}")
        return

    if output is None:
        output = str(video_path.parent / f"{video_path.stem}_annotated{video_path.suffix}")
    output = str(output)

    # 读取全部视频帧
    import decord
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(str(video_path))
    n_total = len(vr)
    print(f"视频总帧数: {n_total}")

    # 读取标签文件
    label_files = sorted(labels_dir.glob("*.json"))
    print(f"标签文件数: {len(label_files)}")

    # 构建帧索引 → labels 的映射
    frame_labels = {}
    for lf in label_files:
        try:
            with open(lf, 'r') as f:
                frame_labels[int(lf.stem)] = json.load(f)
        except Exception as e:
            print(f"  跳过 {lf.name}: {e}")

    # 跳过前 N 秒
    skip_frames = int(skip_seconds * fps)
    if n_total <= skip_frames:
        print(f"[错误] 视频太短，跳过 {skip_seconds}秒后无帧")
        return
    start_frame = skip_frames

    # 读帧
    indices = list(range(start_frame, n_total))
    print(f"读取 {len(indices)} 帧 (跳过前 {skip_frames} 帧)...")
    frames_rgb = vr.get_batch(indices).asnumpy()
    del vr

    h, w = frames_rgb.shape[1], frames_rgb.shape[2]

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (w, h))

    for i, frame_idx in enumerate(indices):
        frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        draw = frame_bgr.copy()
        labels = frame_labels.get(frame_idx)
        if labels:
            for p in labels:
                plabel = p.get("label")
                plabel_int = int(plabel) if plabel is not None else -1
                if plabel_int != 1:
                    continue
                x1, y1, x2, y2 = map(int, p["box"])
                tid = p.get("track_id", -1)

                if "hold_rope" in p:
                    hold = p["hold_rope"]
                    color = (0, 255, 0) if hold else (0, 0, 255)
                    text = "hold_rope" if hold else "no_hold"
                else:
                    color = (128, 128, 128)
                    text = "unclassified"

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f"ID:{tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(draw, text, (x1, y1 - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(draw)

    out.release()
    print(f"标注视频已保存: {output} ({len(indices)}帧, {len(frame_labels)}帧有标签)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将标签绘制到视频上")
    parser.add_argument("video_path", help="输入视频路径")
    parser.add_argument("labels_dir", help="标签 JSON 目录")
    parser.add_argument("--output", "-o", default=None, help="输出视频路径（默认: video_annotated.mp4）")
    parser.add_argument("--fps", type=float, default=25.0, help="输出视频帧率（默认: 25）")
    parser.add_argument("--skip", type=float, default=3.0, help="跳过前 N 秒（默认: 3）")
    args = parser.parse_args()

    draw_labels(args.video_path, args.labels_dir, args.output, args.fps, args.skip)
