"""
可视化标签视频 — 读取视频帧 + labels JSON，绘制标注框+关键点后保存为 output.mp4

用法:
    python visualize_labels_video.py

配置:
    TASK_ID: 任务ID，也是 labels 目录名和视频文件名
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# ── 配置 ──
TASK_ID = "2026-06-11_14-52-40-124"
LABELS_DIR = Path(f"/data2/caimoyao/api_outputs/temp/{TASK_ID}/labels")
VIDEO_PATH = Path(f"/data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos/videos/{TASK_ID}.mp4")
OUTPUT_PATH = Path(f"output_visual.mp4")
# ── 配置结束 ──

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from video_utils import read_video_frames, read_video_frames_ffmpeg


# ── 关键点绘制配置 ──
# COCO 17 点 + 脚部 6 点 + 额外 2 点 = 25 点（mmpose 跳绳模型）
# 索引 0-16 是标准 COCO 骨架，17-22 是脚部，23-24 是额外点
SKELETON_CONNECTIONS = [
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 上肢（左右对称）
    (5, 7), (7, 9),       # 左肩-左肘-左腕
    (6, 8), (8, 10),      # 右肩-右肘-右腕
    (5, 6),               # 左右肩
    # 躯干
    (5, 11), (6, 12),     # 肩-髋
    (11, 12),             # 左右髋
    # 下肢
    (11, 13), (13, 15),   # 左髋-左膝-左踝
    (12, 14), (14, 16),   # 右髋-右膝-右踝
    # 脚部（17-22）
    (15, 17), (15, 19), (17, 18),   # 左脚：踝-大脚趾/脚跟，大脚趾-小脚趾
    (16, 20), (16, 22), (20, 21),   # 右脚：踝-大脚趾/脚跟，大脚趾-小脚趾
    # 额外点（23, 24）：通常为左右脸部，连到眼/耳
    (1, 23), (2, 24),
]

# 关键点置信度阈值：低于此值不绘制
KP_CONF_THRESHOLD = 0.3

# 颜色配置（BGR）
KP_COLOR = (0, 255, 255)        # 黄色：关键点
KP_LOW_CONF_COLOR = (128, 128, 128)  # 灰色：低置信度点
BONE_COLOR = (255, 165, 0)      # 橙色：骨架连线
KP_RADIUS = 4
BONE_THICKNESS = 2


def draw_keypoints(draw, keypoints):
    """绘制 25 个关键点 + 骨架连线。

    Args:
        draw: cv2 图像 (BGR)
        keypoints: [[x, y, conf], ...] 共 25 个点
    """
    if not keypoints or len(keypoints) == 0:
        return

    n_points = len(keypoints)
    pts = []
    confs = []
    for kp in keypoints:
        if len(kp) >= 3:
            x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
        elif len(kp) == 2:
            x, y, c = float(kp[0]), float(kp[1]), 1.0
        else:
            continue
        pts.append((int(x), int(y)))
        confs.append(c)

    # 1. 先画骨架连线（只连接两端点都高置信度的）
    for i, j in SKELETON_CONNECTIONS:
        if i >= n_points or j >= n_points:
            continue
        if confs[i] < KP_CONF_THRESHOLD or confs[j] < KP_CONF_THRESHOLD:
            continue
        cv2.line(draw, pts[i], pts[j], BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    # 2. 再画关键点（覆盖在线条上方）
    for idx, (pt, c) in enumerate(zip(pts, confs)):
        if c < KP_CONF_THRESHOLD:
            # 低置信度点用灰色小点
            cv2.circle(draw, pt, 2, KP_LOW_CONF_COLOR, -1, cv2.LINE_AA)
        else:
            cv2.circle(draw, pt, KP_RADIUS, KP_COLOR, -1, cv2.LINE_AA)
            # 标注点序号（可选，调试用）
            # cv2.putText(draw, str(idx), (pt[0]+5, pt[1]-5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def _frame_index_to_label_file(fidx: int) -> Path:
    """帧位置索引 → 标签文件路径 (6位数字)"""
    return LABELS_DIR / f"{fidx:06d}.json"


def main():
    print(f"视频: {VIDEO_PATH}")
    print(f"标签: {LABELS_DIR}")
    print(f"输出: {OUTPUT_PATH}")

    if not VIDEO_PATH.exists():
        print(f"错误: 视频文件不存在 {VIDEO_PATH}")
        return
    if not LABELS_DIR.exists():
        print(f"错误: 标签目录不存在 {LABELS_DIR}")
        return

    # 1. 读取视频帧（RGB）
    print("读取视频帧...")
    frames_rgb = read_video_frames(
        str(VIDEO_PATH),
    )
    print(f"   共 {len(frames_rgb)} 帧, 形状 {frames_rgb.shape}")

    # 2. 获取第一帧确定宽高
    h, w = frames_rgb.shape[1:3]

    # 3. 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, 25.0, (w, h))

    labeled_count = 0

    for i, frame_rgb in enumerate(frames_rgb):
        # RGB → BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        draw = frame_bgr.copy()

        label_file = _frame_index_to_label_file(i)
        if label_file.exists():
            labeled_count += 1
            with open(label_file, 'r') as f:
                persons = json.load(f)

            for p in persons:
                plabel = p.get("label")
                plabel_int = int(plabel) if plabel is not None else -1

                # 只处理人的标注 (label=1)，背景(label=0)不画框
                if plabel_int != 1:
                    continue

                # 框
                x1, y1, x2, y2 = map(int, p.get("box", [0, 0, 0, 0]))
                tid = p.get("track_id", -1)
                hold = p.get("hold_rope", None)

                if hold is not None:
                    color = (0, 255, 0) if hold else (0, 0, 255)
                    text = "HOLD" if hold else "NO_HOLD"
                else:
                    color = (128, 128, 128)
                    text = "unclassified"

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f"ID:{tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(draw, text, (x1, y1 - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # # ── 绘制关键点 ──
                keypoints = p.get("keypoints")
                if keypoints:
                    draw_keypoints(draw, keypoints)

        # 帧号水印
        cv2.putText(draw, f"Frame:{i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(draw)

    out.release()

    print(f"\n完成! 共 {len(frames_rgb)} 帧, 其中 {labeled_count} 帧有标签")
    print(f"输出视频: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
