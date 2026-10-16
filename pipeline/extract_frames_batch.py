#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量抽帧脚本
对指定目录下的所有视频进行抽帧，支持每秒抽2帧
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any

# 支持的视频格式
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}


def find_video_files(directory: str) -> List[str]:
    """
    递归查找目录中的所有视频文件
    
    Args:
        directory: 要扫描的目录路径
        
    Returns:
        视频文件路径列表
    """
    video_files = []
    directory = Path(directory)
    
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(directory.rglob(f'*{ext}'))
    
    seen = set()
    unique_files = []
    for f in video_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(str(f))
    
    return sorted(unique_files)


def get_video_duration(video_path: str) -> float:
    """
    获取视频时长（秒）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        视频时长（秒），失败返回0
    """
    try:
        if sys.platform == 'win32':
            cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                encoding='utf-8',
                errors='ignore'
            )
        else:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True,
                text=True
            )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return float(output)
    except Exception as e:
        print(f"[ERROR] 获取视频时长失败: {video_path}, 错误: {e}")
    
    return 0.0


def extract_frames_to_file(video_path: str, output_dir: str, fps: float = 0.5, video_index: int = 0) -> Dict[str, Any]:
    """
    从视频中按指定帧率抽帧并保存到文件（所有视频的帧放在同一个目录下）
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（所有帧都放在这个目录下）
        fps: 每秒抽帧数，默认2帧/秒
        video_index: 视频索引，用于生成唯一文件名
        
    Returns:
        抽帧结果字典
    """
    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    
    # 清理文件名中的特殊字符，避免ffmpeg出错
    safe_name = ''.join(c for c in name_without_ext if c.isalnum() or c in ('_', '-', '.'))
    if not safe_name:
        safe_name = f"video_{video_index}"
    
    duration = get_video_duration(video_path)
    if duration <= 0:
        print(f"[WARNING] 无法获取视频时长，跳过: {video_name}")
        return {
            "video_path": video_path,
            "status": "failed",
            "reason": "无法获取视频时长"
        }
    
    # 跳过前 300 秒（5分钟）
    skip_seconds = 3
    if duration <= skip_seconds:
        print(f"[WARNING] 视频时长 {duration:.2f}s <= {skip_seconds}s，跳过: {video_name}")
        return {
            "video_path": video_path,
            "status": "skipped",
            "reason": f"视频时长 {duration:.2f}s 不足 {skip_seconds}s",
        }
    
    effective_duration = duration - skip_seconds
    estimated_frames = int(effective_duration * fps)
    
    # 使用ffmpeg + GPU加速抽帧
    # -ss 300: 跳过前300秒（放在 -i 前为快速 seek）
    # -hwaccel cuda: CUDA 硬件加速解码
    frame_pattern = os.path.join(output_dir, f"video_{video_index:03d}_{safe_name}_frame_%04d.jpg")
    
    # 检测 NVIDIA GPU 是否可用
    has_gpu = False
    try:
        check = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        has_gpu = check.returncode == 0 and bool(check.stdout.strip())
    except Exception:
        pass

    cmd = [
        'ffmpeg',
        '-ss', str(skip_seconds),
    ]
    if has_gpu:
        # 仅使用 GPU 硬件解码加速，不保留帧在 GPU 显存
        # 去掉 -hwaccel_output_format cuda，否则 -f image2 无法输出 JPEG（帧在 GPU 显存中）
        cmd += ['-hwaccel', 'cuda']
    cmd += [
        '-i', video_path,
        '-r', str(fps),
        '-q:v', '2',
        '-f', 'image2',
        frame_pattern,
        '-y',
    ]
    
    print(f"[INFO] 开始抽帧: {video_name}")
    print(f"[INFO] 视频总时长: {duration:.2f}秒, 跳过前 {skip_seconds}s, "
          f"有效时长: {effective_duration:.2f}s, 预计 ~{estimated_frames} 帧")
    print(f"[INFO] GPU 加速: {'启用' if has_gpu else '未启用(降级为CPU)'}")
    
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ' '.join(f'"{x}"' if ' ' in x else x for x in cmd),
                shell=True,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                encoding='utf-8',
                errors='ignore'
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                encoding='utf-8',
                errors='ignore'
            )

        os.remove(video_path)
        print(f"[INFO] 已删除原视频: {video_name}")
        
        if result.returncode == 0:
            # 统计抽帧数量
            prefix = f"video_{video_index:03d}_{safe_name}_frame_"
            frame_files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith('.jpg')]
            frame_count = len(frame_files)
            
            print(f"[SUCCESS] 抽帧完成: {video_name}, 共提取 {frame_count} 帧")
            
            
            return {
                "video_path": video_path,
                "video_name": video_name,
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "output_dir": output_dir,
                "frame_prefix": prefix,
                "status": "success"
            }
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            # 打印完整错误，供上层 subprocess 捕获
            print(f"[ERROR] 抽帧失败: {video_name}")
            print(f"[ERROR] ffmpeg 返回码: {result.returncode}")
            print(f"[ERROR] ffmpeg stderr:\n{error_msg}")
            return {
                "video_path": video_path,
                "video_name": video_name,
                "status": "failed",
                "reason": error_msg[:500] if error_msg else "未知错误"
            }
    
    except Exception as e:
        print(f"[ERROR] 抽帧异常: {video_name}, 错误: {e}")
        return {
            "video_path": video_path,
            "video_name": video_name,
            "status": "failed",
            "reason": str(e)
        }


def main():
    # 默认配置
    input_dir = "/data2/caimoyao/videos/refined_output/篮球"
    output_base_dir = "/data2/caimoyao/videos/refined_output/篮球_frames"
    fps = 0.5  # 每秒抽2帧
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_base_dir = sys.argv[2]
    if len(sys.argv) > 3:
        fps = float(sys.argv[3])
    
    print(f"=" * 60)
    print(f"批量抽帧脚本")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"抽帧帧率: {fps} 帧/秒")
    print(f"=" * 60)
    
    # 检查输入目录
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 查找所有视频文件
    video_files = find_video_files(input_dir)
    total_videos = len(video_files)
    
    print(f"[INFO] 找到 {total_videos} 个视频文件")
    
    if total_videos == 0:
        print("[INFO] 没有找到视频文件，退出")
        sys.exit(0)
    
    # 记录结果
    results = []
    success_count = 0
    failed_count = 0
    
    # 批量抽帧
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{total_videos}] 处理: {os.path.basename(video_path)}")
        
        result = extract_frames_to_file(video_path, output_base_dir, fps, video_index=i)
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1
    
    # 保存结果到JSON文件
    result_file = os.path.join(output_base_dir, "extraction_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"\n" + "=" * 60)
    print(f"抽帧完成!")
    print(f"总视频数: {total_videos}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"结果文件: {result_file}")
    print(f"=" * 60)
    
    # 如果有失败的视频，退出码为非零，让调用方知道
    if failed_count > 0:
        print(f"[ERROR] {failed_count} 个视频抽帧失败，请检查上方日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
