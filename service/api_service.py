"""
EdgeCrafter 推理 API 服务
基于 FastAPI 构建，提供视频推理接口
通过调用外部脚本执行推理和抽帧

兼容命令:
    CUDA_VISIBLE_DEVICES=5 python tools/inference/torch_inf.py -c configs/ecdet/ecdet_l.yml -r /data2/caimoyao/EdgeCrafter-main/ecdetseg/outputs/ecdet_l/best.pth -i /data2/caimoyao/videos/refined_output/篮球_frames -t 0.5

功能特性:
- 视频上传队列管理
- 异步任务处理
- 自动清理临时文件（视频和帧）
- 任务状态查询
"""

import os
import sys
import re
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
import multiprocessing as mp

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_service.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 取消上传大小限制
MAX_UPLOAD_SIZE = None

# ==================== 配置 ====================
class Config:
    """服务配置类"""
    
    # 项目路径（兼容用户命令）
    PROJECT_ROOT = Path("/data2/caimoyao/EdgeCrafter-main/ecdetseg")
    CONFIG_PATH = PROJECT_ROOT / "configs/ecdet/ecdet_l.yml"
    CHECKPOINT_PATH = PROJECT_ROOT / "models/last.pth"
    INFERENCE_SCRIPT = PROJECT_ROOT / "tools/inference/torch_inf.py"
    EXTRACT_FRAMES_SCRIPT = Path("/data2/caimoyao/extract_frames_batch.py")
    
    # 推理配置
    CONF_THRESHOLD = 0.5
    
    # 类别配置（person, face, ball）
    CLASSES = ["person", "face", "ball"]
    
    # 输出配置
    OUTPUT_DIR = Path("/data2/caimoyao/api_outputs")
    RESULT_DIR = OUTPUT_DIR / "results"
    UPLOAD_DIR = OUTPUT_DIR / "uploads"
    FRAME_DIR = OUTPUT_DIR / "frames"
    LOG_DIR = OUTPUT_DIR / "logs"
    
    # 服务配置
    HOST = "0.0.0.0"
    PORT = 8000
    MAX_UPLOAD_SIZE = 0  # 不限制大小
    MAX_WORKERS = 2
    
    # 视频配置
    FPS = 0.5  # OpenCV 抽取帧率（25=全帧率）
    MAX_FRAMES_PER_VIDEO = 0  # 不限制帧数
    
    # 队列配置
    MAX_QUEUE_SIZE = 40
    
    # 输出控制
    SAVE_OUTPUT = False  # 是否保存标注图（output/warning 目录）
    
    # GPU 配置: 多卡并行 + CUDA Stream + 每卡多进程
    GPU_IDS = [ 5, 7]  # 使用的 GPU 卡号列表
    WORKERS_PER_GPU = 4  # 每卡最大 worker 数（启动时根据空闲显存动态计算，每 10G 开 1 个）
    NUM_STREAMS = 4  # 每张卡内部 CUDA stream 数（实测每帧~1.2GB，8路仅1.4GB，A100 80G 安全）
    
    # RAG 配置（可选）
    RAG_CACHE = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/rag_db.pkl"  # e.g. "/data2/caimoyao/EdgeCrafter-main/ecdetseg/tools/inference/rag_db.pkl"
    RAG_THRESHOLD = 0.75
    
    # 输出配置

config = Config()

# ==================== 初始化 ====================
app = FastAPI(
    title="EdgeCrafter Detection API",
    description="""
    EdgeCrafter 目标检测推理服务 API
    
    通过调用外部脚本执行推理和抽帧
    支持视频上传队列、异步处理、自动清理
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务队列和状态管理
task_status = {}  # task_id -> status info（主进程本地 dict，GPU worker 通过 result_queue 回写）
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
# 分离的线程池：result_watcher 独占一个线程，避免被 mp.Queue.get() 阻塞拖死主 executor
_result_executor = ThreadPoolExecutor(max_workers=1)

# ── 多进程 GPU 推理基础设施 ─────────────────────────────────────
mp_ctx = mp.get_context("spawn")  # spawn 模式对 CUDA 安全
gpu_task_queue = None    # mp.Queue — 主进程→GPU workers
gpu_result_queue = None  # mp.Queue — GPU workers→主进程结果
gpu_processes = []       # [Process, ...]
_shutdown_event = asyncio.Event()  # 优雅退出标志

# 任务状态枚举
class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    INFERENCING = "inferencing"
    COMPLETED = "completed"
    FAILED = "failed"


# ══════════════════════════════════════════════════════════════════════
#  GPU Worker Process — 每张卡一个独立进程，内部用 CUDA Stream 并行推理
# ══════════════════════════════════════════════════════════════════════

def gpu_worker_process(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config_dict: dict,
):
    """GPU worker 进程入口（每个进程绑定一张卡，常驻运行）。

    Parameters
    ----------
    gpu_id : GPU 卡号，用于设置 CUDA_VISIBLE_DEVICES
    task_queue : 接收任务的 mp.Queue（元素: dict）
    result_queue : 发送结果的 mp.Queue（元素: dict）
    config_dict : 序列化后的配置（字符串/基本类型，确保可 pickle）
    """
    # ── 绑定 GPU ──────────────────────────────────────────────────
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import json
    import numpy as np
    from pathlib import Path
    from PIL import Image

    device = torch.device("cuda:0")

    # 导入推理模块
    sys.path.insert(0, config_dict["project_root"])
    from tools.inference.torch_inf import (
        ECInferencer, build_model, draw, Result, IMAGE_EXTS,
    )

    # ── 加载模型（仅一次） ─────────────────────────────────────────
    model, img_size, task = build_model(
        config_dict["config_path"],
        config_dict["checkpoint_path"],
        device,
    )
    inferencer = ECInferencer(
        model=model, task=task, device=device,
        size=img_size, thresh=config_dict.get("thresh", 0.5),
    )
    logger.info(f"[GPU-{gpu_id}] 模型加载完成，img_size={img_size}")

    # ── 初始化流式聚类更新器（包含 RAG 知识库） ────────────────────
    from streaming_cluster_updater import StreamingClusterUpdater

    updater = None
    rag_cache = config_dict.get("rag_cache")
    if rag_cache and os.path.exists(rag_cache):
        updater = StreamingClusterUpdater(
            rag_db_path=rag_cache,
            device=device,
        )
        updater.load_state()
        logger.info(f"[GPU-{gpu_id}] RAG 知识库已加载: "
                    f"path={rag_cache}, "
                    f"{updater.rag_db.total_clusters} clusters, "
                    f"{updater.rag_db.total_samples} samples, "
                    f"threshold={updater.rag_db.decision_threshold:.2f}")
    else:
        logger.info(f"[GPU-{gpu_id}] RAG 知识库未加载: "
                    f"path={rag_cache}, "
                    f"exists={os.path.exists(rag_cache) if rag_cache else 'N/A'}")

    num_streams = config_dict.get("num_streams", 4)
    per_frame_mem_mb = None       # 动态测量
    safety_ratio = 0.60            # 最多用 60% 当前可用显存

    # ── 预热 + 测量每帧显存 ─────────────────────────────────────
    from PIL import Image as _PIL
    _warm_img = _PIL.new("RGB", (2048, 2048))
    torch.cuda.reset_peak_memory_stats()
    _ = inferencer.infer(_warm_img)
    torch.cuda.synchronize()
    _base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _ = inferencer.infer(_warm_img)
    torch.cuda.synchronize()
    per_frame_mem_mb = (torch.cuda.max_memory_allocated() - _base) / 1024**2
    logger.info(f"[GPU-{gpu_id}] per-frame memory ≈ {per_frame_mem_mb:.0f} MB")
    del _warm_img

    # ── 主循环：持续从队列取任务 ───────────────────────────────────
    while True:
        task_data = task_queue.get()
        if task_data is None:  # poison pill → 退出
            break

        task_id = task_data["task_id"]
        logger.info(f"[GPU-{gpu_id}] 收到任务: {task_id}")
        conf_threshold = task_data.get("conf_threshold", 0.5)
        save_output = task_data.get("save_output", True)

        # ── 显存门控 ──────────────────────────────────────────────
        _free, _total = torch.cuda.mem_get_info(device)
        if _free < 8 * 1024**3:
            logger.warning(f"[GPU-{gpu_id}] task {task_id} 显存不足 "
                         f"(free={_free/1024**3:.1f}GB < 8GB), "
                         f"该 GPU 标记死亡，任务放回队列...")
            result_queue.put({
                "task_id": "__worker_died__",
                "gpu_id": gpu_id,
                "free_mem": _free,
            })
            result_queue.put({
                "task_id": task_id,
                "status": "retry",
                "video_path": "",
                "conf_threshold": conf_threshold,
            })
            task_queue.task_done()
            os._exit(1)

        try:
            # ── 自行读取视频帧 ──────────────────────────────────────
            video_path = task_data.get("video_path", "")
            if not video_path or not Path(video_path).exists():
                logger.error(f"[GPU-{gpu_id}] task {task_id}: 视频文件不存在 {video_path}")
                result_queue.put({
                    "task_id": task_id,
                    "status": "failed",
                    "error": f"视频文件不存在: {video_path}",
                })
                continue
            sys.path.insert(0, config_dict["project_root"])
            from video_utils import read_video_frames
            _t_read = time.perf_counter()
            frames_np = read_video_frames(video_path)
            n = len(frames_np)
            logger.info(f"[GPU-{gpu_id}] task {task_id}: "
                        f"读取 {n} 帧, "
                        f"耗时: {(time.perf_counter() - _t_read) * 1000:.0f}ms")

            # ── CUDA Stream 并行推理 ──────────────────────────────
            _free, _total = torch.cuda.mem_get_info(device)
            _safe_bytes = _free * safety_ratio
            _max_streams = max(1, int(_safe_bytes / (per_frame_mem_mb * 1024**2)))
            n_streams = min(num_streams, _max_streams, 8)
            logger.info(f"[GPU-{gpu_id}] task {task_id}: "
                        f"free={_free/1024**3:.1f}GB, streams={n_streams}")

            streams = [torch.cuda.Stream() for _ in range(n_streams)]
            import itertools
            stream_cycle = itertools.cycle(streams)

            frame_data: list[dict] = [None] * n
            _t_infer = time.perf_counter()
            for i, frame_np in enumerate(frames_np):
                pil_img = Image.fromarray(frame_np)
                s = next(stream_cycle)
                with torch.cuda.stream(s):
                    results = inferencer.infer(pil_img, return_raw_tensor=False)
                    frame_data[i] = {
                        "image": pil_img,
                        "results": results,
                        "frame_name": f"{i:06d}.jpg",
                    }

            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - _t_infer) * 1000
            del frames_np
            logger.info(f"[GPU-{gpu_id}] 任务 {task_id}: {n} frames "
                        f"推理完成, 耗时: {infer_ms:.0f}ms, "
                        f"平均: {infer_ms / max(n, 1):.1f}ms/帧")

            # ── 跨帧跟踪 ──────────────────────────────────────────
            try:
                import time as _time
                import supervision as sv
                from trackers import CBIoUTracker
                tracker = CBIoUTracker()
                # 日志：记录每帧的 track_id 匹配情况
                _t0 = _time.perf_counter()
                for fd in frame_data:
                    results = fd["results"]
                    track_results = [r for r in results if r.label == 1]
                    if not track_results:
                        empty_det = sv.Detections(
                            xyxy=np.empty((0, 4)),
                            confidence=np.empty((0,)),
                            class_id=np.empty((0,), dtype=int),
                        )
                        tracker.update(empty_det)
                        continue
                    xyxy = np.array([r.box for r in track_results])
                    conf = np.array([r.score for r in track_results])
                    cls = np.array([r.label for r in track_results])
                    detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
                    _t1 = _time.perf_counter()
                    tracked = tracker.update(detections)
                    _t2 = _time.perf_counter()

                    # 正确映射 track_id：tracked = detections[idx] 可能改变顺序，
                    # 不能使用 zip(track_results, tracked.tracker_id)。
                    # 改为用坐标匹配，确保每个检测框拿到正确的 track_id。
                    box_to_tid = {}
                    for tb, tid in zip(tracked.xyxy, tracked.tracker_id):
                        box_to_tid[tuple(tb)] = int(tid)
                    for r in track_results:
                        r.track_id = box_to_tid.get(tuple(r.box), -1)

                    # 日志：分析跟踪效果
                    assigned_ids = [r.track_id for r in track_results if r.track_id >= 0]
                    cur_ids = sorted(assigned_ids)
                _t3 = _time.perf_counter()
                logger.info(f"[GPU-{gpu_id}] 跟踪总耗时: {(_t3-_t0)*1000:.1f}ms / {len(frame_data)}帧")
            except Exception as e:
                logger.warning(f"[GPU-{gpu_id}] 跟踪失败（跳过）: {e}")

            # 阶段 2+3: RAG 决策 + 保存 + 增量更新 + 结果返回（一站式）
            # 委托给 streaming_cluster_updater.process_video_incremental
            from streaming_cluster_updater import process_video_incremental
            output_parent = Path(config_dict["output_dir"]) / "temp" / task_id
            summary = process_video_incremental(
                frame_data=frame_data,
                output_root=output_parent,
                draw_func=draw,
                updater=updater,
                gpu_id=gpu_id,
                save_output=save_output,
            )

            # 先发一个轻量的确认消息，再发完整 summary
            result_queue.put({
                "task_id": task_id,
                "status": "completed",
                "summary": {
                    "processed_frames": summary.get("processed_frames", 0),
                    "total_detections": summary.get("total_detections", 0),
                    "frames_kept": summary.get("frames_kept", 0),
                    "frames_warned": summary.get("frames_warned", 0),
                    "output_dir": summary.get("output_dir", ""),
                    "frame_details": summary.get("frame_details", []),
                },
            })

            logger.info(f"[GPU-{gpu_id}] 任务 {task_id} 完成, "
                        f"{summary['processed_frames']} frames, "
                        f"{summary['total_detections']} detections, "
                        f"keep={summary.get('frames_kept', 0)}, "
                        f"warn={summary.get('frames_warned', 0)}")

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[GPU-{gpu_id}] 任务 {task_id} OOM: {e}, "
                         f"该 GPU 标记死亡，任务放回队列...")
            _free_now, _total = torch.cuda.mem_get_info(device)
            # 清空缓存
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # 先发死亡信号，再发 retry，确保主进程先处理死亡
            result_queue.put({
                "task_id": "__worker_died__",
                "gpu_id": gpu_id,
                "free_mem": _free_now,
            })
            result_queue.put({
                "task_id": task_id,
                "status": "retry",
                "video_path": task_data.get("video_path"),
                "conf_threshold": task_data.get("conf_threshold", 0.5),
            })
            task_queue.task_done()
            os._exit(1)

        except Exception as e:
            logger.error(f"[GPU-{gpu_id}] 任务 {task_id} 失败: {e}")
            result_queue.put({
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
            })

        finally:
            # ── 释放 GPU 显存（PyTorch 缓存分配器 + GC） ────────────
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            task_queue.task_done()

    logger.info(f"[GPU-{gpu_id}] worker 退出")


# ══════════════════════════════════════════════════════════════════════
#  帧读取函数（在 ThreadPoolExecutor 中执行）
# ══════════════════════════════════════════════════════════════════════



# ==================== 数据模型 ====================
class TaskInfo(BaseModel):
    """任务信息模型"""
    task_id: str = Field(description="任务ID")
    filename: str = Field(description="原始文件名")
    status: str = Field(description="任务状态")
    progress: int = Field(description="进度百分比")
    message: str = Field(description="状态消息")
    created_at: str = Field(description="创建时间")
    started_at: Optional[str] = Field(description="开始时间", default=None)
    completed_at: Optional[str] = Field(description="完成时间", default=None)
    processing_time: Optional[float] = Field(description="处理时间(秒)", default=None)
    total_frames: Optional[int] = Field(description="总帧数", default=None)
    processed_frames: Optional[int] = Field(description="处理帧数", default=None)
    total_detections: Optional[int] = Field(description="总检测数", default=None)
    output_dir: Optional[str] = Field(description="输出目录", default=None)
    inference_log: Optional[str] = Field(description="推理过程原始日志", default=None)
    frame_details: Optional[list] = Field(description="每帧推理详情（决策、标签、相似度等）", default=None)

class VideoUploadResponse(BaseModel):
    """视频上传响应模型"""
    task_id: str = Field(description="任务ID")
    filename: str = Field(description="原始文件名")
    status: str = Field(description="任务状态")
    message: str = Field(description="状态消息")
    position_in_queue: int = Field(default=0, description="在队列中的位置")

class QueueStatusResponse(BaseModel):
    """队列状态响应模型"""
    queue_size: int = Field(description="队列中的任务数")
    max_queue_size: int = Field(description="队列最大容量")
    active_tasks: int = Field(description="正在处理的任务数")
    pending_tasks: int = Field(description="等待中的任务数")

# ==================== 辅助函数 ====================
def extract_frames_by_script(video_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    使用外部脚本抽帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 帧输出目录
    
    Returns:
        抽帧结果
    """
    if not config.EXTRACT_FRAMES_SCRIPT.exists():
        raise RuntimeError(f"抽帧脚本不存在: {config.EXTRACT_FRAMES_SCRIPT}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(config.EXTRACT_FRAMES_SCRIPT),
        str(video_path.parent),
        str(output_dir),
        str(config.FPS)
    ]
    
    logger.info(f"执行抽帧命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding='utf-8',
            errors='ignore',
            cwd=config.PROJECT_ROOT
        )
        
        # 改为 debug 级别，不输出到正常日志
        if result.stdout:
            logger.debug(f"抽帧脚本 stdout:\n{result.stdout[:2000]}")
        if result.stderr:
            logger.debug(f"抽帧脚本 stderr:\n{result.stderr[:2000]}")
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(f"抽帧失败 (exit={result.returncode}): {error_msg[:500]}")
        
        # 统计抽帧数量
        frame_count = len([f for f in output_dir.iterdir() if f.is_file() and f.suffix == '.jpg'])
        
        if frame_count == 0:
            raise RuntimeError(
                f"抽帧返回 0 帧。stdout:\n{result.stdout[:500]}\nstderr:\n{result.stderr[:500]}"
            )
        
        return {
            "status": "success",
            "frame_count": frame_count,
            "output_dir": str(output_dir)
        }
    
    except Exception as e:
        logger.error(f"抽帧异常: {e}")
        raise

def run_inference_script(frame_dir: Path, conf_threshold: float) -> Dict[str, Any]:
    """
    使用外部脚本执行推理
    
    Args:
        frame_dir: 帧目录路径
        conf_threshold: 置信度阈值
    
    Returns:
        推理结果
    """
    if not config.INFERENCE_SCRIPT.exists():
        raise RuntimeError(f"推理脚本不存在: {config.INFERENCE_SCRIPT}")
    
    if not config.CONFIG_PATH.exists():
        raise RuntimeError(f"配置文件不存在: {config.CONFIG_PATH}")
    
    if not config.CHECKPOINT_PATH.exists():
        raise RuntimeError(f"模型权重不存在: {config.CHECKPOINT_PATH}")
    
    # 构建命令（兼容用户指定的命令格式）
    cmd = [
        sys.executable,
        str(config.INFERENCE_SCRIPT),
        "-c", str(config.CONFIG_PATH),
        "-r", str(config.CHECKPOINT_PATH),
        "-i", str(frame_dir),
        "-t", str(conf_threshold)
    ]
    
    logger.info(f"执行推理命令: {' '.join(cmd)}")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '7'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding='utf-8',
            errors='ignore',
            cwd=config.PROJECT_ROOT,
            env=env,
            # timeout=300  # 5分钟超时
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            logger.error(f"推理失败: {error_msg}")
            raise RuntimeError(f"推理失败: {error_msg[:500]}")
        
        # 拆解标准输出中的关键信息
        raw_stdout = result.stdout or ""
        
        # 检查输出目录（模型脚本应该在输入目录下创建output文件夹）
        output_dir = frame_dir / "output"
        if output_dir.exists():
            # 统计检测结果数量
            txt_files = list(output_dir.glob("*.txt"))
            total_detections = 0
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                total_detections += 1
                except:
                    continue
            
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "total_detections": total_detections,
                "processed_frames": len(txt_files),
                "raw_stdout": raw_stdout,
            }
        
        return {
            "status": "success",
            "output_dir": str(frame_dir),
            "total_detections": 0,
            "processed_frames": 0,
            "raw_stdout": raw_stdout,
        }
    
    except subprocess.TimeoutExpired:
        raise RuntimeError("推理超时")
    except Exception as e:
        logger.error(f"推理异常: {e}")
        raise

def clean_temp_files(video_path: Path, frame_dir: Path):
    """
    清理临时文件
    
    Args:
        video_path: 视频文件路径
        frame_dir: 帧目录路径
    """
    try:
        # 删除视频文件
        if video_path.exists():
            video_path.unlink()
            logger.debug(f"已删除视频文件: {video_path}")
        
        # 删除帧目录
        if frame_dir.exists() and frame_dir.is_dir():
            shutil.rmtree(frame_dir)
            logger.debug(f"已删除帧目录: {frame_dir}")
        
    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")

def parse_torch_inference_stdout(raw_stdout: str) -> list[Dict[str, Any]]:
    """
    解析 torch_inf.py 的标准输出，提取每帧的结构化信息。

    stdout 格式（RAG 模式）:
        [RAG] mean sim 0.85 >= thr 0.75 → keep
               obj label=0 score=0.92 → cluster sim=0.87
        Saved original to ...
        Saved result to ...
        [RAG] Detected 2 instances, 2 unique labels ([0, 1]), avg confidence: 0.88 -> saved to output

    stdout 格式（static 模式）:
        Saved original to ...
        Saved result to ...
        [static] Detected 3 instances, 3 unique labels ([0, 1, 2]), avg confidence: 0.82 -> saved to output

    Returns:
        [ { "frame": "xxx.jpg", "decision": "keep"/"warn",
            "decision_tag": "RAG"/"static", "save_location": "output"/"warning",
            "num_instances": int, "unique_labels": [int,...],
            "avg_confidence": float, "per_object": [{"label":..., "score":..., "similarity":...}] }, ... ]
    """
    import re
    frames = []
    # 按 "Detected ... -> saved to" 分帧：每帧输出以该行结束
    blocks = re.split(r'(?=\[.*\] Detected \d+)', raw_stdout.strip())
    for block in blocks:
        if not block.strip():
            continue
        info: Dict[str, Any] = {}

        # 1. 决策标签: [RAG] 或 [static]
        m = re.search(r'^\[(\w+)\]', block)
        info['decision_tag'] = m.group(1) if m else "unknown"

        # 2. 决策结果: keep / warn
        m = re.search(r'(keep|warn)', block)
        info['decision'] = m.group(1) if m else "unknown"

        # 3. 保存位置
        m = re.search(r'saved to (\w+)', block)
        info['save_location'] = m.group(1) if m else "unknown"

        # 4. 实例数 + 唯一类别 + 平均置信度
        # [RAG] Detected 2 instances, 2 unique labels ([0, 1]), avg confidence: 0.88 -> saved to output
        m = re.search(
            r'Detected (\d+) instances, (\d+) unique labels \((\[.*?\])\), avg confidence: ([\d.]+)',
            block,
        )
        if m:
            info['num_instances'] = int(m.group(1))
            info['num_unique_labels'] = int(m.group(2))
            # 解析标签列表 [0, 1]
            try:
                info['unique_labels'] = json.loads(m.group(3))
            except Exception:
                info['unique_labels'] = []
            info['avg_confidence'] = float(m.group(4))

        # 5. 每帧文件名: 从 "Saved result to" 行提取
        m = re.search(r'Saved result to .+[\\/](\S+\.\w+)', block)
        info['frame'] = m.group(1) if m else ""

        # 6. 每个检测对象的详情
        objs = []
        for line in block.split('\n'):
            m2 = re.search(
                r'obj label=(\d+) score=([\d.]+) → cluster sim=([\d.]+)',
                line,
            )
            if m2:
                objs.append({
                    "label": int(m2.group(1)),
                    "score": float(m2.group(2)),
                    "cluster_similarity": float(m2.group(3)),
                })
        if objs:
            info['per_object'] = objs

        # 7. 平均聚类相似度（RAG 模式）
        m = re.search(r'mean sim ([\d.]+)', block)
        if m:
            info['mean_similarity'] = float(m.group(1))

        frames.append(info)

    return frames

# ==================== API 路由 ====================
async def result_watcher():
    """持续从 gpu_result_queue 读取结果，更新 task_status 并清理临时文件。"""
    logger.info("[result_watcher] 协程已启动，开始监听 GPU 结果队列")
    loop_count = 0

    def _poll():
        """线程内用 timeout get，避免线程永久阻塞。"""
        try:
            return gpu_result_queue.get(timeout=1.0)
        except Exception as e:
            # 超时是正常现象，其他异常需要记录
            if not isinstance(e, __import__("queue").Empty):
                logger.error(f"[result_watcher] _poll 异常: {type(e).__name__}: {e}")
            return None

    while not _shutdown_event.is_set():
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                _result_executor, _poll,
            )
        except Exception as e:
            logger.error(f"[result_watcher] run_in_executor 异常: {type(e).__name__}: {e}")
            await asyncio.sleep(0.1)
            continue

        if result is None:
            loop_count += 1
            if loop_count % 60 == 0:  # 每 60 秒输出一次心跳
                logger.info(f"[result_watcher] 正在等待结果... ({loop_count}s)")
                # 同时检查队列状态
                try:
                    qsize = gpu_result_queue.qsize()
                    logger.info(f"[result_watcher] 结果队列大小: {qsize}")
                except Exception as qe:
                    logger.error(f"[result_watcher] 队列状态异常: {qe}")
            await asyncio.sleep(1)
            continue

        loop_count = 0
        task_id = result["task_id"]
        logger.info(f"[result_watcher] 收到结果: task_id={task_id}, status={result.get('status')}")

        # ── 检测 GPU worker 崩溃信号 ─────────────────────────────
        if task_id == "__worker_died__":
            gpu_id = result["gpu_id"]
            alive = [p for p in gpu_processes if p.is_alive()]
            logger.error(f"⚠ GPU worker (GPU-{gpu_id}) 因 OOM 退出, "
                         f"free_mem={result['free_mem']/1024**3:.1f}GB, "
                         f"剩余存活: {len(alive)}/{len(gpu_processes)}")
            if not alive:
                logger.critical("🚨 所有 GPU worker 均已退出！需重启服务")
            continue

        info = task_status.get(task_id)
        if info is None:
            logger.warning(f"[result_watcher] 未知 task_id: {task_id}，跳过")
            continue

        try:
            if result["status"] == "completed":
                summary = result.get("summary", {})
                info["processed_frames"] = summary.get("processed_frames", 0)
                info["total_detections"] = summary.get("total_detections", 0)
                info["output_dir"] = summary.get("output_dir")
                info["frame_details"] = summary.get("frame_details", [])
                info["progress"] = 95
                info["message"] = "推理完成"

                info["status"] = TaskStatus.COMPLETED
                info["completed_at"] = datetime.now().isoformat()
                info["processing_time"] = (
                    datetime.fromisoformat(info["completed_at"])
                    - datetime.fromisoformat(info["started_at"])
                ).total_seconds()
                info["progress"] = 100
                info["message"] = "处理完成"

                logger.info(f"✅ 任务完成: {task_id} | "
                            f"帧数: {summary.get('processed_frames', 0)} | "
                            f"检测数: {summary.get('total_detections', 0)}")

            elif result["status"] == "retry":
                logger.info(f"🔄 任务 {task_id} 显存不足，重新分配 GPU worker...")
                info = task_status.get(task_id)
                if info:
                    retry_data = {
                        "task_id": task_id,
                        "video_path": info.get("video_path"),
                        "conf_threshold": info.get("conf_threshold", config.CONF_THRESHOLD),
                        "save_output": info.get("save_output", config.SAVE_OUTPUT),
                    }
                    await asyncio.get_event_loop().run_in_executor(
                        None, gpu_task_queue.put, retry_data
                    )
                else:
                    logger.warning(f"任务 {task_id} 无状态信息，跳过重试")
                continue

            elif result["status"] == "failed":
                info["status"] = TaskStatus.FAILED
                info["completed_at"] = datetime.now().isoformat()
                info["message"] = result.get("error", "推理失败")
                logger.error(f"❌ 任务失败: {task_id} | 错误: {info['message']}")

            else:
                logger.warning(f"[result_watcher] 未知状态: task_id={task_id}, status={result.get('status')}")
        except Exception as e:
            logger.error(f"[result_watcher] 处理结果异常: task_id={task_id}, error={e}", exc_info=True)

# ==================== API 路由 ====================
@app.on_event("startup")
async def startup_event():
    """服务启动时执行"""
    logger.info("=" * 60)
    logger.info("EdgeCrafter API 服务启动中...")
    logger.info("=" * 60)
    
    # 创建输出目录
    config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── 初始化 GPU 进程间通信 ──────────────────────────────────────
    global gpu_task_queue, gpu_result_queue, gpu_processes
    gpu_task_queue = mp_ctx.JoinableQueue()
    gpu_result_queue = mp_ctx.Queue()
    
    # ── 为每张 GPU 启动多个 worker 进程（自动根据显存计算数量） ──
    cfg_dict = {
        "project_root": str(config.PROJECT_ROOT),
        "config_path": str(config.CONFIG_PATH),
        "checkpoint_path": str(config.CHECKPOINT_PATH),
        "thresh": config.CONF_THRESHOLD,
        "num_streams": config.NUM_STREAMS,
        "output_dir": str(config.OUTPUT_DIR),
        "rag_cache": str(config.RAG_CACHE) if config.RAG_CACHE else None,
        "rag_threshold": config.RAG_THRESHOLD,
    }
    gpu_processes = []
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

    for gpu_id in config.GPU_IDS:
        free_mb = _gpu_free.get(gpu_id, 0)
        n_workers = max(1, min(config.WORKERS_PER_GPU, free_mb // 10240))  # 每 10G 开 1 个，不超过 WORKERS_PER_GPU
        logger.info(f"🔍 GPU {gpu_id}: 空闲 {free_mb}MB → {n_workers} 个 worker")
        for w_idx in range(n_workers):
            p = mp_ctx.Process(
                target=gpu_worker_process,
                args=(gpu_id, gpu_task_queue, gpu_result_queue, cfg_dict),
                daemon=True,
                name=f"gpu-{gpu_id}-worker-{w_idx}",
            )
            p.start()
            gpu_processes.append(p)
            logger.info(f"🚀 GPU {gpu_id} worker-{w_idx} 已启动 (PID={p.pid})")
    
    logger.info(f"🚀 已启动 {len(gpu_processes)} 个 GPU worker")
    
    # ── 启动 result_watcher（监听 GPU 结果并更新状态） ──────────────
    asyncio.create_task(result_watcher(), name="result-watcher")
    
    logger.info("=" * 60)
    logger.info("✅ 服务启动成功！")
    logger.info(f"📖 API 文档: http://{config.HOST}:{config.PORT}/docs")
    logger.info(f"📖 ReDoc 文档: http://{config.HOST}:{config.PORT}/redoc")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时执行"""
    logger.info("正在关闭服务...")
    
    # 通知所有异步协程退出
    _shutdown_event.set()
    
    # ── 停止 GPU worker ───────────────────────────────────────────
    for p in gpu_processes:
        if p.is_alive():
            p.terminate()
    await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: [p.join(timeout=3) for p in gpu_processes if p.is_alive()],
    )
    for p in gpu_processes:
        if p.is_alive():
            p.kill()
            p.join(timeout=1)
    logger.info(f"已停止 {len(gpu_processes)} 个 GPU worker")
    
    executor.shutdown(wait=True)
    _result_executor.shutdown(wait=False)
    logger.info("服务已关闭")

@app.get("/", tags=["首页"])
async def root():
    """API 根路径 - 服务状态信息"""
    return {
        "service": "EdgeCrafter Detection API",
        "version": "2.0.0",
        "status": "running",
        "description": "EdgeCrafter 目标检测推理服务（异步队列处理）",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "infer/video": "/infer/video",
            "tasks/{task_id}": "/tasks/{task_id}",
            "tasks": "/tasks",
            "queue/status": "/queue/status"
        },
        "config": {
            "project_root": str(config.PROJECT_ROOT),
            "inference_script": str(config.INFERENCE_SCRIPT),
            "extract_frames_script": str(config.EXTRACT_FRAMES_SCRIPT),
            "classes": config.CLASSES,
            "conf_threshold": config.CONF_THRESHOLD,
            "fps": config.FPS,
            "max_queue_size": config.MAX_QUEUE_SIZE,
        }
    }

@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    script_exists = config.INFERENCE_SCRIPT.exists()
    config_exists = config.CONFIG_PATH.exists()
    checkpoint_exists = config.CHECKPOINT_PATH.exists()
    extract_script_exists = config.EXTRACT_FRAMES_SCRIPT.exists()
    
    return {
        "status": "healthy" if all([script_exists, config_exists, checkpoint_exists, extract_script_exists]) else "degraded",
        "inference_script_exists": script_exists,
        "config_exists": config_exists,
        "checkpoint_exists": checkpoint_exists,
        "extract_frames_script_exists": extract_script_exists,
        "queue": {
            "queue_size": gpu_task_queue.qsize() if gpu_task_queue else 0,
            "max_queue_size": config.MAX_QUEUE_SIZE,
            "is_full": gpu_task_queue.qsize() >= config.MAX_QUEUE_SIZE if gpu_task_queue else False,
            "active_tasks": sum(1 for t in task_status.values() if t.get('status') in ('processing', 'extracting', 'inferencing')),
            "pending_tasks": gpu_task_queue.qsize() if gpu_task_queue else 0
        },
        "config": {
            "max_upload_size_mb": config.MAX_UPLOAD_SIZE // 1024 // 1024 if config.MAX_UPLOAD_SIZE else "unlimited",
            "max_workers": config.MAX_WORKERS,
            "max_queue_size": config.MAX_QUEUE_SIZE
        }
    }

# ==================== 视频推理路由 ====================
@app.post("/infer/video", response_model=VideoUploadResponse, tags=["视频推理"])
async def infer_video(
    file: UploadFile = File(..., description="视频文件，支持 mp4, avi, mov, mkv"),
    conf_threshold: float = Query(config.CONF_THRESHOLD, description="置信度阈值")
):
    """
    视频推理接口 - 异步队列处理
    
    上传视频文件，添加到队列等待处理：
    1. 接收视频并保存到临时目录
    2. 创建任务并加入队列
    3. 返回任务ID，前端可通过任务ID查询状态
    
    - **file**: 视频文件（必填）
    - **conf_threshold**: 置信度阈值，默认0.5
    
    返回任务ID和队列位置
    """
    # 检查文件类型
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式: {file_ext}。支持的格式: {', '.join(allowed_extensions)}"
        )
    
    # 检查 GPU 队列是否已满
    try:
        gpu_qsize = gpu_task_queue.qsize()
        if gpu_qsize >= config.MAX_QUEUE_SIZE:
            active_count = sum(1 for t in task_status.values() if t.get('status') in ('processing', 'extracting', 'inferencing'))
            logger.warning(f"⛔ 队列已满，拒绝上传: {file.filename} | GPU队列: {gpu_qsize}/{config.MAX_QUEUE_SIZE} | 活跃: {active_count}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "队列已满，请稍后重试",
                    "queue_size": gpu_qsize,
                    "max_queue_size": config.MAX_QUEUE_SIZE,
                    "active_tasks": active_count,
                    "pending_tasks": gpu_qsize,
                    "message": f"当前队列容量已满 ({gpu_qsize}/{config.MAX_QUEUE_SIZE})，请等待正在处理的任务完成后重试"
                }
            )
    except NotImplementedError:
        pass  # qsize 在某些平台上不可用
    
    # 生成任务ID - 使用视频文件名作为基础，添加sanitize和唯一性后缀
    raw_name = Path(file.filename).stem
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_name)
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')
    if not safe_name:
        safe_name = "video"
    task_id = f"{safe_name}_{uuid.uuid4().hex[:8]}"
    
    # 创建任务临时目录
    task_temp_dir = config.OUTPUT_DIR / "temp" / task_id
    task_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存上传的视频文件
    video_path = task_temp_dir / f"input{file_ext}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # 记录任务状态
    task_status[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "status": TaskStatus.INFERENCING,
        "progress": 0,
        "message": "已添加到队列",
        "created_at": datetime.now().isoformat(),
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "processing_time": None,
        "total_frames": None,
        "processed_frames": None,
        "total_detections": None,
        "output_dir": None,
        "inference_log": None,
        "frame_details": None,
        "video_path": str(video_path),
        "conf_threshold": conf_threshold,
        "save_output": config.SAVE_OUTPUT,
    }
    
    # 直接派发给 GPU worker
    task_data = {
        "task_id": task_id,
        "video_path": str(video_path),
        "conf_threshold": conf_threshold,
        "save_output": config.SAVE_OUTPUT,
    }
    
    await asyncio.get_event_loop().run_in_executor(
        None, gpu_task_queue.put, task_data
    )
    
    logger.info(f"📥 任务加入 GPU 队列: {task_id} | 文件名: {file.filename}")
    
    return VideoUploadResponse(
        task_id=task_id,
        filename=file.filename,
        status=TaskStatus.INFERENCING,
        message="已添加到队列，等待处理",
    )

# ==================== 任务状态查询 ====================
@app.get("/tasks/{task_id}", response_model=TaskInfo, tags=["任务管理"])
async def get_task_status(task_id: str):
    """
    查询任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务状态信息
    """
    if task_id not in task_status:
        raise HTTPException(
            status_code=404,
            detail=f"任务不存在: {task_id}"
        )
    
    info = task_status[task_id]
    return TaskInfo(
        task_id=info['task_id'],
        filename=info['filename'],
        status=info['status'],
        progress=info['progress'],
        message=info['message'],
        created_at=info['created_at'],
        started_at=info.get('started_at'),
        completed_at=info.get('completed_at'),
        processing_time=info.get('processing_time'),
        total_frames=info.get('total_frames'),
        processed_frames=info.get('processed_frames'),
        total_detections=info.get('total_detections'),
        output_dir=info.get('output_dir'),
        inference_log=info.get('inference_log'),
        frame_details=info.get('frame_details'),
    )

@app.get("/tasks", tags=["任务管理"])
async def list_tasks(
    status: Optional[str] = Query(None, description="按状态过滤: pending/processing/completed/failed"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    列出所有任务
    
    Args:
        status: 按状态过滤
        limit: 返回数量
        offset: 偏移量
    
    Returns:
        任务列表
    """
    tasks = []
    for task_id, info in task_status.items():
        if status and info['status'] != status:
            continue
        
        tasks.append({
            "task_id": info['task_id'],
            "filename": info['filename'],
            "status": info['status'],
            "progress": info['progress'],
            "created_at": info['created_at'],
            "completed_at": info.get('completed_at'),
            "total_detections": info.get('total_detections')
        })
    
    # 按创建时间排序（最新的在前）
    tasks.sort(key=lambda x: x['created_at'], reverse=True)
    
    total = len(tasks)
    tasks = tasks[offset:offset + limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "tasks": tasks
    }

@app.delete("/tasks/{task_id}", tags=["任务管理"])
async def delete_task(task_id: str):
    """
    删除任务记录
    
    Args:
        task_id: 任务ID
    
    Returns:
        删除结果
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 如果任务正在处理中，不允许删除
    if task_status[task_id]['status'] in [TaskStatus.PROCESSING, TaskStatus.EXTRACTING, TaskStatus.INFERENCING]:
        raise HTTPException(status_code=400, detail="任务正在处理中，无法删除")
    
    del task_status[task_id]
    
    return {"message": f"任务记录已删除: {task_id}"}

# ==================== 队列状态 ====================
@app.get("/queue/status", response_model=QueueStatusResponse, tags=["队列管理"])
async def get_queue_status():
    """
    获取队列状态
    
    Returns:
        队列状态信息
    """
    active_count = sum(
        1 for info in task_status.values() 
        if info['status'] in [TaskStatus.PROCESSING, TaskStatus.EXTRACTING, TaskStatus.INFERENCING]
    )
    pending_count = sum(
        1 for info in task_status.values() 
        if info['status'] == TaskStatus.INFERENCING
    )
    
    gpu_qsize = gpu_task_queue.qsize() if gpu_task_queue else 0
    return QueueStatusResponse(
        queue_size=gpu_qsize,
        max_queue_size=config.MAX_QUEUE_SIZE,
        active_tasks=active_count,
        pending_tasks=pending_count
    )

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("   EdgeCrafter 推理 API 服务 v2.0")
    print("=" * 60)
    print(f"📂 项目路径: {config.PROJECT_ROOT}")
    print(f"📂 推理脚本: {config.INFERENCE_SCRIPT}")
    print(f"📂 抽帧脚本: {config.EXTRACT_FRAMES_SCRIPT}")
    print(f"📂 模型路径: {config.CHECKPOINT_PATH}")
    print(f"📂 输出路径: {config.OUTPUT_DIR}")
    print(f"🌐 服务地址: http://{config.HOST}:{config.PORT}")
    print(f"📖 API 文档: http://{config.HOST}:{config.PORT}/docs")
    print(f"🔄 最大队列大小: {config.MAX_QUEUE_SIZE}")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        access_log=True,
        timeout_keep_alive=600,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )