"""视频帧读取工具 — 用 decord 批量读取视频帧。"""

import numpy as np


def read_video_frames(video_path: str, target_fps: float = 25.0,
                      skip_seconds: float = 3.0, max_duration: float = 100000.0,
                      return_indices: bool = False,
                      chunk_size: int = 0, chunk_index: int = 0):
    """用 decord 读取视频帧。

    Parameters
    ----------
    video_path : str
        视频文件路径。
    target_fps : float
        目标采样帧率（默认 25fps）。
    skip_seconds : float
        跳过视频前 N 秒（默认 3 秒，避免片头黑场）。
    max_duration : float
        最多提取多少秒的帧（默认 100000 秒）。
    return_indices : bool
        是否同时返回帧索引列表（用于与原帧号映射）。
    chunk_size : int
        分块大小，一次读取多少帧（默认 0，读取全部帧）。
    chunk_index : int
        从第几个 chunk 开始读取（从 0 开始）。

    Returns
    -------
    np.ndarray or tuple
        默认返回 shape (N, H, W, 3) RGB uint8。
        若 return_indices=True，返回 (frames, indices)。
        若 chunk_size > 0，额外返回 (total_frames, total_chunks)。
    """
    import decord
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path)
    n_total = len(vr)
    video_fps = vr.get_avg_fps() if hasattr(vr, 'get_avg_fps') else 30.0
    frame_interval = max(1, int(round(video_fps / target_fps)))
    start_index = int(skip_seconds * video_fps)
    end_index = min(n_total, int((skip_seconds + max_duration) * video_fps))
    abs_indices = list(range(start_index, end_index, frame_interval))

    total_frames = len(abs_indices)
    total_chunks = 1

    if chunk_size > 0:
        total_chunks = (total_frames + chunk_size - 1) // chunk_size
        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_frames)
        if chunk_start >= total_frames:
            frames = []
            rel_indices = []
        else:
            selected_abs = abs_indices[chunk_start:chunk_end]
            frames = vr.get_batch(selected_abs).asnumpy()
            rel_indices = list(range(chunk_start, chunk_end))
    else:
        frames = vr.get_batch(abs_indices).asnumpy()
        rel_indices = list(range(total_frames))

    del vr

    result = []
    if return_indices:
        result.extend([frames, rel_indices])
    else:
        result.append(frames)

    if chunk_size > 0:
        result.extend([total_frames, total_chunks])

    if len(result) == 1:
        return result[0]
    return tuple(result)


# ============================================================
#  ffmpeg 抽帧（支持 NVIDIA GPU 解码，替代 decord）
# ============================================================

def _yuv420p_to_rgb_numpy(yuv: np.ndarray, h: int, w: int,
                          full_range: bool = True) -> np.ndarray:
    """纯 numpy 的 YUV420P → RGB 转换（CPU fallback，慢）。

    yuv shape: (N, H*W*3//2)（每帧按字节偏移：Y=H*W, U/V 各 1/4）。
    """
    import numpy as np
    n = yuv.shape[0]
    hw = h * w
    u_size = hw // 4
    y = yuv[:, :hw].astype(np.float32).reshape(n, h, w)
    u = yuv[:, hw:hw + u_size].astype(np.float32).reshape(n, h // 2, w // 2)
    v = yuv[:, hw + u_size:hw + u_size * 2].astype(np.float32).reshape(n, h // 2, w // 2)
    u = u.repeat(2, axis=1).repeat(2, axis=2)
    v = v.repeat(2, axis=1).repeat(2, axis=2)
    if full_range:
        r = y + 1.402 * (v - 128.0)
        g = y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)
        b = y + 1.772 * (u - 128.0)
    else:
        # limited range (BT.601): 先扩展 YUV 到 0-255
        y = (y - 16.0) * 255.0 / 219.0
        u = (u - 128.0) * 255.0 / 224.0
        v = (v - 128.0) * 255.0 / 224.0
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _yuv420p_to_rgb_torch(yuv: np.ndarray, h: int, w: int,
                          full_range: bool = True) -> np.ndarray:
    """torch GPU 的 YUV420P → RGB 转换（快）。

    yuv shape: (N, H*W*3//2)。整块一次 H2D，GPU 侧切片/计算，减少 PCIe 小传输。
    """
    import torch
    import torch.nn.functional as F
    n = yuv.shape[0]
    hw = h * w
    u_size = hw // 4
    device = torch.cuda.current_device()

    # 一次 H2D（uint8）；.float() 产物为连续张量，.reshape 无需拷贝
    yuv_g = torch.from_numpy(yuv).to(device)
    y = yuv_g[:, :hw].float().reshape(n, h, w)
    u = yuv_g[:, hw:hw + u_size].float().reshape(n, h // 2, w // 2)
    v = yuv_g[:, hw + u_size:hw + u_size * 2].float().reshape(n, h // 2, w // 2)
    u = F.interpolate(u[:, None], scale_factor=2, mode="nearest")[:, 0]
    v = F.interpolate(v[:, None], scale_factor=2, mode="nearest")[:, 0]

    if full_range:
        r = y + 1.402 * (v - 128.0)
        g = y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)
        b = y + 1.772 * (u - 128.0)
    else:
        y = (y - 16.0) * 255.0 / 219.0
        u = (u - 128.0) * 255.0 / 224.0
        v = (v - 128.0) * 255.0 / 224.0
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u

    rgb = torch.stack([r, g, b], dim=-1).clamp(0, 255).to(torch.uint8)
    out = rgb.cpu().numpy()
    return out


def _decode_yuv_frames(cmd, w, h, full_range, convert, n_total_est):
    """执行 ffmpeg 命令，流式读取 yuv420p rawvideo 并分块转换为 RGB。

    返回 (frames, err, ret)。流水线设计：
    - 后台读线程持续 drain stdout（避免管道短读/阻塞）；
    - 主线程并行做 GPU 转换，与 ffmpeg 解码重叠；
    - 按 ffprobe 估算帧数预分配输出数组，避免末尾 np.concatenate 大拷贝；
    - stderr 由独立线程 drain，防止管道写满死锁。
    """
    import queue
    import subprocess
    import threading

    frame_bytes = w * h * 3 // 2
    chunk_frames = 128
    chunk_bytes = frame_bytes * chunk_frames

    cap = max(64, n_total_est)
    frames = np.empty((cap, h, w, 3), dtype=np.uint8)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=1 << 20)
    q = queue.Queue(maxsize=2)
    err_chunks = []

    def reader():
        try:
            while True:
                raw = proc.stdout.read(chunk_bytes)  # BufferedReader 内部累积，无短读问题
                if not raw:
                    break
                q.put(raw)
        except Exception as e:
            q.put(("ERR", e))
        finally:
            q.put(None)

    def err_reader():
        err_chunks.append(proc.stderr.read())

    t = threading.Thread(target=reader, daemon=True)
    t_err = threading.Thread(target=err_reader, daemon=True)
    t.start()
    t_err.start()

    n_read = 0
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, tuple) and item[0] == "ERR":
            raise item[1]
        n = len(item) // frame_bytes
        if n == 0:
            continue
        # offset 视图无拷贝；.copy() 让数组可写（torch.from_numpy 需要）
        yuv = np.frombuffer(item, dtype=np.uint8, count=n * frame_bytes).reshape(n, frame_bytes).copy()
        if n_read + n > cap:  # 估算不足，扩容（罕见）
            extra = np.empty((n_read + n - cap, h, w, 3), dtype=np.uint8)
            frames = np.concatenate([frames, extra], axis=0)
            cap = len(frames)
        frames[n_read:n_read + n] = convert(yuv, h, w, full_range)
        n_read += n

    t.join()
    t_err.join()
    err = b"".join(err_chunks).decode('utf-8', errors='ignore')
    proc.stdout.close()
    proc.stderr.close()
    ret = proc.wait()
    return frames[:n_read], err, ret


def read_video_frames_ffmpeg(video_path: str, target_fps: float = 25.0,
                             skip_seconds: float = 3.0, max_duration: float = 100000.0,
                             return_indices: bool = False,
                             chunk_size: int = 0, chunk_index: int = 0,
                             use_cuda: bool = True):
    """用 ffmpeg 抽帧（优先 NVIDIA GPU 解码，替代 decord）。

    流程：ffmpeg(-hwaccel cuda) 解码 + fps 采样 → yuv420p rawvideo 到 pipe
          → torch GPU 转 RGB（GPU 不可用时 numpy CPU fallback）。

    参数与 read_video_frames 完全一致，返回值格式也一致。
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg/ffprobe 不可用")

    # 1. ffprobe 获取分辨率 / 帧率 / 像素格式
    probe = subprocess.run(
        [ffprobe, '-v', 'error',
         '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,pix_fmt',
         '-of', 'csv=p=0', video_path],
        capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(f"ffprobe 失败: {probe.stderr.strip()}")
    w_s, h_s, pix_fmt = probe.stdout.strip().split(',')
    w, h = int(w_s), int(h_s)
    # 源为 full range(yuvj*) 时输出 yuvj420p 保持范围（无重标定损耗），否则 yuv420p(limited)。
    # 注意：full_range 标志必须与【输出】像素格式一致，不能按源格式判断。
    full_range = pix_fmt.startswith("yuvj")
    out_pix_fmt = "yuvj420p" if full_range else "yuv420p"

    # 用视频时长估算帧数（预分配输出数组，避免末尾大拷贝）
    dur_probe = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', video_path],
        capture_output=True, text=True)
    try:
        duration = float(dur_probe.stdout.strip())
    except ValueError:
        duration = 0.0
    usable = min(max(duration - skip_seconds, 0.0), max_duration)
    n_total_est = int(usable * target_fps) + 4

    # 2. ffmpeg 解码抽帧 → rawvideo 到 pipe（流水线分块转换）
    base_cmd = [ffmpeg, '-v', 'error', '-ss', str(skip_seconds),
                '-i', video_path, '-t', str(max_duration),
                '-vf', f'fps={target_fps}',
                '-f', 'rawvideo', '-pix_fmt', out_pix_fmt, 'pipe:1']
    cuda_cmd = [ffmpeg, '-v', 'error', '-hwaccel', 'cuda',
                '-ss', str(skip_seconds), '-i', video_path, '-t', str(max_duration),
                '-vf', f'fps={target_fps}',
                '-f', 'rawvideo', '-pix_fmt', out_pix_fmt, 'pipe:1']

    err, ret = "", -1
    if use_cuda:
        try:
            frames, err, ret = _decode_yuv_frames(
                cuda_cmd, w, h, full_range, _yuv420p_to_rgb_torch, n_total_est)
        except Exception as e:
            err = f"cuda 转换失败: {e}"
            ret = -1
        # cuda 失败 → 回退软解 + numpy 转换
        if ret != 0:
            frames, err, ret = _decode_yuv_frames(
                base_cmd, w, h, full_range, _yuv420p_to_rgb_numpy, n_total_est)
    else:
        frames, err, ret = _decode_yuv_frames(
            base_cmd, w, h, full_range, _yuv420p_to_rgb_numpy, n_total_est)

    if ret != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败(ret={ret}): {err[:500]}")

    # 3. 与 read_video_frames 一致的 chunk/索引逻辑
    n_total = len(frames)
    total_frames = n_total
    total_chunks = 1
    if chunk_size > 0:
        total_chunks = (total_frames + chunk_size - 1) // chunk_size
        cs = chunk_index * chunk_size
        ce = min(cs + chunk_size, total_frames)
        frames = frames[cs:ce]
        rel_indices = list(range(cs, ce))
    else:
        rel_indices = list(range(total_frames))

    result = []
    if return_indices:
        result.append(frames)
        result.append(rel_indices)
    else:
        result.append(frames)
    if chunk_size > 0:
        result.extend([total_frames, total_chunks])
    if len(result) == 1:
        return result[0]
    return tuple(result)

