import cv2
import os
import time
import datetime
import threading
import numpy as np
import torch
import onnxruntime as ort
import concurrent.futures


# 预处理线程池：OpenCV 函数释放 GIL，多线程可并行
_PREPROCESS_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(os.cpu_count() or 4, 8),
    thread_name_prefix='preproc',
)


def _ts():
    """返回形如 15:35:18.260 的时间戳"""
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


# 归一化常量（模块级，避免每 crop 重复创建）
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_image(img, input_size=(192, 256)):
    """
    对输入图片进行预处理，使用保持宽高比的缩放 (Letterbox)，并在不足处填充 114。
    返回 (input_tensor, pad_x, pad_y, scale)。
    input_tensor 是 (1, C, H, W) float32，transpose 为 view 无额外分配。
    """
    orig_h, orig_w = img.shape[:2]
    target_w, target_h = input_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    # copyMakeBorder 替代 np.full + 赋值，单次调用完成填充
    pad_y = (target_h - new_h) // 2
    pad_x = (target_w - new_w) // 2
    top, bottom = pad_y, target_h - new_h - pad_y
    left, right = pad_x, target_w - new_w - pad_x
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )

    # BGR → RGB → float32 → 就地归一化（节省一次 float32 分配）
    img_float = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_float -= _MEAN
    img_float /= _STD

    # transpose 是 view，不产生新分配
    input_tensor = img_float.transpose(2, 0, 1)[np.newaxis, :]

    return input_tensor, pad_x, pad_y, scale


def _prep_one(args):
    """
    单 crop 预处理（线程池 worker）。
    pinned_buffer 不为 None 时直接写入对应槽位，返回 pad_info。
    """
    i, img, input_size, pinned_buffer = args
    t, pad_x, pad_y, scale = preprocess_image(img, input_size=input_size)
    if pinned_buffer is not None:
        pinned_buffer[i].copy_(torch.from_numpy(t[0]))
    return pad_x, pad_y, scale


def batch_preprocess_images(imgs, input_size=(288, 384), gpu_id=None, frame_idx=None,
                           pinned_buffer=None):
    """
    CPU 预处理：多线程并行，每个 crop 直接写入 pinned_buffer。
    返回 (batch_pinned, actual_n, pad_info_list)。

    pinned_buffer: 预分配锁页 tensor (MAX_BATCH, C, H, W)。
    """
    t0 = time.perf_counter()
    n = len(imgs)
    pad_info_list = []

    if pinned_buffer is not None:
        if n >= 4:
            # 线程池并行（OpenCV 释放 GIL，>4 crop 时有明显收益）
            work = [(i, img, input_size, pinned_buffer) for i, img in enumerate(imgs)]
            pad_info_list = list(_PREPROCESS_POOL.map(_prep_one, work))
        else:
            # 少量 crop 时串行更划算
            for i, img in enumerate(imgs):
                pad_x, pad_y, scale = _prep_one((i, img, input_size, pinned_buffer))
                pad_info_list.append((pad_x, pad_y, scale))
        batch_pinned = pinned_buffer[:n]
    else:
        # 回退路径：无预分配 buffer 时动态分配
        batch_list = []
        for img in imgs:
            t, pad_x, pad_y, scale = preprocess_image(img, input_size=input_size)
            batch_list.append(t)
            pad_info_list.append((pad_x, pad_y, scale))
        batch_np = np.concatenate(batch_list, axis=0)
        batch_pinned = torch.from_numpy(batch_np).pin_memory()

    dt = (time.perf_counter() - t0) * 1000
    tag = f"[GPU-{gpu_id}]" if gpu_id is not None else ""
    # print(f"[{_ts()}] [TIMING] {tag} frame {frame_idx}: batch_preprocess: {n} crops, {dt:.1f}ms")
    return batch_pinned, n, pad_info_list


def batch_postprocess(outputs, pad_info_list, offset_list, gpu_id=None, frame_idx=None):
    """
    CPU 阶段：将 batch 推理输出 ([simcc_x, simcc_y]) 转为每个 crop 的关键点（原图坐标）。
    """
    t0 = time.perf_counter()
    out1, out2 = outputs[0], outputs[1]
    if out1.shape[-1] > out2.shape[-1]:
        simcc_y, simcc_x = out1, out2
    else:
        simcc_x, simcc_y = out1, out2

    all_kps = []
    for i in range(simcc_x.shape[0]):
        pad_x, pad_y, scale = pad_info_list[i]
        x1, y1 = offset_list[i]

        sx = simcc_x[i]
        sy = simcc_y[i]

        max_x_pos = np.argmax(sx, axis=1)
        score_x = np.max(sx, axis=1)
        max_y_pos = np.argmax(sy, axis=1)
        score_y = np.max(sy, axis=1)

        pose_x = max_x_pos // 2
        pose_y = max_y_pos // 2
        scores = np.maximum(score_x, score_y)

        keypoints = []
        for px, py, s in zip(pose_x, pose_y, scores):
            px = (px - pad_x) / scale + x1
            py = (py - pad_y) / scale + y1
            keypoints.append([int(px), int(py), float(s)])

        body = keypoints[:23]
        left = keypoints[100] if len(keypoints) > 100 else [0, 0, 0.0]
        right = keypoints[121] if len(keypoints) > 121 else [0, 0, 0.0]
        all_kps.append(body + [left, right])

    dt = (time.perf_counter() - t0) * 1000
    tag = f"[GPU-{gpu_id}]" if gpu_id is not None else ""
    # print(f"[{_ts()}] [TIMING] {tag} frame {frame_idx}: batch_postprocess: {len(all_kps)} people, {dt:.1f}ms")
    return all_kps


# class RTMPoseModel:
#     """同步推理（无流水线），保留兼容。"""
#     def __init__(self, model_path: str, input_size: tuple = (288, 384), device_id: int = 2):
#         self.model_path = model_path
#         self.input_size = input_size
#         self.device_id = device_id

#         providers = [
#             'TensorrtExecutionProvider',
#             'CUDAExecutionProvider',
#             'CPUExecutionProvider',
#         ]
#         self.session = ort.InferenceSession(model_path, providers=providers)
#         self.input_name = self.session.get_inputs()[0].name
#         actual = self.session.get_providers()
#         print(f"RTMPoseModel 加载完成, 输入: {self.input_name}, Provider: {actual}", flush=True)

#     def inference_batch(self, batch_np, frame_idx=None):
#         t0 = time.perf_counter()
#         outputs = self.session.run(None, {self.input_name: batch_np})
#         dt = (time.perf_counter() - t0) * 1000
#         print(f"[{_ts()}] [TIMING] [GPU-{self.device_id}] frame {frame_idx}: inference_batch: "
#               f"batch_size={batch_np.shape[0]}, {dt:.2f}ms")
#         return outputs


class RTMPoseModelStream:
    """
    异步推理模型：submit() 提交 → CPU 继续工作 → wait() 取结果。
    CUDA Stream 上 H2D → 推理 → D2H 全部异步，CPU 不阻塞。

    所有 buffer（CPU 锁页 / GPU 显存）在 __init__ 阶段一次性预分配，
    运行时零内存分配，避免因系统内存整理导致 enqueue 抖动。
    """
    MAX_BATCH = 256  # 预分配最大 batch 大小（超出时会回退到动态分配）

    def __init__(self, model_path: str, input_size: tuple = (288, 384), device_id: int = 2):
        self.device = torch.device(f'cuda:{device_id}')
        self.input_size = input_size
        self.device_id = device_id

        # 1. 专用 CUDA Stream
        self.stream = torch.cuda.Stream(device=self.device)

        # 2. ORT Session（不绑定 user_compute_stream，避免多上下文下 IO binding 失败）
        provider_opts = {
            'device_id': device_id,
        }
        self.session = ort.InferenceSession(
            model_path,
            providers=[('CUDAExecutionProvider', provider_opts), 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 3. 用一次 warmup 推理发现真实输出 shape（session.get_outputs() 在动态模型中不可靠）
        dummy_np = np.zeros((1, 3, input_size[1], input_size[0]), dtype=np.float32)
        warmup_outs = self.session.run(None, {self.input_name: dummy_np})
        self.output_shapes = [o.shape for o in warmup_outs]

        # 4. 静态预分配所有 buffer
        B = self.MAX_BATCH
        C, H, W = 3, input_size[1], input_size[0]

        # CPU 锁页输入 buffer
        self._input_pinned = torch.empty((B, C, H, W), pin_memory=True)

        # GPU 输入 buffer
        self._gpu_input_buf = torch.empty((B, C, H, W), device=self.device)

        # GPU 输出 + CPU 锁页输出 buffers
        self._gpu_out_bufs = []
        self._cpu_out_pinned = []
        for static_shape in self.output_shapes:
            shape = (B,) + static_shape[1:]
            self._gpu_out_bufs.append(
                torch.empty(shape, dtype=torch.float32, device=self.device)
            )
            self._cpu_out_pinned.append(
                torch.empty(shape, pin_memory=True)
            )

        # 5. 复用 io_binding，避免每次 submit 重新创建导致内存池碎片化
        self.io = self.session.io_binding()

        actual = self.session.get_providers()
        # print(f"RTMPoseModelStream 加载完成, 输入: {self.input_name}, "
        #       f"输出 shapes: {self.output_shapes}, Provider: {actual}", flush=True)

    def submit(self, batch_pinned, frame_idx=None):
        """
        异步提交 batch 推理。
        batch_pinned: pinned tensor (B, C, H, W)，B ≤ MAX_BATCH
        返回: cpu_tensors 列表 — 此时数据**还未就绪**，调用 wait() 后可用。
        """
        t0 = time.perf_counter()
        batch_size = batch_pinned.shape[0]

        with torch.cuda.stream(self.stream):
            # ─── H2D 异步拷贝（仅实际 batch 大小） ───
            self._gpu_input_buf[:batch_size].copy_(batch_pinned, non_blocking=True)

            gpu_input = self._gpu_input_buf[:batch_size]
            self.io.bind_input(
                self.input_name, 'cuda', self.device_id,
                np.float32, tuple(gpu_input.shape), gpu_input.data_ptr(),
            )

            # ─── 输出绑定 + D2H（仅实际 batch 大小） ───
            cpu_tensors = []
            for i, name in enumerate(self.output_names):
                gbuf = self._gpu_out_bufs[i][:batch_size]
                cbuf = self._cpu_out_pinned[i][:batch_size]
                self.io.bind_output(
                    name, 'cuda', self.device_id,
                    np.float32, tuple(gbuf.shape), gbuf.data_ptr(),
                )
                cpu_tensors.append(cbuf)

            # ─── 推理 ───
            self.session.run_with_iobinding(self.io)

            # ─── D2H 异步拷贝 ───
            for i in range(len(self.output_names)):
                self._cpu_out_pinned[i][:batch_size].copy_(
                    self._gpu_out_bufs[i][:batch_size], non_blocking=True
                )

        dt = (time.perf_counter() - t0) * 1000
        fid = f"frame {frame_idx}" if frame_idx is not None else ""
        # print(f"[{_ts()}] [TIMING] [GPU-{self.device_id}] {fid} submit: batch_size={batch_size}, enqueue={dt:.1f}ms", flush=True)
        return cpu_tensors

    def wait(self, cpu_tensors, frame_idx=None):
        """
        等待所有 Stream 完成（H2D → 推理 → D2H 全部完成）。
        返回 numpy 列表，可传入 batch_postprocess。
        """
        t0 = time.perf_counter()
        # 同步自定义 stream（H2D / D2H）
        self.stream.synchronize()
        # 同步默认 stream（ORT 推理，因未绑 user_compute_stream）
        torch.cuda.synchronize(device=self.device)
        dt = (time.perf_counter() - t0) * 1000
        fid = f"frame {frame_idx}" if frame_idx is not None else ""
        # print(f"[{_ts()}] [TIMING] [GPU-{self.device_id}] {fid} wait: {dt:.1f}ms", flush=True)
        return [t.numpy() for t in cpu_tensors]