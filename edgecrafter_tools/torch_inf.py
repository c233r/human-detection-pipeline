"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import concurrent.futures
import json
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from engine.core import YAMLConfig
from engine.data.dataset.coco_dataset import mscoco_label2name_remap80


@dataclass
class Result:
    label: int
    score: float
    box: np.ndarray
    mask: np.ndarray = None
    track_id: int = -1


COCO_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128),
    (255, 128, 255), (128, 255, 255), (192, 0, 0), (0, 192, 0), (0, 0, 192),
    (192, 192, 0), (192, 0, 192), (0, 192, 192), (255, 192, 0), (255, 0, 192),
    (0, 255, 192), (192, 255, 0), (255, 192, 128), (192, 255, 128), (128, 192, 255),
    (255, 128, 192), (128, 255, 192), (192, 128, 255), (255, 192, 192), (192, 255, 192),
    (192, 192, 255), (255, 255, 192), (255, 192, 255), (192, 255, 255), (64, 0, 0),
    (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), (128, 64, 0),
    (128, 0, 64), (0, 128, 64), (64, 128, 0), (128, 64, 128), (64, 128, 128), (128, 128, 64),
    (192, 64, 0), (192, 0, 64), (0, 192, 64), (64, 192, 0), (192, 64, 192), (64, 192, 192),
    (192, 192, 64), (255, 64, 0), (255, 0, 64), (0, 255, 64), (64, 255, 0),
    (255, 64, 128), (64, 255, 128), (128, 64, 255), (255, 128, 64), (128, 255, 64),
    (64, 128, 255), (192, 64, 128), (192, 128, 64), (64, 192, 128), (128, 192, 64),
    (64, 128, 192), (128, 64, 192), (192, 128, 192), (128, 192, 192), (192, 192, 128),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BOUNDARY_KERNEL_CACHE: dict[int, np.ndarray] = {}


def get_class_color(label: int):
    return COCO_COLORS[label % len(COCO_COLORS)]


def get_draw_params(image_shape: tuple[int, int, int]):
    height, width = image_shape[:2]
    min_side = max(1, min(height, width))
    base = min_side / 640.0
    font_scale = max(0.9, 0.95 * base)
    text_thickness = max(2, int(round(1.6 * base)))
    box_thickness = max(2, int(round(2.0 * base)))
    boundary_thickness = max(2, int(round(2.0 * base)))
    return font_scale, text_thickness, box_thickness, boundary_thickness


def draw_white_boundary_fast(image: np.ndarray, mask: np.ndarray, thickness: int = 2):
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return

    k = max(1, int(thickness))
    kernel = _BOUNDARY_KERNEL_CACHE.get(k)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        _BOUNDARY_KERNEL_CACHE[k] = kernel
    edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    image[edge > 0] = (255, 255, 255)


def draw_to_numpy(image: Image.Image, results: list[Result], alpha: float = 0.5):
    im_np = np.array(image, copy=True)
    font_scale, text_thickness, box_thickness, boundary_thickness = get_draw_params(im_np.shape)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if results and alpha > 0:
        overlay = im_np.copy()
        any_mask = False
        merged_mask = np.zeros(im_np.shape[:2], dtype=np.uint8)
        count = 0
        for res in results:
            count += 1
            if res.mask is None:
                continue
            mask_bool = res.mask.astype(bool, copy=False)
            if not np.any(mask_bool):
                continue
            any_mask = True
            color_rgb = get_class_color(res.label)
            overlay[mask_bool] = color_rgb
            merged_mask[mask_bool] = 1

        if any_mask:
            im_np = cv2.addWeighted(im_np, 1.0 - alpha, overlay, alpha, 0)
            draw_white_boundary_fast(im_np, merged_mask, thickness=boundary_thickness)

    for res in results:
        color_rgb = get_class_color(res.label)
        x1, y1, x2, y2 = res.box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(im_np.shape[1] - 1, x2)
        y2 = min(im_np.shape[0] - 1, y2)

        cv2.rectangle(im_np, (x1, y1), (x2, y2), color_rgb, box_thickness)
        text = f"{mscoco_label2name_remap80.get(res.label, str(res.label))} {res.score:.2f}"
        if res.track_id >= 0:
            text = f"#{res.track_id} {text}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = x1
        text_y = max(th + baseline + 2, y1)
        cv2.rectangle(
            im_np,
            (text_x, text_y - th - baseline - 4),
            (text_x + tw + 4, text_y + 2),
            color_rgb,
            -1,
        )
        cv2.putText(
            im_np,
            text,
            (text_x + 2, text_y - baseline - 1),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return im_np


def draw(image: Image.Image, results: list[Result], alpha: float = 0.4, label_filter: list[int] | None = None):
    if label_filter is not None:
        results = [r for r in results if r.label in label_filter]
    return Image.fromarray(draw_to_numpy(image, results, alpha=alpha).astype(np.uint8))


class ECInferencer:
    """统一推理器：自动检测 ONNX，可用则走 ONNX 路径，否则走 PyTorch 路径。

    对外接口 infer / infer_tensor / infer_batch / infer_images_batch 签名不变，
    内部根据 self.use_onnx 自动分流到对应实现。
    """

    # ONNX 固定输入尺寸
    ONNX_TARGET_H = 1080
    ONNX_TARGET_W = 1920

    def __init__(self, model, task, device, size, thresh, onnx_path=None,
                 compile_mode=None, square_size=None):
        self.task = task
        self.device = device
        self.size = size
        self.thresh = thresh
        # PyTorch 路径：推理输入拉伸到 square_size×square_size 正方形，
        # 与训练(Resize 到 2048×2048 正方形拉伸)分布一致，消除横/竖图输入导致的框偏移。
        # 拉伸保持归一化坐标不变，故 box 还原基准直接用原图尺寸即可。
        self.square_size = square_size

        # ONNX 路径仅在显式传入 onnx_path 时启用；
        # 默认走 PyTorch FP16（实测最快）。
        self.use_onnx = onnx_path is not None and os.path.exists(onnx_path)

        if self.use_onnx:
            # ── ONNX 路径：不初始化 pth model ──
            self.model = None
            self.onnx_path = onnx_path
            self._build_onnx_session(onnx_path)
            # 复用 transform（带 Resize 到 1080x1920）
            self.transforms = self._build_resize_transforms(
                self.ONNX_TARGET_H, self.ONNX_TARGET_W)
            self._gpu_transforms = self._build_gpu_resize_transforms(
                self.ONNX_TARGET_H, self.ONNX_TARGET_W)
            print(f"[ECInferencer] 使用 ONNX 推理: {onnx_path}")
        else:
            # ── PyTorch 路径 ──
            self.model = model
            self.onnx_path = None
            # 复用 transform（无 Resize，支持动态尺寸）
            self.transforms = self._build_transforms()
            self._gpu_transforms = self._build_gpu_transforms()
            # torch.compile 融合优化（对 Transformer/ViT 效果通常不错）
            # compile_mode 可选 "reduce-overhead" / "default" / "max-autotune"；
            # suppress_errors=True: 允许编译失败的部分静默回退 eager，不抛异常；
            # fullgraph=False: 不强制全图编译，进一步放宽限制；
            # 首次调用有编译延迟，默认 None（关闭），需要时由调用方显式开启。
            if compile_mode:
                try:
                    # 关键配置：忽略所有编译错误，回退到 eager 执行
                    import logging as _logging
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                    # 抑制 torch.compile 内部的 Warning 日志（如 Identity 不支持等）
                    torch._logging.set_logs(all=_logging.ERROR)
                    
                    self.model = torch.compile(
                        self.model, mode=compile_mode,
                        fullgraph=False,  # 关键：不强制全图编译，遇到不支持的算子自动回退 eager
                        dynamic=False,    # 固定 shape 优化（你的模型输入尺寸固定，可开）
                    )
                    print(f"[ECInferencer] PyTorch 模型已启用 torch.compile(mode={compile_mode}, suppress_errors=True)")
                except Exception as e:
                    print(f"[ECInferencer] torch.compile 失败，继续使用原模型: {e}")
            else:
                print(f"[ECInferencer] 使用 PyTorch 推理"
                      + (f"，输入拉伸到 {square_size}×{square_size} 正方形（与训练分布一致）"
                         if square_size else ""))

    # ════════════════════════════════════════════════════════════
    #  Transform 构建
    # ════════════════════════════════════════════════════════════

    def _build_transforms(self):
        """PyTorch 路径 CPU transform（无 Resize，支持动态尺寸）。"""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_resize_transforms(self, target_h, target_w):
        """ONNX 路径 CPU transform（带 Resize 到固定尺寸）。"""
        return T.Compose([
            T.Resize((target_h, target_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_gpu_transforms(self):
        """PyTorch 路径 GPU transform（无 Resize）。"""
        import torchvision.transforms.v2 as v2
        return v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_gpu_resize_transforms(self, target_h, target_w):
        """ONNX 路径 GPU transform（带 Resize 到固定尺寸）。"""
        import torchvision.transforms.v2 as v2
        return v2.Compose([
            v2.Resize((target_h, target_w)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ════════════════════════════════════════════════════════════
    #  ONNX session
    # ════════════════════════════════════════════════════════════

    def _build_onnx_session(self, onnx_path):
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        # 优先级：CUDA > CPU
        # 注意：TensorRT EP 对 ViT 架构不支持（norm+broadcast+transpose+reshape 融合算子），
        # 且 1080x1920 输入的 autotuning 需要 >1.5GB 显存，故默认使用 CUDA EP。
        provider_attempts = []

        if "TensorrtExecutionProvider" in available:
            provider_attempts.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            provider_attempts.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
            }))
        provider_attempts.append("CPUExecutionProvider")

        session = None
        for prov in provider_attempts:
            try:
                if isinstance(prov, tuple):
                    session = ort.InferenceSession(onnx_path, sess_options, providers=[prov])
                else:
                    session = ort.InferenceSession(onnx_path, sess_options, providers=[prov])
                # 验证实际使用的 provider（ORT 可能静默 fallback）
                actual = session.get_providers()
                name = prov if isinstance(prov, str) else prov[0]
                if name in actual or (name == "CUDAExecutionProvider" and "CUDAExecutionProvider" in actual):
                    print(f"[ECInferencer] ONNX session: {name}")
                    break
                else:
                    # 实际用的不是请求的 provider，继续尝试下一个
                    print(f"[ECInferencer] {name} 静默回退到 {actual}，尝试下一个")
                    session = None
            except Exception as e:
                name = prov if isinstance(prov, str) else prov[0]
                print(f"[ECInferencer] {name} 失败: {e}")
                session = None

        if session is None:
            raise RuntimeError("所有 ORT providers 都初始化失败！")

        self._onnx_session = session
        self._onnx_output_names = [o.name for o in session.get_outputs()]
        # 记录每个输入期望的 numpy dtype（FP16 模型需要 FP16 输入）
        self._onnx_input_dtypes = {}
        for inp in session.get_inputs():
            self._onnx_input_dtypes[inp.name] = np.dtype(
                self._ort_elem_type_to_np(inp.type))

    @staticmethod
    def _ort_elem_type_to_np(elem_type: str) -> str:
        """onnxruntime 输入类型字符串 → numpy dtype 字符串。"""
        if "float16" in elem_type:
            return "float16"
        if "float" in elem_type:
            return "float32"
        if "double" in elem_type:
            return "float64"
        if "int64" in elem_type:
            return "int64"
        if "int32" in elem_type:
            return "int32"
        return "float32"  # 兜底

    def _onnx_run(self, input_np, orig_sizes_np):
        """ONNX 推理并返回过滤后的 (labels, boxes, scores) numpy 数组。

        根据模型输入期望的 dtype 自动转换（FP16 模型转 FP16 输入）。
        """
        # 转换 images dtype 匹配模型
        img_dtype = self._onnx_input_dtypes.get("images", "float32")
        if input_np.dtype != img_dtype:
            input_np = input_np.astype(img_dtype)
        # 转换 orig_target_sizes dtype 匹配模型
        sz_dtype = self._onnx_input_dtypes.get("orig_target_sizes", "float32")
        if orig_sizes_np.dtype != sz_dtype:
            orig_sizes_np = orig_sizes_np.astype(sz_dtype)

        onnx_result = self._onnx_session.run(
            self._onnx_output_names,
            {"images": input_np, "orig_target_sizes": orig_sizes_np},
        )
        return onnx_result  # [labels, boxes, scores]

    @staticmethod
    def _filter_to_results(labels, boxes, scores, thresh, batch_idx=0):
        """将 ONNX 输出过滤为 Result 列表。"""
        keep = scores[batch_idx] > thresh
        lbls = labels[batch_idx][keep]
        bxs = boxes[batch_idx][keep]
        scs = scores[batch_idx][keep]
        results = []
        for j in range(len(lbls)):
            results.append(Result(
                label=int(lbls[j]),
                score=float(scs[j]),
                box=bxs[j].astype(np.float32),
            ))
        return results

    # ════════════════════════════════════════════════════════════
    #  统一推理接口（自动分流）
    # ════════════════════════════════════════════════════════════

    def infer(self, image, return_raw_tensor=False):
        """单张 PIL 图片推理。"""
        if self.use_onnx:
            return self._infer_onnx_pil(image, return_raw_tensor)
        return self._infer_torch_pil(image, return_raw_tensor)

    def infer_tensor(self, tensor, orig_size):
        """单张 GPU tensor 推理，tensor 格式 (3, H, W) uint8。"""
        if self.use_onnx:
            return self._infer_onnx_tensor(tensor, orig_size)
        return self._infer_torch_tensor(tensor, orig_size)

    def infer_batch(self, tensor_batch, orig_sizes):
        """批量 GPU tensor 推理，tensor_batch 格式 (N, 3, H, W) uint8。"""
        if self.use_onnx:
            return self._infer_onnx_batch(tensor_batch, orig_sizes)
        return self._infer_torch_batch(tensor_batch, orig_sizes)

    def infer_images_batch(self, images, batch_size=8):
        """批量 PIL 图片推理。"""
        if self.use_onnx:
            return self._infer_onnx_images_batch(images, batch_size)
        return self._infer_torch_images_batch(images, batch_size)

    # ════════════════════════════════════════════════════════════
    #  PyTorch 推理路径
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _pad_to_stride(tensor_batch: torch.Tensor, stride: int = 32) -> torch.Tensor:
        """将 batch tensor (N,3,H,W) 的 H/W padding 到 stride 的倍数。

        模型 backbone 下采样 stride=32，FPN 融合时对高层特征做 2x 上采样；
        若输入 H/W 不是 32 的倍数（如 1080），某些层会出现奇数尺寸，
        upsample(2x) 后与低层特征尺寸差 1 像素 → RuntimeError:
        "The size of tensor a (134) must match the size of tensor b (135)"。

        padding 到 32 倍数后各层均为偶数，问题消除。
        postprocessor 用归一化坐标 × orig_target_sizes 还原 box，
        padding 引入的误差 ≤ stride/短边 < 1%，对检测精度影响可忽略。
        """
        h, w = tensor_batch.shape[-2:]
        ph = (stride - h % stride) % stride
        pw = (stride - w % stride) % stride
        if ph or pw:
            tensor_batch = torch.nn.functional.pad(tensor_batch, (0, pw, 0, ph))
        return tensor_batch

    @staticmethod
    def _clip_boxes_to_orig(boxes_np: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
        """将 padded 坐标系下的 xyxy 框裁剪回原图范围。

        推理输入被 _pad_to_stride 在右下 padding 到 stride 倍数，原图位于
        padded 图左上区域；模型输出的归一化坐标相对 padded 图，还原时用
        padded 尺寸作为基准（见各入口），得到的坐标即原图坐标，这里把
        落在 padding 区域（如 1080~1088）的边界裁回原图。
        """
        ow, oh = orig_size
        return np.clip(boxes_np, 0, [ow, oh, ow, oh])

    @torch.no_grad()
    def _infer_torch_tensor(self, tensor: torch.Tensor, orig_size: tuple[int, int]) -> list:
        """PyTorch GPU tensor 推理，避免 PIL CPU 开销。

        tensor: (3, H, W) uint8 GPU tensor
        orig_size: (width, height)
        """
        tensor = self._gpu_transforms(tensor.unsqueeze(0))  # (1,3,H',W') float32 GPU
        if self.square_size:
            # 拉伸到正方形，与训练(2048×2048 拉伸)分布一致；拉伸保持归一化坐标不变
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self.square_size, self.square_size),
                mode="bilinear", align_corners=False)
        tensor = self._pad_to_stride(tensor)  # square_size 为 32 倍数时 no-op；否则自动补齐避免 FPN 报错
        if self.square_size:
            # 拉伸方案：归一化坐标相对拉伸图但比值不变，还原基准用原图尺寸
            orig_w, orig_h = orig_size
            orig_sizes = torch.tensor([[orig_w, orig_h]], device=self.device)
        else:
            # padding 方案：归一化坐标相对 padded 图，原图位于 padded 图左上区域
            pad_h, pad_w = tensor.shape[-2], tensor.shape[-1]
            orig_sizes = torch.tensor([[pad_w, pad_h]], device=self.device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(tensor, orig_sizes)

        if self.task == "segmentation":
            labels, boxes, scores, masks = outputs
        elif self.task == "detection":
            labels, boxes, scores = outputs
            masks = None
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        keep = scores[0] > self.thresh
        lbls = labels[0][keep]
        bxs = boxes[0][keep]
        scs = scores[0][keep]

        results = []
        if masks is not None:
            img_h, img_w = orig_size[1], orig_size[0]
            msks = masks[0].unsqueeze(0)
            if self.square_size:
                # 拉伸保持归一化坐标不变：mask 直接上采样到原图尺寸
                msks = torch.nn.functional.interpolate(
                    msks, size=(img_h, img_w), mode="bilinear", align_corners=False,
                )[0]
            else:
                # 先上采样到 padded 尺寸，再裁剪回原图（原图位于 padded 图左上角）
                msks = torch.nn.functional.interpolate(
                    msks, size=(pad_h, pad_w), mode="bilinear", align_corners=False,
                )[0][:, :img_h, :img_w]
            msks = msks > 0.0
            msks = msks[keep]
            for j in range(len(lbls)):
                results.append(Result(
                    label=int(lbls[j].item()),
                    score=float(scs[j].item()),
                    box=self._clip_boxes_to_orig(bxs[j].cpu().numpy(), orig_size),
                    mask=msks[j].cpu().numpy(),
                ))
        else:
            for j in range(len(lbls)):
                results.append(Result(
                    label=int(lbls[j].item()),
                    score=float(scs[j].item()),
                    box=self._clip_boxes_to_orig(bxs[j].cpu().numpy(), orig_size),
                ))

        return results

    @torch.no_grad()
    def _infer_torch_batch(
        self, tensor_batch: torch.Tensor, orig_sizes: list[tuple[int, int]]
    ) -> list[list]:
        """PyTorch 批量 GPU tensor 推理。

        tensor_batch: (N, 3, H, W) uint8 GPU tensor
        orig_sizes: list[(W, H), ...]
        """
        n_frames = len(orig_sizes)

        # 批量 GPU 变换：Normalize，一次完成 N 帧
        tensor_batch = self._gpu_transforms(tensor_batch)  # (N, 3, H', W') float32 GPU
        if self.square_size:
            # 拉伸到正方形，与训练(2048×2048 拉伸)分布一致；拉伸保持归一化坐标不变
            tensor_batch = torch.nn.functional.interpolate(
                tensor_batch, size=(self.square_size, self.square_size),
                mode="bilinear", align_corners=False)
        tensor_batch = self._pad_to_stride(tensor_batch)  # square_size 为 32 倍数时 no-op
        if self.square_size:
            # 拉伸方案：归一化坐标相对拉伸图但比值不变，还原基准用原图尺寸
            orig_sizes_t = torch.tensor(orig_sizes, device=self.device)  # [(W,H),...]
        else:
            # padding 方案：归一化坐标相对 padded 图，原图位于 padded 图左上区域
            pad_h, pad_w = tensor_batch.shape[-2], tensor_batch.shape[-1]
            padded_sizes = [(pad_w, pad_h)] * n_frames
            orig_sizes_t = torch.tensor(padded_sizes, device=self.device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(tensor_batch, orig_sizes_t)

        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 4:
                labels, boxes, scores, masks = outputs
            elif len(outputs) == 3:
                labels, boxes, scores = outputs
                masks = None
            else:
                raise RuntimeError(f"Unexpected PostProcessor output length: {len(outputs)}")
        else:
            raise RuntimeError(f"Unexpected PostProcessor output type: {type(outputs)}")

        all_results = []
        for i in range(n_frames):
            keep = scores[i] > self.thresh
            lbls = labels[i][keep]
            bxs = boxes[i][keep]
            scs = scores[i][keep]

            frame_results = []
            if masks is not None:
                orig_w, orig_h = orig_sizes[i]
                msks = masks[i].unsqueeze(0)
                if self.square_size:
                    # 拉伸保持归一化坐标不变：mask 直接上采样到原图尺寸
                    msks = torch.nn.functional.interpolate(
                        msks, size=(orig_h, orig_w), mode="bilinear", align_corners=False,
                    )[0]
                else:
                    # 先上采样到 padded 尺寸，再裁剪回原图（原图位于 padded 图左上角）
                    msks = torch.nn.functional.interpolate(
                        msks, size=(pad_h, pad_w), mode="bilinear", align_corners=False,
                    )[0][:, :orig_h, :orig_w]
                msks = msks > 0.0
                msks = msks[keep]
                for j in range(len(lbls)):
                    frame_results.append(Result(
                        label=int(lbls[j].item()),
                        score=float(scs[j].item()),
                        box=self._clip_boxes_to_orig(bxs[j].cpu().numpy(), orig_sizes[i]),
                        mask=msks[j].cpu().numpy(),
                    ))
            else:
                for j in range(len(lbls)):
                    frame_results.append(Result(
                        label=int(lbls[j].item()),
                        score=float(scs[j].item()),
                        box=self._clip_boxes_to_orig(bxs[j].cpu().numpy(), orig_sizes[i]),
                    ))
            all_results.append(frame_results)

        return all_results

    @torch.no_grad()
    def _infer_torch_pil(self, image: Image.Image, return_raw_tensor: bool = False):
        """PyTorch 单张 PIL 图片推理。"""
        orig_size = (image.size[0], image.size[1])
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        if self.square_size:
            # 拉伸到正方形，与训练(2048×2048 拉伸)分布一致；拉伸保持归一化坐标不变
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self.square_size, self.square_size),
                mode="bilinear", align_corners=False)
        tensor = self._pad_to_stride(tensor)  # square_size 为 32 倍数时 no-op
        if self.square_size:
            # 拉伸方案：归一化坐标相对拉伸图但比值不变，还原基准用原图尺寸
            orig_w, orig_h = orig_size
            orig_sizes = torch.tensor([[orig_w, orig_h]], device=self.device)
        else:
            # padding 方案：归一化坐标相对 padded 图，原图位于 padded 图左上区域
            pad_h, pad_w = tensor.shape[-2], tensor.shape[-1]
            orig_sizes = torch.tensor([[pad_w, pad_h]], device=self.device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(tensor, orig_sizes)

        if self.task == "segmentation":
            labels, boxes, scores, masks = outputs
        elif self.task == "detection":
            labels, boxes, scores = outputs
            masks = None
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        keep = scores[0] > self.thresh
        lbls = labels[0][keep]
        bxs = boxes[0][keep]
        scs = scores[0][keep]

        results = []
        if masks is not None:
            img_h, img_w = image.size[1], image.size[0]
            msks = masks[0].unsqueeze(0)
            if self.square_size:
                # 拉伸保持归一化坐标不变：mask 直接上采样到原图尺寸
                msks = torch.nn.functional.interpolate(
                    msks, size=(img_h, img_w), mode="bilinear", align_corners=False,
                )[0]
            else:
                # 先上采样到 padded 尺寸，再裁剪回原图（原图位于 padded 图左上角）
                msks = torch.nn.functional.interpolate(
                    msks, size=(pad_h, pad_w), mode="bilinear", align_corners=False,
                )[0][:, :img_h, :img_w]
            msks = msks > 0.0
            msks = msks[keep]
            for j in range(len(lbls)):
                results.append(Result(
                    label=int(lbls[j].item()),
                    score=float(scs[j].item()),
                    box=self._clip_boxes_to_orig(bxs[j].cpu().numpy(), orig_size),
                    mask=msks[j].cpu().numpy(),
                ))
        else:
            for j in range(len(lbls)):
                results.append(Result(
                    label=int(lbls[j].item()),
                    score=float(scs[j].item()),
                    box=self._clip_boxes_to_orig(bxs[j].cpu().numpy(), orig_size),
                ))

        if return_raw_tensor:
            raw_tensor = T.ToTensor()(image).to(self.device)
            return results, raw_tensor
        return results

    def _infer_torch_images_batch(self, images: list[Image.Image], batch_size: int = 8):
        """PyTorch 批量 PIL 图片推理。"""
        all_results = [None] * len(images)
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            n = len(batch)

            tensors = []
            orig_sizes_list = []
            for img in batch:
                tensors.append(self.transforms(img))
                orig_sizes_list.append((img.size[0], img.size[1]))  # (W, H)
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)
            if self.square_size:
                # 拉伸到正方形，与训练(2048×2048 拉伸)分布一致；拉伸保持归一化坐标不变
                batch_tensor = torch.nn.functional.interpolate(
                    batch_tensor, size=(self.square_size, self.square_size),
                    mode="bilinear", align_corners=False)
            batch_tensor = self._pad_to_stride(batch_tensor)  # square_size 为 32 倍数时 no-op
            if self.square_size:
                # 拉伸方案：归一化坐标相对拉伸图但比值不变，还原基准用原图尺寸
                orig_sizes = torch.tensor(orig_sizes_list, device=self.device)  # [(W,H),...]
            else:
                # padding 方案：归一化坐标相对 padded 图，原图位于 padded 图左上区域
                pad_h, pad_w = batch_tensor.shape[-2], batch_tensor.shape[-1]
                padded_sizes = [(pad_w, pad_h)] * n
                orig_sizes = torch.tensor(padded_sizes, device=self.device)

            outputs = self.model(batch_tensor, orig_sizes)

            if self.task == "segmentation":
                labels, boxes, scores, masks = outputs
            elif self.task == "detection":
                labels, boxes, scores = outputs
                masks = None
            else:
                raise ValueError(f"Unsupported task: {self.task}")

            for j in range(n):
                keep = scores[j] > self.thresh
                lbls = labels[j][keep]
                bxs = boxes[j][keep]
                scs = scores[j][keep]
                img_w, img_h = batch[j].size

                results = []
                if masks is not None:
                    m = masks[j].unsqueeze(0)
                    if self.square_size:
                        # 拉伸保持归一化坐标不变：mask 直接上采样到原图尺寸
                        m = torch.nn.functional.interpolate(
                            m, size=(img_h, img_w), mode="bilinear", align_corners=False,
                        )[0]
                    else:
                        # 先上采样到 padded 尺寸，再裁剪回原图（原图位于 padded 图左上角）
                        m = torch.nn.functional.interpolate(
                            m, size=(pad_h, pad_w), mode="bilinear", align_corners=False,
                        )[0][:, :img_h, :img_w]
                    m = m > 0.0
                    m = m[keep]
                    for k in range(len(lbls)):
                        results.append(Result(
                            label=int(lbls[k].item()),
                            score=float(scs[k].item()),
                            box=self._clip_boxes_to_orig(bxs[k].cpu().numpy(), (img_w, img_h)),
                            mask=m[k].cpu().numpy(),
                        ))
                else:
                    for k in range(len(lbls)):
                        results.append(Result(
                            label=int(lbls[k].item()),
                            score=float(scs[k].item()),
                            box=self._clip_boxes_to_orig(bxs[k].cpu().numpy(), (img_w, img_h)),
                        ))
                all_results[start + j] = results

        return all_results

    # ════════════════════════════════════════════════════════════
    #  ONNX 推理路径（固定 1080x1920 输入）
    # ════════════════════════════════════════════════════════════

    def _infer_onnx_pil(self, image: Image.Image, return_raw_tensor: bool = False):
        """ONNX 单张 PIL 图片推理。

        Resize 到 (1080, 1920) → ONNX 推理 → box 自动还原到原始尺寸。
        """
        orig_w, orig_h = image.size
        # CPU transform: Resize -> ToTensor -> Normalize
        tensor = self.transforms(image)                  # (3, 1080, 1920) float32
        input_np = tensor.unsqueeze(0).numpy()           # (1, 3, 1080, 1920)
        orig_sizes_np = np.array([[orig_w, orig_h]], dtype=np.float32)

        labels, boxes, scores = self._onnx_run(input_np, orig_sizes_np)
        results = self._filter_to_results(labels, boxes, scores, self.thresh, batch_idx=0)

        if return_raw_tensor:
            raw_tensor = T.ToTensor()(image).to(self.device)
            return results, raw_tensor
        return results

    def _infer_onnx_tensor(self, tensor: torch.Tensor, orig_size: tuple[int, int]) -> list:
        """ONNX 单张 GPU tensor 推理。

        tensor: (3, H, W) uint8 GPU tensor
        orig_size: (width, height)
        """
        # GPU transform: Resize -> ToDtype -> Normalize
        tensor = self._gpu_transforms(tensor.unsqueeze(0))  # (1, 3, 1080, 1920) float32 GPU
        # ONNX 需要 numpy 输入
        input_np = tensor.cpu().numpy()
        orig_sizes_np = np.array([[orig_size[0], orig_size[1]]], dtype=np.float32)

        labels, boxes, scores = self._onnx_run(input_np, orig_sizes_np)
        return self._filter_to_results(labels, boxes, scores, self.thresh, batch_idx=0)

    def _infer_onnx_batch(
        self, tensor_batch: torch.Tensor, orig_sizes: list[tuple[int, int]]
    ) -> list[list]:
        """ONNX 批量 GPU tensor 推理。

        tensor_batch: (N, 3, H, W) uint8 GPU tensor
        orig_sizes: list[(W, H), ...]
        """
        n_frames = len(orig_sizes)

        # 批量 GPU 变换：Resize + Normalize
        tensor_batch = self._gpu_transforms(tensor_batch)  # (N, 3, 1080, 1920) float32 GPU
        # ONNX 需要 numpy 输入（GPU→CPU 拷贝）
        input_np = tensor_batch.cpu().numpy()
        orig_sizes_np = np.array(orig_sizes, dtype=np.float32)

        labels, boxes, scores = self._onnx_run(input_np, orig_sizes_np)

        all_results = []
        for i in range(n_frames):
            results = self._filter_to_results(labels, boxes, scores, self.thresh, batch_idx=i)
            all_results.append(results)

        return all_results

    def _infer_onnx_images_batch(self, images: list[Image.Image], batch_size: int = 8):
        """ONNX 批量 PIL 图片推理。"""
        all_results = [None] * len(images)

        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch_images = images[start:end]
            actual_bs = len(batch_images)

            # 批量 CPU transform
            batch_tensors = []
            batch_orig_sizes = []
            for img in batch_images:
                t = self.transforms(img)  # (3, 1080, 1920)
                batch_tensors.append(t)
                batch_orig_sizes.append([img.size[0], img.size[1]])

            input_np = torch.stack(batch_tensors).numpy()  # (N, 3, 1080, 1920)
            orig_sizes_np = np.array(batch_orig_sizes, dtype=np.float32)

            labels, boxes, scores = self._onnx_run(input_np, orig_sizes_np)

            for i in range(actual_bs):
                results = self._filter_to_results(
                    labels, boxes, scores, self.thresh, batch_idx=i)
                all_results[start + i] = results

        return all_results


class VideoReader(threading.Thread):
    def __init__(self, cap, queue_size=32):
        super().__init__()
        self.cap = cap
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.daemon = True

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        while not self.q.empty():
            self.q.get()


def process_image(inferencer: ECInferencer, path: Path, device="cuda:0"):
    image = Image.open(path).convert("RGB")
    results = inferencer.infer(image)

    # ── 保存结果 ──────────────────────────────────────────────────
    from streaming_cluster_updater import StreamingClusterUpdater

    save_info = StreamingClusterUpdater.save_frame_results(
        image=image,
        results=results,
        keep=True,            # 离线推理全部保留，RAG 决策由 api_service 中的 updater 负责
        output_root=path.parent.parent,
        frame_stem=path.stem,
        draw_func=draw,
    )

    print(f"Saved original to {save_info['init_path']}")
    print(f"Saved result to {save_info['output_path']}")
    print(f"Saved labels to {save_info['labels_path']}")
    print(f"Detected {len(results)} instances -> saved to {save_info['save_location']}")


def process_image_dir(inferencer: ECInferencer, dir_path: Path, device="cuda:0",
                      updater=None):
    """处理目录中所有图片，可选地在处理完后更新 RAG 聚类中心。

    Parameters
    ----------
    updater : StreamingClusterUpdater or None
        如果提供，每段视频处理后调用 updater.process_video() 更新知识库。
        写操作带文件锁，多进程共享安全。
    """
    image_paths = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise ValueError(f"No image files found in directory: {dir_path}")

    print(f"Found {len(image_paths)} images in {dir_path}")

    frames_data = [] if updater is not None else None

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}")

        image = Image.open(img_path).convert("RGB")
        results = inferencer.infer(image)

        # 如果 updater 存在，用 decide_frame 做决策；否则全部保留
        if updater is not None:
            decision = updater.decide_frame(results, image)
            keep = decision["keep"]
            print(f"  [RAG] {decision['reason']}")
            frames_data.append({
                "image": image,
                "results": results,
                "frame_name": img_path.name,
            })
        else:
            keep = True

        from streaming_cluster_updater import StreamingClusterUpdater
        save_info = StreamingClusterUpdater.save_frame_results(
            image=image,
            results=results,
            keep=keep,
            output_root=img_path.parent.parent,
            frame_stem=img_path.stem,
            draw_func=draw,
        )
        print(f"  Saved to {save_info['save_location']}")

    # ── 整个视频处理完后，更新 RAG 聚类中心（在 torch_inf 中调用） ──
    if updater is not None and frames_data:
        try:
            result = updater.process_video(frames_data)
            print(f"[StreamingUpdater] 聚类更新: "
                  f"{result['total_novel_frames']}/{len(frames_data)} frames novel, "
                  f"+{result['total_new_clusters']} new clusters")
        except Exception as e:
            print(f"[StreamingUpdater] 更新失败(不阻塞): {e}")


def process_video(inferencer: ECInferencer, path: Path, num_workers: int = 4):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = path.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{path.stem}_torch_pose_inference{path.suffix}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video writer: {output_path}")

    reader = VideoReader(cap)
    reader.start()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    futures_queue = queue.Queue(maxsize=max(32, num_workers * 4))

    def process_and_draw(pil_img: Image.Image, results: list[Result]):
        res_np = draw_to_numpy(pil_img, results)
        return cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)

    frame_count = 0
    writer_error: dict[str, Exception | None] = {"exc": None}

    def writer_worker():
        nonlocal frame_count
        try:
            while True:
                future = futures_queue.get()
                if future is None:
                    break
                frame_out = future.result()
                out.write(frame_out)
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{total_frame} frames")
        except Exception as exc:  # noqa: BLE001
            writer_error["exc"] = exc

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    try:
        while True:
            if writer_error["exc"] is not None:
                raise writer_error["exc"]

            frame = reader.read()
            if frame is None:
                break

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = inferencer.infer(pil_img)
            future = executor.submit(process_and_draw, pil_img, results)
            futures_queue.put(future)
    finally:
        reader.stop()
        futures_queue.put(None)
        writer_thread.join()
        executor.shutdown()
        cap.release()
        out.release()

    if writer_error["exc"] is not None:
        raise writer_error["exc"]

    print(f"Saved video result to {output_path}")


# ════════════════════════════════════════════════════════════
#  模型构建
# ════════════════════════════════════════════════════════════

def build_model(config_path: str, resume_path: str, device: torch.device):
    """构建模型。

    返回值保持 3 元组 (model, img_size, task)，调用方式不变。
    始终加载 pth 模型（PyTorch FP16 推理最快，ONNX 路径仅通过
    ECInferencer(onnx_path=...) 显式启用）。
    """
    cfg = YAMLConfig(config_path, resume=resume_path)
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    task = cfg.yaml_cfg["task"]

    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()

    return model, cfg.yaml_cfg["eval_spatial_size"], cfg.yaml_cfg["task"]


def main(args):
    device = torch.device(args.device)
    model, img_size, task = build_model(args.config, args.resume, device)

    inferencer = ECInferencer(
        model=model,
        task=task,
        device=device,
        size=img_size,
        thresh=args.thresh,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # ── 流式聚类更新器（离线推理时同步更新 RAG 知识库） ──────────
    updater = None
    rag_cache = getattr(args, "rag_cache", None)
    if rag_cache and os.path.exists(rag_cache):
        from streaming_cluster_updater import StreamingClusterUpdater
        updater = StreamingClusterUpdater(
            rag_db_path=rag_cache,
            device=args.device,
        )
        updater.load_state()
        print(f"StreamingUpdater 已加载: {updater.rag_db.total_clusters} clusters")

    if input_path.is_dir():
        process_image_dir(inferencer, input_path, device=args.device, updater=updater)
    elif input_path.suffix.lower() in IMAGE_EXTS:
        process_image(inferencer, input_path, device=args.device)
    else:
        process_video(inferencer, input_path, num_workers=args.num_workers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EdgeCrafter Inference")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-i", "--input", required=True, help="Image path, image directory path, or video path")
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        help="Device to run inference on (e.g., 'cuda:0' or 'cpu')",
    )
    parser.add_argument("-t", "--thresh", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=4, help="Thread workers for video draw/write")
    parser.add_argument(
        "--rag-cache",
        default="/data2/caimoyao/EdgeCrafter-main/ecdetseg/rag_db.pkl",
        help="Path to shared RAG DB (.pkl). When set, each video batch triggers "
             "a clustering update with file-lock protection for multi-process safety.",
    )

    main(parser.parse_args())
