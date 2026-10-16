"""调试 torch.compile 的 graph break：查看模型中哪些部分无法被捕获。

用法：
    cd /data2/caimoyao/EdgeCrafter-main/ecdetseg/tools/inference
    python debug_compile_graph_break.py

依赖：torch (api_service 环境)
"""

import os
import sys
import time

# 只打印 graph_breaks，关闭其他日志（避免日志量太大淹没关键信息）
os.environ.setdefault("TORCH_LOGS", "+graph_breaks")

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from engine.core import YAMLConfig


# ── 1. 加载模型 ──
CONFIG_PATH = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/configs/ecdet/ecdet_l.yml"
CKPT_PATH = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/models/last.pth"
INPUT_H, INPUT_W = 1080, 1920
DEVICE = "cuda:0"

cfg = YAMLConfig(CONFIG_PATH, resume=CKPT_PATH)
cfg.yaml_cfg["eval_spatial_size"] = [INPUT_H, INPUT_W]
cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
IGNORED = {"anchors", "valid_mask"}
filtered_state = {k: v for k, v in state.items() if not any(ig in k for ig in IGNORED)}
cfg.model.load_state_dict(filtered_state, strict=False)


class InferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


model = InferenceModel().to(DEVICE).eval()
dummy_images = torch.randn(1, 3, INPUT_H, INPUT_W, device=DEVICE, dtype=torch.float32)
dummy_sizes = torch.tensor([[INPUT_W, INPUT_H]], device=DEVICE, dtype=torch.float32)

# ── 2. 用 torch._dynamo.explain 分析 graph break ──
# 直接用 torch.compile + fullgraph=True 触发编译失败，让日志打印 break 信息
# 但这会抛异常，所以我们用 fullgraph=False 正常编译，并通过日志看 fallback 情况

print("=" * 60)
print("  编译并打印 graph break 位置（TORCH_LOGS=+graph_breaks 已开启）")
print("=" * 60)

# 先 warmup 一次（会触发编译，打印 break 信息）
print("\n  ── 首次编译（会打印 graph break 信息）──\n")
try:
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
    with torch.no_grad():
        _ = compiled(dummy_images, dummy_sizes)
        torch.cuda.synchronize()
    print("\n  ✅ 编译成功")
except Exception as e:
    print(f"\n  ❌ 编译失败: {e}")

# ── 3. 测速对比 ──
print("\n" + "=" * 60)
print("  测速对比")
print("=" * 60)

iters = 20
# compiled
with torch.no_grad():
    for _ in range(5):
        _ = compiled(dummy_images, dummy_sizes)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = compiled(dummy_images, dummy_sizes)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  compiled: {elapsed/iters*1000:.2f} ms/frame")

# eager
with torch.no_grad():
    for _ in range(5):
        _ = model(dummy_images, dummy_sizes)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(dummy_images, dummy_sizes)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"  eager:    {elapsed/iters*1000:.2f} ms/frame")

