"""
将 EdgeCrafter-ecdet_l 检测模型导出为 FP16 ONNX，并验证 PyTorch / ONNX-CUDA / TensorRT 的精度与速度。

特点：
- 固定输入尺寸 (1080, 1920)（视频原生分辨率），动态 batch
- 端到端导出：backbone + decoder + PostProcessor(deploy) 合在一起
- 依次对比 PyTorch FP16、ONNX FP16 (CUDA EP)、TensorRT FP16 的精度与速度

用法：
    cd /data2/caimoyao/EdgeCrafter-main/ecdetseg/tools/inference
    conda activate api_service
    python export_ecdet_onnx.py

依赖（如缺失先安装）：
    pip install onnx onnxsim onnxconverter_common onnxruntime-gpu[cu12]
    # TensorRT 可选：pip install tensorrt
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parents[2]  # .../ecdetseg
CONFIG_PATH = BASE_DIR / "configs" / "ecdet" / "ecdet_l.yml"
# 使用您训练的 3 类模型权重
CKPT_PATH = BASE_DIR / "models" / "last.pth"

# 固定输入尺寸（视频原生分辨率，H, W）
INPUT_H, INPUT_W = 1080, 1920

# ONNX 导出参数
OPSET_VERSION = 17
ONNX_OUTPUT_BASE = BASE_DIR / "models" / f"ecdet_l_{INPUT_H}x{INPUT_W}"

ATOL = 1e-4

# 需要忽略的 buffer（基于训练时 eval_spatial_size 生成，推理时会重新计算）
IGNORED_STATE_KEYS = {"anchors", "valid_mask"}


# ==================== 工具函数 ====================
def get_onnx_stats(onnx_model):
    node_count = len(onnx_model.graph.node)
    initializer_count = len(onnx_model.graph.initializer)
    input_count = len(onnx_model.graph.input)
    output_count = len(onnx_model.graph.output)
    ops_count = {}
    for node in onnx_model.graph.node:
        ops_count[node.op_type] = ops_count.get(node.op_type, 0) + 1
    return {
        "nodes": node_count,
        "initializers": initializer_count,
        "inputs": input_count,
        "outputs": output_count,
        "op_types": ops_count,
    }


def print_stats(name, stats, file_size_mb=None):
    print(f"\n{'='*60}")
    print(f"   {name}")
    print(f"{'='*60}")
    if file_size_mb is not None:
        print(f"   文件大小: {file_size_mb:.2f} MB")
    print(f"   节点数: {stats['nodes']}")
    print(f"   初始化器: {stats['initializers']}")
    print(f"   输入/输出: {stats['inputs']} / {stats['outputs']}")
    print("   算子分布 (前10):")
    for op, cnt in sorted(stats["op_types"].items(), key=lambda x: -x[1])[:10]:
        print(f"     {op:20s}: {cnt}")


# ==================== 模型构建 ====================
def build_model(device):
    """与 torch_inf.py 中的 build_model 保持一致。"""
    sys.path.insert(0, str(BASE_DIR))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(CONFIG_PATH), resume=str(CKPT_PATH))
    cfg.yaml_cfg["eval_spatial_size"] = [INPUT_H, INPUT_W]
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    checkpoint = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]

    # 过滤掉基于训练分辨率生成的 anchors/valid_mask buffer
    # 这些 buffer 在推理时会根据实际 eval_spatial_size 动态重新生成
    filtered_state = {
        k: v for k, v in state.items()
        if not any(ignored in k for ignored in IGNORED_STATE_KEYS)
    }

    cfg.model.load_state_dict(filtered_state, strict=False)

    # 检查加载情况
    missing, unexpected = cfg.model.state_dict().keys(), filtered_state.keys()
    loaded_keys = set(filtered_state.keys())
    model_keys = set(cfg.model.state_dict().keys())
    skipped = model_keys - loaded_keys
    if skipped:
        print(f"   ⚠️ 跳过加载 {len(skipped)} 个 key（均为 buffer，推理时自动生成）")

    class InferenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = InferenceModel().to(device)
    model.eval()
    return model


# ==================== 导出 ====================
def export_onnx(model, device):
    import onnx

    print(f"\n{'='*60}")
    print(f"   阶段 1: ONNX 导出")
    print(f"{'='*60}")

    # 先用 FP32 导出（避免模型内部混合精度问题）
    # onnxruntime 推理时可以用 graph optimization 自动降精度
    model_fp32 = model.float().to(device)
    model_fp32.eval()

    dummy_images = torch.randn(1, 3, INPUT_H, INPUT_W, dtype=torch.float32, device=device)
    dummy_sizes = torch.tensor([[INPUT_W, INPUT_H]], dtype=torch.float32, device=device)

    print("   测试 FP32 推理...")
    with torch.no_grad():
        _ = model_fp32(dummy_images, dummy_sizes)
    print("   ✅ FP32 推理正常")

    onnx_raw = f"{ONNX_OUTPUT_BASE}_fp32_raw.onnx"
    print(f"\n导出 ONNX (opset={OPSET_VERSION}, FP32) → {onnx_raw}")

    dynamic_axes = {
        "images": {0: "batch"},
        "orig_target_sizes": {0: "batch"},
        "labels": {0: "batch"},
        "boxes": {0: "batch"},
        "scores": {0: "batch"},
    }

    torch.onnx.export(
        model_fp32,
        (dummy_images, dummy_sizes),
        onnx_raw,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    raw_model = onnx.load(onnx_raw)
    onnx.checker.check_model(raw_model)
    raw_size = os.path.getsize(onnx_raw) / 1024 / 1024
    raw_stats = get_onnx_stats(raw_model)
    print_stats("原始 ONNX FP32", raw_stats, raw_size)

    for name, tensor in [
        (i.name, i.type.tensor_type) for i in raw_model.graph.input
    ] + [(o.name, o.type.tensor_type) for o in raw_model.graph.output]:
        dims = []
        for d in tensor.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        print(f"   {name}: {dims}")

    return raw_model, onnx_raw, raw_stats, raw_size


# ==================== onnxsim 简化 ====================
def simplify_onnx(onnx_model, onnx_raw, raw_stats, raw_size):
    try:
        import onnxsim, onnx
    except ImportError:
        print("\n跳过 onnxsim（未安装，pip install onnxsim）")
        return onnx_model, onnx_raw, raw_stats, raw_size

    onnx_sim_path = str(ONNX_OUTPUT_BASE) + "_fp16_sim.onnx"
    print(f"\nonnxsim 简化 → {onnx_sim_path}")
    try:
        simplified, check = onnxsim.simplify(onnx_model)
        onnx.save(simplified, onnx_sim_path)
        sim_size = os.path.getsize(onnx_sim_path) / 1024 / 1024
        sim_stats = get_onnx_stats(simplified)
        print_stats("onnxsim 简化后", sim_stats, sim_size)
        reduction = (1 - sim_stats["nodes"] / raw_stats["nodes"]) * 100
        print(f"   节点数减少: {reduction:.1f}%")
        return simplified, onnx_sim_path, sim_stats, sim_size
    except Exception as e:
        print(f"   ⚠️ onnxsim 失败: {e}")
        return onnx_model, onnx_raw, raw_stats, raw_size


# ==================== 主流程（验证 + 测速） ====================
def main():
    # cuDNN 初始化容错
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False

    print("=" * 60)
    print("   EdgeCrafter-ecdet_l ONNX 导出 + 验证")
    print("=" * 60)
    print(f"   配置: {CONFIG_PATH}")
    print(f"   权重: {CKPT_PATH}")
    print(f"   输入尺寸: (H={INPUT_H}, W={INPUT_W})")
    print(f"   精度: FP32")
    print(f"   cuDNN benchmark: {cudnn.benchmark}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"   设备: {device}")

    # cuDNN 预测试：如果失败，标记需要 fallback
    cudnn_ok = True
    if torch.cuda.is_available():
        try:
            test_tensor = torch.randn(1, 3, 64, 64, device=device)
            _ = torch.nn.functional.conv2d(test_tensor, torch.randn(16, 3, 3, 3, device=device), padding=1)
            torch.cuda.synchronize()
            print("   ✅ cuDNN 初始化成功")
        except Exception as e:
            print(f"   ⚠️ cuDNN 异常: {e}")
            print("   → 尝试禁用 cuDNN，使用 CUDA 原生算子")
            cudnn.enabled = False
            try:
                _ = torch.nn.functional.conv2d(test_tensor, torch.randn(16, 3, 3, 3, device=device), padding=1)
                torch.cuda.synchronize()
                print("   ✅ 禁用 cuDNN 后正常")
                cudnn_ok = True
            except Exception as e2:
                print(f"   ⚠️ 禁用 cuDNN 后仍异常: {e2}")
                print("   → 将使用 CPU 进行模型加载和 ONNX 导出")
                cudnn_ok = False

    # ── 1. 构建模型（cuDNN 不可用时用 CPU） ──
    export_device = device if cudnn_ok else torch.device("cpu")
    print(f"\n使用 {export_device} 进行模型加载和 ONNX 导出")
    
    print("\n构建模型...")
    model = build_model(export_device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {total_params:,}")

    # ── 2. 导出 ONNX ──
    raw_model, onnx_path, raw_stats, raw_size = export_onnx(model, export_device)

    # 如果是在 CPU 上导出，迁移到 GPU 做验证（如果可能）
    if export_device.type == "cpu" and torch.cuda.is_available() and cudnn_ok:
        print("\n迁移模型到 GPU 进行验证...")
        model = model.to(device)
        # 保持 FP32，避免混合精度问题

    # ── 3. onnxsim 简化 ──
    sim_model, sim_path, sim_stats, sim_size = simplify_onnx(
        raw_model, onnx_path, raw_stats, raw_size
    )
    final_path = sim_path if sim_path != onnx_path else onnx_path
    final_size = os.path.getsize(final_path) / 1024 / 1024

    # ── 4. PyTorch vs ONNX 精度 + 速度对比 ──
    print(f"\n{'='*60}")
    print(f"   阶段 2: PyTorch vs ONNX 对比")
    print(f"{'='*60}")

    import onnxruntime as ort

    torch.manual_seed(42)
    np.random.seed(42)

    # 固定随机输入（使用实际可用的设备，FP32）
    val_device = next(model.parameters()).device
    test_input_np = np.random.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np).to(device=val_device, dtype=torch.float32)
    test_sizes_np = np.array([[INPUT_W, INPUT_H]], dtype=np.float32)
    test_sizes_torch = torch.from_numpy(test_sizes_np).to(device=val_device, dtype=torch.float32)

    # ── PyTorch 推理 ──
    device_type = "GPU" if val_device.type == "cuda" else "CPU"
    print(f"\n--- PyTorch FP32 ({device_type}) ---")
    model.eval()
    with torch.no_grad():
        pt_labels, pt_boxes, pt_scores = model(test_input_torch, test_sizes_torch)
    pt_labels_cpu = pt_labels.cpu().numpy()
    pt_boxes_cpu = pt_boxes.cpu().numpy()
    pt_scores_cpu = pt_scores.cpu().numpy()
    print(f"   输出: labels={pt_labels_cpu.shape}, boxes={pt_boxes_cpu.shape}, scores={pt_scores_cpu.shape}")
    print(f"   Top-5 scores: {pt_scores_cpu[0, :5]}")
    print(f"   Top-5 labels: {pt_labels_cpu[0, :5]}")
    print(f"   Top-5 boxes:  {pt_boxes_cpu[0, :5]}")

    # PyTorch 速度测试
    warmup = 5
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(test_input_torch, test_sizes_torch)
    if val_device.type == "cuda":
        torch.cuda.synchronize()

    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(test_input_torch, test_sizes_torch)
    if val_device.type == "cuda":
        torch.cuda.synchronize()
    pt_elapsed = time.perf_counter() - start
    print(f"   平均耗时: {pt_elapsed/iters*1000:.2f} ms")
    print(f"   吞吐量:   {iters/pt_elapsed:.2f} FPS")

    # ── ONNX Runtime 推理（带 provider fallback） ──
    # 关键：先释放 PyTorch 占用的 GPU 显存，否则 ORT CUDA EP 会 OOM
    # （PyTorch caching allocator 会持有显存不释放，导致 ORT 分配失败）
    if val_device.type == "cuda":
        print("\n   释放 PyTorch GPU 显存...")
        model.cpu()                       # 把模型移到 CPU
        test_input_torch = test_input_torch.cpu()
        test_sizes_torch = test_sizes_torch.cpu()
        torch.cuda.empty_cache()          # 清空 PyTorch CUDA 缓存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        val_device = next(model.parameters()).device  # 更新为 cpu
        print("   ✅ 已释放")

    available_providers = ort.get_available_providers()
    print(f"\n   可用 ORT providers: {available_providers}")

    # 逐个尝试 provider，按优先级：TensorRT > CUDA > CPU
    session = None
    active_provider = None

    provider_attempts = []
    if "TensorrtExecutionProvider" in available_providers:
        provider_attempts.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available_providers:
        provider_attempts.append("CUDAExecutionProvider")
    provider_attempts.append("CPUExecutionProvider")

    for prov in provider_attempts:
        try:
            print(f"\n--- 尝试 {prov.replace('ExecutionProvider', '')} ---")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # CUDA EP 专用配置：避免显存预分配过大
            if prov == "CUDAExecutionProvider":
                provider_options = [{
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",  # 按需分配，不预分配大块
                }]
                session = ort.InferenceSession(
                    final_path, sess_options,
                    providers=[("CUDAExecutionProvider", provider_options[0])]
                )
            else:
                session = ort.InferenceSession(final_path, sess_options, providers=[prov])
            active_provider = prov
            print(f"   ✅ {prov} 初始化成功")
            break
        except Exception as e:
            print(f"   ⚠️ {prov} 失败: {e}")
            session = None

    if session is None:
        raise RuntimeError("所有 ORT providers 都初始化失败！")

    provider_name = active_provider.replace("ExecutionProvider", "")
    print(f"\n--- ONNX FP32 ({provider_name}) ---")

    onnx_result = session.run(["labels", "boxes", "scores"], {
        "images": test_input_np,
        "orig_target_sizes": test_sizes_np,
    })
    onnx_labels, onnx_boxes, onnx_scores = onnx_result
    print(f"   输出: labels={onnx_labels.shape}, boxes={onnx_boxes.shape}, scores={onnx_scores.shape}")
    print(f"   Top-1 scores: {onnx_scores[0, :1]}")
    print(f"   Top-1 labels: {onnx_labels[0, :1]}")
    print(f"   Top-1 boxes:  {onnx_boxes[0, :1]}")

    # ONNX 速度测试
    for _ in range(warmup):
        session.run(["labels", "boxes", "scores"], {
            "images": test_input_np,
            "orig_target_sizes": test_sizes_np,
        })

    start = time.perf_counter()
    for _ in range(iters):
        session.run(["labels", "boxes", "scores"], {
            "images": test_input_np,
            "orig_target_sizes": test_sizes_np,
        })
    onnx_elapsed = time.perf_counter() - start
    print(f"   平均耗时: {onnx_elapsed/iters*1000:.2f} ms")
    print(f"   吞吐量:   {iters/onnx_elapsed:.2f} FPS")

    # ── 对比总结 ──
    # PostProcessor 输出 300 个 query，按 score 的 TopK 排序。
    # 当 PyTorch/ONNX 的 score 有微小数值差异时，TopK 排序顺序会不同，
    # 导致逐位置对比失效（尤其低 score 的 query，其 box 坐标是无意义的噪声）。
    # 正确做法：按 score 降序排序后，只对比 top-K 个有效检测结果。
    def sort_by_score(labels, boxes, scores):
        """按 score 降序排序，返回一维数组。"""
        order = np.argsort(-scores[0])  # 降序
        return labels[0, order], boxes[0, order], scores[0, order]

    pt_l, pt_b, pt_s = sort_by_score(pt_labels_cpu, pt_boxes_cpu, pt_scores_cpu)
    on_l, on_b, on_s = sort_by_score(onnx_labels, onnx_boxes, onnx_scores)

    # 只对比 top-1（实际使用只取第一个结果）
    pt_l_k = pt_l[:1]; pt_b_k = pt_b[:1]; pt_s_k = pt_s[:1]
    on_l_k = on_l[:1]; on_b_k = on_b[:1]; on_s_k = on_s[:1]

    label_match = bool(pt_l_k[0] == on_l_k[0])
    box_max_diff = float(np.abs(pt_b_k[0] - on_b_k[0]).max())
    score_max_diff = float(np.abs(pt_s_k[0] - on_s_k[0]).max())
    speedup = pt_elapsed / onnx_elapsed

    print(f"\n{'='*60}")
    print(f"   对比总结 (batch=1, top-1 按 score 降序)")
    print(f"{'='*60}")
    print(f"   标签一致:     {'✅' if label_match else '❌'} (PyTorch={int(pt_l_k[0])} vs ONNX={int(on_l_k[0])})")
    print(f"   boxes 最大差: {box_max_diff:.6e} {'✅' if box_max_diff < ATOL else '❌'} (阈值 {ATOL})")
    print(f"   scores 最大差: {score_max_diff:.6e} {'✅' if score_max_diff < ATOL else '❌'} (阈值 {ATOL})")
    print(f"   速度提升:     {speedup:.2f}x")
    print(f"   PyTorch:      {pt_elapsed/iters*1000:.2f} ms / {iters/pt_elapsed:.2f} FPS")
    print(f"   ONNX:         {onnx_elapsed/iters*1000:.2f} ms / {iters/onnx_elapsed:.2f} FPS")

    print(f"\n   --- Top-1 详细对比 ---")
    print(f"   PyTorch:  label={int(pt_l_k[0])}, score={pt_s_k[0]:.6f}, box={np.round(pt_b_k[0], 2)}")
    print(f"   ONNX:     label={int(on_l_k[0])}, score={on_s_k[0]:.6f}, box={np.round(on_b_k[0], 2)}")
    print(f"   box 差异:   {np.round(np.abs(pt_b_k[0] - on_b_k[0]), 2)}")
    print(f"   score 差异: {abs(pt_s_k[0] - on_s_k[0]):.6e}")

    # # ── 5. 多 batch 测试（验证 dynamic batch） ──
    # print(f"\n{'='*60}")
    # print(f"   多 Batch 测试 (batch=8)")
    # print(f"{'='*60}")
    # batch_size = 8
    # multi_input = np.random.randn(batch_size, 3, INPUT_H, INPUT_W).astype(np.float32)
    # multi_sizes = np.array([[INPUT_W, INPUT_H]] * batch_size, dtype=np.float32)

    # # warmup
    # for _ in range(3):
    #     session.run(["labels", "boxes", "scores"], {
    #         "images": multi_input[:1],
    #         "orig_target_sizes": multi_sizes[:1],
    #     })

    # start = time.perf_counter()
    # onnx_multi_out = session.run(["labels", "boxes", "scores"], {
    #     "images": multi_input,
    #     "orig_target_sizes": multi_sizes,
    # })
    # onnx_multi_elapsed = time.perf_counter() - start

    # print(f"   输入形状: images=({batch_size}, 3, {INPUT_H}, {INPUT_W}), sizes=({batch_size}, 2)")
    # print(f"   输出形状: labels={onnx_multi_out[0].shape}, boxes={onnx_multi_out[1].shape}, scores={onnx_multi_out[2].shape}")
    # print(f"   单次耗时: {onnx_multi_elapsed*1000:.2f} ms")
    # print(f"   每张耗时: {onnx_multi_elapsed/batch_size*1000:.2f} ms")
    # print(f"   吞吐量:   {batch_size/onnx_multi_elapsed:.2f} FPS")

    # ── 6. TensorRT 速度测试（可选，独立于 ORT） ──
    trt_result = None
    try:
        import tensorrt as trt
        print(f"\n{'='*60}")
        print(f"   阶段 3: TensorRT FP16 速度测试")
        print(f"{'='*60}")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(final_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"   parse error {i}: {parser.get_error(i)}")
                raise RuntimeError("ONNX parse failed")
        print("   ✅ ONNX parsing 成功")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        # 从 ONNX 获取输入信息
        onnx_m = onnx.load(final_path)
        images_input = next((i for i in onnx_m.graph.input if "images" in i.name), onnx_m.graph.input[0])
        sizes_input = next((i for i in onnx_m.graph.input if "orig_target" in i.name), onnx_m.graph.input[1])

        def get_onnx_dims(tensor):
            dims = []
            for d in tensor.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value else 1)
            return dims

        img_dims = get_onnx_dims(images_input)
        size_dims = get_onnx_dims(sizes_input)

        # 动态 batch profile
        profile = builder.create_optimization_profile()
        for inp, dims in [(images_input, img_dims), (sizes_input, size_dims)]:
            min_s = [1] + dims[1:]
            opt_s = [4] + dims[1:]
            max_s = [8] + dims[1:]
            profile.set_shape(inp.name, min_s, opt_s, max_s)
        config.add_optimization_profile(profile)

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   启用 FP16")

        print("   构建 TensorRT engine (首次可能需要几分钟) ...")
        serialized = builder.build_serialized_network(network, config)
        engine_path = str(ONNX_OUTPUT_BASE) + "_fp16.engine"
        with open(engine_path, "wb") as f:
            f.write(serialized)
        print(f"   engine 已保存: {engine_path} ({len(serialized)/1024/1024:.1f} MB)")

        # 执行
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()

        # FP16 推理：TensorRT engine 用 FP16 构建，但 I/O buffer 用 FP32 确保数据正确传输
        # （ONNX 模型是 FP32，TensorRT 内部会自动做 FP32→FP16→FP32 的精度转换）
        input_buffers = []
        output_buffers = []
        for inp, dims in [(images_input, img_dims), (sizes_input, size_dims)]:
            shape = [1] + dims[1:]
            size = int(np.prod(shape))
            buf = torch.empty(size, dtype=torch.float32, device="cuda")
            input_buffers.append(buf)

        out_names = [o.name for o in onnx_m.graph.output]
        for name in out_names:
            out_t = onnx_m.graph.output[out_names.index(name)]
            shape = [1]
            for d in out_t.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value else 1)
            size = int(np.prod(shape))
            buf = torch.empty(size, dtype=torch.float32, device="cuda")
            output_buffers.append(buf)

        stream = torch.cuda.Stream()
        img_t = torch.from_numpy(test_input_np).cuda().contiguous()   # FP32
        sizes_t = torch.from_numpy(test_sizes_np).cuda().contiguous()  # FP32

        # warmup（FP16 engine 推理，FP32 输入）
        for _ in range(warmup):
            img_t.copy_(input_buffers[0])
            sizes_t.copy_(input_buffers[1])
            context.execute_v2(input_buffers + output_buffers, stream=stream.cuda_stream)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            img_t.copy_(input_buffers[0])
            sizes_t.copy_(input_buffers[1])
            context.execute_v2(input_buffers + output_buffers, stream=stream.cuda_stream)
        torch.cuda.synchronize()
        trt_elapsed = time.perf_counter() - start
        trt_per = trt_elapsed / iters * 1000

        # 验证 FP16 输出正确性（只做一次精度对比）
        if iters >= 10:
            trt_labels = output_buffers[0].cpu().numpy().reshape(1, -1)
            trt_boxes = output_buffers[1].cpu().numpy().reshape(1, -1, 4)
            trt_scores = output_buffers[2].cpu().numpy().reshape(1, -1)
            trt_order = np.argsort(-trt_scores[0])
            trt_l = trt_labels[0, trt_order]
            trt_s = trt_scores[0, trt_order]
            print(f"\n   --- FP16 精度验证 ---")
            print(f"   Top-1: label={int(trt_l[0])}, score={trt_s[0]:.6f}")
            print(f"   PyTorch Top-1: label={int(pt_l_k[0])}, score={pt_s_k[0]:.6f}")
            print(f"   score 差异: {abs(trt_s[0] - pt_s_k[0]):.6e}")
            print(f"   label 一致: {'✅' if trt_l[0] == pt_l_k[0] else '❌'}")

        print(f"   迭代次数: {iters}")
        print(f"   平均耗时: {trt_per:.2f} ms ({1000/trt_per:.2f} FPS)")
        trt_speedup = pt_elapsed / trt_elapsed

        trt_result = {
            "elapsed": trt_elapsed,
            "iters": iters,
            "speedup": trt_speedup,
        }

    except ImportError as e:
        print(f"\n⚠️ 跳过 TensorRT：{e}")
        print(f"   如需 TensorRT 测试: pip install tensorrt")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n⚠️ TensorRT 测试失败: {e}")

    # ── 7. 最终速度对比表 ──
    print(f"\n{'='*60}")
    print(f"   最终速度对比 (batch=1)")
    print(f"{'='*60}")
    print(f"   {'后端':<20} | {'单次耗时':<12} | {'FPS':<10} | {'相对 PyTorch':<12}")
    print(f"   {'-'*20}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")
    print(f"   {'PyTorch FP32':<20} | {pt_elapsed/iters*1000:8.2f} ms | {iters/pt_elapsed:7.2f} | {'1.00x':<12}")
    print(f"   {f'ONNX-{provider_name}':<20} | {onnx_elapsed/iters*1000:8.2f} ms | {iters/onnx_elapsed:7.2f} | {speedup:.2f}x{'':<8}")
    if trt_result:
        print(f"   {'TensorRT FP16':<20} | {trt_per:8.2f} ms | {1000/trt_per:7.2f} | {trt_result['speedup']:.2f}x{'':<8}")

    # ── 8. 最终产物信息 ──
    print(f"\n{'='*60}")
    print(f"   最终 ONNX 模型信息")
    print(f"{'='*60}")
    print(f"   路径: {final_path}")
    print(f"   大小: {final_size:.2f} MB")
    print(f"   精度: FP32 (float32)")
    print(f"   输入: images [batch, 3, {INPUT_H}, {INPUT_W}] float32")
    print(f"         orig_target_sizes [batch, 2] float32")
    print(f"   输出: labels [batch, 300] int64")
    print(f"         boxes [batch, 300, 4] float32")
    print(f"         scores [batch, 300] float32")
    print(f"\n   推理代码示例:")
    print(f"   ```python")
    print(f"   import onnxruntime as ort")
    print(f"   sess_options = ort.SessionOptions()")
    print(f"   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL")
    print(f"   session = ort.InferenceSession('{final_path}', sess_options,")
    print(f"       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])")
    print(f"   result = session.run(['labels', 'boxes', 'scores'], {{")
    print(f"       'images': img_np.astype(np.float16),")
    print(f"       'orig_target_sizes': np.array([[W, H]], dtype=np.float16)")
    print(f"   }})")
    print(f"   labels, boxes, scores = result")
    print(f"   ```")


if __name__ == "__main__":
    main()
