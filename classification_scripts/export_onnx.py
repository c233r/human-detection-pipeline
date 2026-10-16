"""
将训练好的 ConvNeXt-base 分类模型导出为 FP16 ONNX 格式。
包含 onnxsim 简化、onnxoptimizer 优化、算子融合可视化。
用于 hand_crop_service_kafka.py 中的推理加速。

用法:
    cd /data2/caimoyao/Jump_rope/classification/scripts_cmy
    python export_onnx.py
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.train_utils import get_info, file2dict
from models.build import BuildNet


# ==================== 配置 ====================
CLASSES_MAP = '/data2/caimoyao/Jump_rope/classification/datas/annotations.txt'
CONFIG_PATH = '/data2/caimoyao/Jump_rope/classification/logs/ConvNeXt/2026-07-16-11-48-44/convnext_base_fix.py'
CKPT_PATH = '/data2/caimoyao/Jump_rope/classification/logs/ConvNeXt/2026-07-16-11-48-44/Val_Epoch096-Acc99.566.pth'

# ONNX 导出参数
ONNX_OUTPUT_BASE = '/data2/caimoyao/Jump_rope/classification/logs/ConvNeXt/2026-07-16-11-48-44/convnext_base_fix'
OPSET_VERSION = 17
# 匹配 val_pipeline: Resize(size=(208, -1), adaptive_side='long')
# 配合 _pad_square 后输出为 208x208
DUMMY_INPUT_SIZE = 208


def get_model_stats(onnx_model):
    """获取 ONNX 模型统计信息"""
    node_count = len(onnx_model.graph.node)
    initializer_count = len(onnx_model.graph.initializer)
    input_count = len(onnx_model.graph.input)
    output_count = len(onnx_model.graph.output)
    
    ops_count = {}
    for node in onnx_model.graph.node:
        op_type = node.op_type
        ops_count[op_type] = ops_count.get(op_type, 0) + 1
    
    return {
        'nodes': node_count,
        'initializers': initializer_count,
        'inputs': input_count,
        'outputs': output_count,
        'op_types': ops_count,
    }


def print_model_stats(name, stats, file_size_mb=None):
    """打印模型统计信息"""
    print(f"\n{'='*60}")
    print(f"   {name}")
    print(f"{'='*60}")
    if file_size_mb:
        print(f"   文件大小: {file_size_mb:.2f} MB")
    print(f"   节点总数: {stats['nodes']}")
    print(f"   初始化器: {stats['initializers']}")
    print(f"   输入/输出: {stats['inputs']} / {stats['outputs']}")
    print(f"   算子类型分布 (前10):")
    for op_type, count in sorted(stats['op_types'].items(), key=lambda x: -x[1])[:10]:
        print(f"     {op_type:20s}: {count}")


def export():
    print("=" * 60)
    print("   ConvNeXt → FP16 ONNX 导出 + 优化")
    print("=" * 60)

    # 1. 加载类别信息
    print(f"\n[1/7] 加载类别信息...")
    classes_names, label_names = get_info(CLASSES_MAP)
    print(f"      {len(classes_names)} 个类别: {classes_names}")

    # 2. 加载模型配置
    print(f"\n[2/7] 加载模型配置...")
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(CONFIG_PATH)

    # 3. 构建网络模型
    print(f"\n[3/7] 构建网络模型...")
    net = BuildNet(model_cfg)
    print(f"      参数数量: {sum(p.numel() for p in net.parameters()):,}")

    # 4. 加载权重
    print(f"\n[4/7] 加载权重文件: {os.path.basename(CKPT_PATH)}")
    net.eval()
    state_dict = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"      ⚠️ 缺失 {len(missing)} 个参数 (可忽略)")
    if unexpected:
        print(f"      ⚠️ 多余 {len(unexpected)} 个参数 (可忽略)")
    model = net

    # 5. 包装为纯推理模型
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            feat = self.net.extract_feat(x)
            return self.net.head.simple_test(feat)

    print(f"\n[5/7] 转换为 FP16 精度并移至 GPU...")
    device = torch.device("cuda:0")
    wrapped = InferenceWrapper(model).half().to(device)

    # 6. 导出 FP16 ONNX
    onnx_raw = f"{ONNX_OUTPUT_BASE}_fp16_raw.onnx"
    print(f"\n[6/7] 导出 FP16 ONNX (opset={OPSET_VERSION}) → {onnx_raw}")

    dummy_input = torch.randn(1, 3, DUMMY_INPUT_SIZE, DUMMY_INPUT_SIZE, dtype=torch.float16, device=device)
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size'},
    }

    torch.onnx.export(
        wrapped,
        dummy_input,
        onnx_raw,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    print(f"\n✅ 原始 ONNX 导出成功")

    # 7. 加载原始模型并统计
    import onnx
    raw_model = onnx.load(onnx_raw)
    raw_stats = get_model_stats(raw_model)
    raw_size = os.path.getsize(onnx_raw) / 1024 / 1024
    print_model_stats("原始 ONNX 模型", raw_stats, raw_size)

    # 验证 dynamic batch & spatial 维度
    raw_input = raw_model.graph.input[0]
    raw_shape = raw_input.type.tensor_type.shape
    dim_names = ['batch', 'channel', 'height', 'width']
    for i, dim in enumerate(raw_shape.dim):
        if dim.dim_param:
            print(f"   ✅ {dim_names[i]} 维度动态: {dim.dim_param}")
        elif i == 1:
            print(f"   ✅ {dim_names[i]} 维度固定: {dim.dim_value}")
        else:
            print(f"   ⚠️ {dim_names[i]} 维度被折叠为固定值 {dim.dim_value}")

    # 8. onnxsim 简化（仅用于参考，简化可能折叠 dynamic batch）
    onnx_simplified = f"{ONNX_OUTPUT_BASE}_fp16_simplified.onnx"
    print(f"\n[7.1/7] 使用 onnxsim 简化模型（仅参考）...")
    try:
        import onnxsim
        # 不传入 input_shapes，避免折叠 dynamic batch
        simplified_model, check = onnxsim.simplify(raw_model)
        onnx.save(simplified_model, onnx_simplified)
        sim_stats = get_model_stats(simplified_model)
        sim_size = os.path.getsize(onnx_simplified) / 1024 / 1024
        print_model_stats("onnxsim 简化后", sim_stats, sim_size)
        print(f"\n✅ onnxsim 简化完成，节点数: {raw_stats['nodes']} → {sim_stats['nodes']} ({(1-sim_stats['nodes']/raw_stats['nodes'])*100:.1f}% 减少)")

        # 检查简化后 all dims 保留
        sim_input = simplified_model.graph.input[0]
        sim_dims = sim_input.type.tensor_type.shape.dim
        dynamic_ok = all(dim.dim_param for i, dim in enumerate(sim_dims) if i != 1)
        if dynamic_ok:
            print(f"   ✅ dynamic batch/spatial 保留: batch={sim_dims[0].dim_param}, h={sim_dims[2].dim_param}, w={sim_dims[3].dim_param}")
            current_model = simplified_model
            current_path = onnx_simplified
        else:
            print(f"   ⚠️ onnxsim 折叠了动态维度，回退到原始模型")
            current_model = raw_model
            current_path = onnx_raw
    except Exception as e:
        print(f"⚠️ onnxsim 简化失败: {e}")
        print(f"   跳过 onnxsim，继续使用原始模型")
        current_model = raw_model
        current_path = onnx_raw
        sim_stats = raw_stats
        sim_size = raw_size

    # 9. onnxoptimizer 优化（仅用于参考，可能折叠 dynamic batch）
    onnx_optimized = f"{ONNX_OUTPUT_BASE}_fp16_optimized.onnx"
    print(f"\n[7.2/7] 使用 onnxoptimizer 优化模型（仅参考）...")
    try:
        import onnxoptimizer
        passes = onnxoptimizer.get_fuse_and_elimination_passes()
        optimized_model = onnxoptimizer.optimize(current_model, passes)
        onnx.save(optimized_model, onnx_optimized)
        opt_stats = get_model_stats(optimized_model)
        opt_size = os.path.getsize(onnx_optimized) / 1024 / 1024
        print_model_stats("onnxoptimizer 优化后", opt_stats, opt_size)
        print(f"\n✅ onnxoptimizer 优化完成，节点数: {sim_stats['nodes']} → {opt_stats['nodes']}")

        # 检查优化后 all dims 保留
        opt_input = optimized_model.graph.input[0]
        opt_dims = opt_input.type.tensor_type.shape.dim
        opt_ok = all(dim.dim_param for i, dim in enumerate(opt_dims) if i != 1)
        if opt_ok:
            print(f"   ✅ dynamic batch/spatial 保留: batch={opt_dims[0].dim_param}, h={opt_dims[2].dim_param}, w={opt_dims[3].dim_param}")
            final_model = optimized_model
            final_path = onnx_optimized
        else:
            print(f"   ⚠️ onnxoptimizer 折叠了动态维度，回退到之前模型")
            final_model = current_model
            final_path = current_path
    except Exception as e:
        print(f"⚠️ onnxoptimizer 优化失败: {e}")
        print(f"   跳过 onnxoptimizer，继续使用之前模型")
        final_model = current_model
        final_path = current_path

    # 10. 验证最终 ONNX
    print(f"\n[7.3/7] 验证最终 ONNX 模型...")
    onnx.checker.check_model(final_model)
    print(f"✅ ONNX 验证通过")

    # 11. 复制到最终输出路径
    final_output = f"{ONNX_OUTPUT_BASE}_fp16.onnx"
    os.system(f"cp {final_path} {final_output}")
    print(f"\n✅ 最终模型已保存: {final_output}")

    # 12. 优化对比总结
    print(f"\n{'='*60}")
    print(f"   优化对比总结")
    print(f"{'='*60}")
    print(f"   阶段           | 节点数 | 文件大小")
    print(f"   ----------------|--------|----------")
    print(f"   原始模型       | {raw_stats['nodes']:6d} | {raw_size:6.2f} MB")
    print(f"   onnxsim 简化   | {sim_stats['nodes']:6d} | {sim_size:6.2f} MB")
    print(f"   onnxoptimizer  | {opt_stats['nodes']:6d} | {opt_size:6.2f} MB")
    print(f"\n   累计优化: 节点数减少 {(1-opt_stats['nodes']/raw_stats['nodes'])*100:.1f}%")

    # 13. ONNX Runtime GPU 推理测试
    print(f"\n{'='*60}")
    print(f"   原始 PyTorch vs ONNX 推理对比")
    print(f"{'='*60}")

    import onnxruntime as ort
    import time

    # ── 固定随机种子，保证输入一致 ──
    np.random.seed(42)
    torch.manual_seed(42)
    test_input_np = np.random.randn(1, 3, DUMMY_INPUT_SIZE, DUMMY_INPUT_SIZE).astype(np.float16)
    test_input_torch = torch.from_numpy(test_input_np).to(device)

    # ── PyTorch 推理 ──
    print(f"\n--- PyTorch FP16 (GPU) ---")
    with torch.no_grad():
        torch_out = wrapped(test_input_torch)
    torch_out_cpu = torch_out.float().cpu().numpy()
    print(f"   输出: {torch_out_cpu[0]}")
    print(f"   预测类别: {torch_out_cpu[0].argmax()} (置信度: {torch_out_cpu[0].max():.6f})")

    # PyTorch 速度测试
    warmup = 5
    for _ in range(warmup):
        with torch.no_grad():
            _ = wrapped(test_input_torch)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            _ = wrapped(test_input_torch)
    torch.cuda.synchronize()
    pt_elapsed = time.perf_counter() - start
    print(f"   平均耗时: {pt_elapsed/50*1000:.2f} ms")
    print(f"   吞吐量:   {50/pt_elapsed:.2f} FPS")

    # ── ONNX Runtime 推理 ──
    print(f"\n--- ONNX FP16 (CUDAExecutionProvider) ---")
    available_providers = ort.get_available_providers()
    print(f"   可用 providers: {available_providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(final_output, sess_options, providers=providers)

    onnx_result = session.run(['output'], {'input': test_input_np})
    onnx_out = onnx_result[0]
    print(f"   输出: {onnx_out[0]}")
    print(f"   预测类别: {onnx_out[0].argmax()} (置信度: {onnx_out[0].max():.6f})")

    # ONNX 速度测试
    for _ in range(warmup):
        session.run(['output'], {'input': test_input_np})

    start = time.perf_counter()
    for _ in range(50):
        session.run(['output'], {'input': test_input_np})
    onnx_elapsed = time.perf_counter() - start
    print(f"   平均耗时: {onnx_elapsed/50*1000:.2f} ms")
    print(f"   吞吐量:   {50/onnx_elapsed:.2f} FPS")

    # ── 对比总结 ──
    diff = np.abs(torch_out_cpu - onnx_out).max()
    speedup = pt_elapsed / onnx_elapsed
    same_class = torch_out_cpu[0].argmax() == onnx_out[0].argmax()

    print(f"\n{'='*60}")
    print(f"   对比总结")
    print(f"{'='*60}")
    print(f"   最大数值差异: {diff:.6e}")
    print(f"   预测类别一致: {'✅' if same_class else '❌'}")
    print(f"   速度提升:     {speedup:.2f}x")
    print(f"   PyTorch:      {pt_elapsed/50*1000:.2f} ms / {50/pt_elapsed:.2f} FPS")
    print(f"   ONNX:         {onnx_elapsed/50*1000:.2f} ms / {50/onnx_elapsed:.2f} FPS")

    # ── 多 batch 测试（验证 dynamic batch） ──
    print(f"\n{'='*60}")
    print(f"   多 Batch 测试 (batch=100)")
    print(f"{'='*60}")
    batch_size = 100
    multi_input = np.random.randn(batch_size, 3, DUMMY_INPUT_SIZE, DUMMY_INPUT_SIZE).astype(np.float16)

    # warmup
    for _ in range(3):
        session.run(['output'], {'input': multi_input[:1]})

    start = time.perf_counter()
    onnx_multi_out = session.run(['output'], {'input': multi_input})[0]
    onnx_multi_elapsed = time.perf_counter() - start

    print(f"   输入形状: ({batch_size}, 3, {DUMMY_INPUT_SIZE}, {DUMMY_INPUT_SIZE})")
    print(f"   输出形状: {onnx_multi_out.shape}")
    print(f"   单次耗时: {onnx_multi_elapsed*1000:.2f} ms")
    print(f"   每张耗时: {onnx_multi_elapsed/batch_size*1000:.2f} ms")
    print(f"   吞吐量:   {batch_size/onnx_multi_elapsed:.2f} FPS")
    print(f"   预测分布: 类别0={(onnx_multi_out[:,0] > onnx_multi_out[:,1]).sum()} "
           f"类别1={(onnx_multi_out[:,1] > onnx_multi_out[:,0]).sum()}")

    print(f"\n✅ 全部完成!")

    print(f"\n{'='*60}")
    print(f"   最终 ONNX 模型信息")
    print(f"{'='*60}")
    final_size = os.path.getsize(final_output) / 1024 / 1024
    print(f"   路径: {final_output}")
    print(f"   大小: {final_size:.2f} MB")
    print(f"   精度: FP16 (float16)")
    print(f"   输入: (batch, 3, {DUMMY_INPUT_SIZE}, {DUMMY_INPUT_SIZE})")
    print(f"   输出: (batch, {len(classes_names)})")
    print(f"\n   推理代码示例:")
    print(f"   ```python")
    print(f"   import onnxruntime as ort")
    print(f"   sess_options = ort.SessionOptions()")
    print(f"   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL")
    print(f"   session = ort.InferenceSession('{final_output}', sess_options,")
    print(f"       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])")
    print(f"   result = session.run(['output'], {{'input': img_np.astype(np.float16)}})")
    print(f"   ```")


if __name__ == "__main__":
    export()
