"""将 FP32 ONNX 模型真正转换为 FP16 图（官方工具）。

用 onnxruntime.transformers.float16.convert_float_to_float16，
比手动转换可靠：自动处理 LayerNorm/特殊节点，插入必要 Cast。
"""
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

SRC = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/models/ecdet_l_1080x1920_fp32_raw.onnx"
DST = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/models/ecdet_l_1080x1920_fp16_quant.onnx"

print(f"源模型: {SRC}")
m = onnx.load(SRC)
for inp in m.graph.input:
    print(f"  输入: {inp.name}, dtype={inp.type.tensor_type.elem_type}")

print(f"\n转换 FP16 → {DST}")
m_fp16 = convert_float_to_float16(
    m,
    keep_io_types=False,       # 输入/输出也转 FP16
    op_block_list=None,        # 不排除任何算子
    node_block_list=None,
)
onnx.save(m_fp16, DST)
print("✅ 转换完成")

# 验证
m2 = onnx.load(DST)
for inp in m2.graph.input:
    print(f"  输入: {inp.name}, dtype={inp.type.tensor_type.elem_type} (1=FP32, 10=FP16)")
for out in m2.graph.output:
    print(f"  输出: {out.name}, dtype={out.type.tensor_type.elem_type} (1=FP32, 10=FP16)")
print(f"  大小: {m2.ByteSize() / 1024 / 1024:.1f} MB")
