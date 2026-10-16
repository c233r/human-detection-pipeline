"""ONNX 推理瓶颈定位 benchmark。

分别测：
1. GPU transform (resize) 耗时
2. .cpu().numpy() D2H 拷贝耗时
3. ORT session.run 耗时
4. 合计 vs 实测
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/data2/caimoyao/EdgeCrafter-main/ecdetseg")

ONNX_PATH = "/data2/caimoyao/EdgeCrafter-main/ecdetseg/models/ecdet_l_1080x1920_fp16_quant.onnx"
ONNX_INPUT_DTYPE = "float16"

device = torch.device("cuda:0")


def bench(fn, iters=10, warmup=3, desc=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    avg = sum(ts) / len(ts)
    print(f"  {desc}: {avg:.2f} ms")
    return avg


def main():
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        ONNX_PATH, sess_options,
        providers=[("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
        })],
    )
    print(f"providers: {session.get_providers()}")

    for batch in [1, 4, 8]:
        print(f"\n===== batch={batch} =====")
        # 模拟 api_service 输入：原始分辨率 1080x1920 uint8
        src_h, src_w = 1080, 1920
        batch_nhwc = torch.randint(0, 256, (batch, src_h, src_w, 3),
                                   dtype=torch.uint8, device=device)

        # 1. permute + contiguous + cuda（模拟预取线程的 H2D）
        gpu_tensor = batch_nhwc.permute(0, 3, 1, 2).contiguous()

        # 2. GPU transform（resize）
        import torchvision.transforms.v2 as v2
        gpu_transforms = v2.Compose([
            v2.Resize((1080, 1920)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformed = gpu_transforms(gpu_tensor)
        print(f"  transformed shape: {tuple(transformed.shape)}, dtype: {transformed.dtype}")

        # 3. D2H 拷贝
        bench(lambda: transformed.cpu().numpy(), desc=".cpu().numpy() D2H")

        # 4. ORT run（dtype 匹配模型）
        orig_sizes_np = np.array([[1920, 1080]] * batch, dtype=np.float32)
        def run_ort():
            input_np = transformed.cpu().numpy()
            if ONNX_INPUT_DTYPE == "float16":
                input_np = input_np.astype(np.float16)
                orig_sz = orig_sizes_np.astype(np.float16)
            else:
                orig_sz = orig_sizes_np
            return session.run(["labels", "boxes", "scores"],
                               {"images": input_np, "orig_target_sizes": orig_sz})
        bench(run_ort, desc="full: transform+D2H+ORT")

        # 5. 只测 ORT run（固定 numpy 输入，dtype 匹配模型）
        input_np = transformed.cpu().numpy()
        if ONNX_INPUT_DTYPE == "float16":
            input_np = input_np.astype(np.float16)
        bench(lambda: session.run(["labels", "boxes", "scores"],
                                  {"images": input_np, "orig_target_sizes": orig_sizes_np.astype(np.float16)}),
              desc=f"ORT run only ({ONNX_INPUT_DTYPE} input)")

        # 6. 完整流水线模拟
        def full_pipeline():
            t = gpu_transforms(gpu_tensor)
            inp = t.cpu().numpy()
            if ONNX_INPUT_DTYPE == "float16":
                inp = inp.astype(np.float16)
            return session.run(["labels", "boxes", "scores"],
                               {"images": inp,
                                "orig_target_sizes": orig_sizes_np.astype(np.float16)})
        bench(full_pipeline, desc=f"full pipeline ({ONNX_INPUT_DTYPE} input)")


if __name__ == "__main__":
    main()
