import os
import cv2
import onnx
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

def postprocess_numpy(simcc_x_result, simcc_y_result, pad_x, pad_y, scale):
    """
    使用 NumPy 进行向量化加速，不再需要写 for 循环
    """
    # 如果带有 batch 维度 (1, K, W)，将其去掉
    if simcc_x_result.ndim == 3:
        simcc_x_result = simcc_x_result[0]
        simcc_y_result = simcc_y_result[0]
        
    # 直接在最后的维度上求最大值的索引与最大值本身
    max_x_pos = np.argmax(simcc_x_result, axis=1)
    score_x = np.max(simcc_x_result, axis=1)
    
    max_y_pos = np.argmax(simcc_y_result, axis=1)
    score_y = np.max(simcc_y_result, axis=1)
    
    # //2 相当于 C++ 中的整数除法
    pose_x = max_x_pos // 2
    pose_y = max_y_pos // 2
    
    # 逐元素取 x 和 y 得分的较大值
    scores = np.maximum(score_x, score_y)
    
    keypoint_result = []
    for px, py, s in zip(pose_x, pose_y, scores):
        # 将坐标从缩放后的图像映射回原始图像
        px = (px - pad_x) / scale
        py = (py - pad_y) / scale
        # print(f"pose_x: {px}, pose_y: {py}")
        keypoint_result.append({
            "x": int(px),
            "y": int(py),
            "score": float(s)
        })
        
    return keypoint_result


def preprocess_image(img, input_size=(192, 256)):
    """
    对输入图片进行预处理，使用保持宽高比的缩放 (Letterbox)，并在不足处填充 114。
    :param image_path: 图片路径
    :param input_size: 模型输入尺寸 (宽W, 高H)，对应 rtmpose-m 的设定
    """
    # 1. 读取图片 (默认BGR格式)
    # img = cv2.imread(image_path)
    # if img is None:
    #     raise ValueError(f"无法读取图片，请检查路径: {image_path}")

    # 2. 保持宽高比缩放 (Letterbox)
    orig_h, orig_w = img.shape[:2]
    target_w, target_h = input_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # 创建填充 114 的背景图
    img_padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    # 将缩放后的图片居中贴到背景图上
    pad_y = (target_h - new_h) // 2
    pad_x = (target_w - new_w) // 2
    img_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

    # 3. BGR 转 RGB (根据需要启用)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    # 4. 转换为 float32 并进行均值方差归一化
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    img_normalized = (img_rgb.astype(np.float32) - mean) / std

    # 5. HWC 转换为 CHW (Channel, Height, Width)
    img_chw = img_normalized.astype(np.float32).transpose(2, 0, 1)

    # 6. 增加 Batch 维度 (1, C, H, W)
    input_tensor = np.expand_dims(img_chw, axis=0)
    
    return input_tensor, pad_x, pad_y, scale


class RTMPoseModel:
    def __init__(self, model_path: str, input_size: tuple = (288, 384), device_id: int = 2):
        self.model_path = model_path
        self.input_size = input_size
        self.device_id = device_id
        self.session = None
        self.input_name = None
        self._load_model()

    def _load_model(self):
        print(f"正在加载 RTMPose ONNX 模型: {self.model_path} (GPU {self.device_id})")
        providers = [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"模型加载完成，输入节点: {self.input_name}")

    def inference(self, img):
        input_tensor, pad_x, pad_y, scale = preprocess_image(img, input_size=self.input_size)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        out1, out2 = outputs[0], outputs[1]
        if out1.shape[-1] > out2.shape[-1]:
            simcc_y, simcc_x = out1, out2
        else:
            simcc_x, simcc_y = out1, out2
        keypoints = postprocess_numpy(simcc_x, simcc_y, pad_x, pad_y, scale)
        bodykepoint = keypoints[:23]
        letfhand = keypoints[100] if len(keypoints) > 100 else {"x": 0, "y": 0, "score": 0.0}
        righthand = keypoints[121] if len(keypoints) > 121 else {"x": 0, "y": 0, "score": 0.0}
        merge_keypoint = bodykepoint + [letfhand, righthand]
        return merge_keypoint


# def main():
#     onnx_model_path = 'weights/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'
#     input_size = (288, 384) 
    
#     model = RTMPoseModel(onnx_model_path, input_size=input_size)
    
#     image_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages1/noHoldRope"
#     save_label_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages1/noHoldRope_labels"
#     os.makedirs(save_label_dir, exist_ok=True)
#     image_list = os.listdir(image_dir)
#     print(f"共找到 {len(image_list)} 张图片，正在处理...")
#     for image_name in tqdm(image_list[:], desc="Processing images"):
#         image_path = os.path.join(image_dir, image_name)
#         image = cv2.imread(image_path)
#         try:
#             height, width, _ = image.shape
#         except Exception as e:
#             print(f"无法读取图片 {image_path}，跳过。错误信息: {e}")
#             continue
        
#         merge_keypoint = model.inference(image)
        
#         with open(os.path.join(save_label_dir, image_name.replace(".jpg", ".txt")), "w") as f:
#             for kp in merge_keypoint:
#                 f.write(f"{kp['x']} {kp['y']} {kp['score']}\n")


# if __name__ == '__main__':
#     main()