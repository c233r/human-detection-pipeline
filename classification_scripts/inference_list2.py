from argparse import ArgumentParser
import os
import cv2
from tqdm import tqdm
import sys
import numpy as np
import random
random.seed(42)
sys.path.insert(0,os.getcwd())
import torch

from utils.inference import inference_model, init_model, show_result_pyplot
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

def main():
    image_dir = "/data1/xieyangyang/DataSet/POSE_DET/ORIGIN/jumpRopeCounter/data_sub8"
    save_hold_rope_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages/holdRope"
    save_show_hold_rope_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages/holdRope_show"
    save_hold_rope_label_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages/holdRope_labels"
    
    save_no_hold_rope_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages/noHoldRope"
    save_show_no_hold_rope_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages/noHoldRope_show"
    save_no_hold_rope_label_dir = "/data2/xieyangyang/JumpRopeDataset/CropImages/noHoldRope_labels"
    os.makedirs(save_hold_rope_dir, exist_ok=True)
    os.makedirs(save_show_hold_rope_dir, exist_ok=True)
    os.makedirs(save_hold_rope_label_dir, exist_ok=True)
    os.makedirs(save_no_hold_rope_dir, exist_ok=True)
    os.makedirs(save_show_no_hold_rope_dir, exist_ok=True)
    os.makedirs(save_no_hold_rope_label_dir, exist_ok=True)
    
    classes_map = '/data1/xieyangyang/Projects/classification/datas/annotations.txt'
    config = '/data1/xieyangyang/Projects/classification/models/resnet/resnet18_fix.py'
    classes_names, label_names = get_info(classes_map)
    # build the model from a config file and a checkpoint file
    model_cfg, train_pipeline, val_pipeline,data_cfg, lr_config, optimizer_cfg = file2dict(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')
    usedImageList = []
    videoList = os.listdir(image_dir)
    for video_name in tqdm(videoList):
        viddo_path = os.path.join(image_dir, video_name)
        fileList = os.listdir(viddo_path)
        for idx, file_name in enumerate(fileList):
            if idx % 5 != 0:
                continue
            # ratio = random.uniform(1.1, 1.25)
            if file_name.endswith('.jpg'):
                image_path = os.path.join(viddo_path, file_name)
                label_path = os.path.join(viddo_path, file_name.replace('.jpg', '.txt'))
                if not os.path.exists(label_path):
                    continue
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                line = lines[0].strip().split()
                x1 = float(line[5])
                y1 = float(line[6])
                x2 = float(line[7])
                y2 = float(line[8])
                points = np.array(line[9:], dtype=float).reshape(-1, 2)
                leftShoulder = points[0]
                rightShoulder = points[1]
                shoudDist = ((leftShoulder[0] - rightShoulder[0]) ** 2 + (leftShoulder[1] - rightShoulder[1]) ** 2) ** 0.5
                shoudDist = max(shoudDist, 64)
                LeftWrist = points[4]
                RightWrist = points[5]

                cropLeftHandXmin = max(0, LeftWrist[0] - shoudDist * 0.8)
                cropLeftHandXmax = min(width, LeftWrist[0] + shoudDist * 0.8)
                cropLeftHandYmin = max(0, LeftWrist[1] - shoudDist * 0.8)
                cropLeftHandYmax = min(height, LeftWrist[1] + shoudDist * 0.8)
                
                cropRightHandXmin = max(0, RightWrist[0] - shoudDist * 0.8)
                cropRightHandXmax = min(width, RightWrist[0] + shoudDist * 0.8)
                cropRightHandYmin = max(0, RightWrist[1] - shoudDist * 0.8)
                cropRightHandYmax = min(height, RightWrist[1] + shoudDist * 0.8)
                
                cropImageLeftXmin = max(min(min(cropLeftHandXmin, cropRightHandXmin), x1), 0)
                cropImageXmax = min(max(max(cropLeftHandXmax, cropRightHandXmax), x2), width)
                cropImageYmin = max(min(min(cropLeftHandYmin, cropRightHandYmin), y1), 0)
                cropImageYmax = min(max(max(cropLeftHandYmax, cropRightHandYmax), y2), height)

                crop_image = image[int(cropImageYmin):int(cropImageYmax), int(cropImageLeftXmin):int(cropImageXmax)]
                save_image = crop_image.copy()
                show_image = crop_image.copy()
                cropLeftHandXmin = cropLeftHandXmin - cropImageLeftXmin
                cropLeftHandXmax = cropLeftHandXmax - cropImageLeftXmin
                cropLeftHandYmin = cropLeftHandYmin - cropImageYmin
                cropLeftHandYmax = cropLeftHandYmax - cropImageYmin
                
                cropRightHandXmin = cropRightHandXmin - cropImageLeftXmin
                cropRightHandXmax = cropRightHandXmax - cropImageLeftXmin
                cropRightHandYmin = cropRightHandYmin - cropImageYmin
                cropRightHandYmax = cropRightHandYmax - cropImageYmin
                cv2.rectangle(show_image, (int(cropLeftHandXmin), int(cropLeftHandYmin)), (int(cropLeftHandXmax), int(cropLeftHandYmax)), (0, 255, 0), 2)
                cv2.rectangle(show_image, (int(cropRightHandXmin), int(cropRightHandYmin)), (int(cropRightHandXmax), int(cropRightHandYmax)), (0, 255, 0), 2)   
                cropLeftHandImage = crop_image[int(cropLeftHandYmin):int(cropLeftHandYmax), int(cropLeftHandXmin):int(cropLeftHandXmax)]
                cropRightHandImage = crop_image[int(cropRightHandYmin):int(cropRightHandYmax), int(cropRightHandXmin):int(cropRightHandXmax)]
                # Pad to square with zeros
                h, w = cropLeftHandImage.shape[:2]
                if h != w:
                    max_side = max(h, w)
                    pad_h = (max_side - h) // 2
                    pad_w = (max_side - w) // 2
                    cropLeftHandImage = cv2.copyMakeBorder(
                        cropLeftHandImage,
                        pad_h, max_side - h - pad_h,
                        pad_w, max_side - w - pad_w,
                        borderType=cv2.BORDER_CONSTANT, value=0
                    )                
                result = inference_model(model, cropLeftHandImage, val_pipeline, classes_names,label_names)
                label = result['pred_label']
                if label == 0:
                    cv2.imwrite(os.path.join(save_hold_rope_dir, file_name.replace('.jpg', '_%d.jpg'%idx)), save_image)
                    cv2.putText(show_image, 'hold_rope', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(save_show_hold_rope_dir, file_name.replace('.jpg', '_%d.jpg'%idx)), show_image)
                    with open(os.path.join(save_hold_rope_label_dir, file_name.replace('.jpg', '_%d.txt'%idx)), 'w') as f:
                        leftbox = [cropLeftHandXmin, cropLeftHandYmin, cropLeftHandXmax, cropLeftHandYmax]
                        # rightbox = [cropRightHandXmin, cropRightHandYmin, cropRightHandXmax, cropRightHandYmax]
                        f.write('hold_rope %d %d %d %d\n' % (leftbox[0], leftbox[1], leftbox[2], leftbox[3]))
                        # f.write('hold_rope %d %d %d %d\n' % (rightbox[0], rightbox[1], rightbox[2], rightbox[3]))
                else:
                    cv2.imwrite(os.path.join(save_no_hold_rope_dir, file_name.replace('.jpg', '_%d.jpg'%idx)), save_image)
                    cv2.putText(show_image, 'no_hold_rope', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(save_show_no_hold_rope_dir, file_name.replace('.jpg', '_%d.jpg'%idx)), show_image)
                    with open(os.path.join(save_no_hold_rope_label_dir, file_name.replace('.jpg', '_%d.txt'%idx)), 'w') as f:
                        leftbox = [cropLeftHandXmin, cropLeftHandYmin, cropLeftHandXmax, cropLeftHandYmax]
                        # rightbox = [cropRightHandXmin, cropRightHandYmin, cropRightHandXmax, cropRightHandYmax]
                        f.write('no_hold_rope %d %d %d %d\n' % (leftbox[0], leftbox[1], leftbox[2], leftbox[3]))
                        # f.write('no_hold_rope %d %d %d %d\n' % (rightbox[0], rightbox[1], rightbox[2], rightbox[3]))
            

if __name__ == '__main__':
    main()
