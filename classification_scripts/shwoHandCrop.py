import os
import cv2
from tqdm import tqdm
imageDir = "/data2/xieyangyang/JumpRopeDataset/CropImages1/noHoldRope"
labelDir = "/data2/xieyangyang/JumpRopeDataset/CropImages1/noHoldRope_labels"
saveImageDir = "/data2/xieyangyang/JumpRopeDataset/CropImages1/noHoldRope_show"
saveLabelDir = "/data2/xieyangyang/JumpRopeDataset/CropImages1/noHoldRope_show_labels"
os.makedirs(saveImageDir, exist_ok=True)
os.makedirs(saveLabelDir, exist_ok=True)
imageList= os.listdir(imageDir)
for index, imageName in enumerate(tqdm(imageList)):
    if not imageName.endswith('.jpg'):
        continue
    imagePath = os.path.join(imageDir, imageName)
    labelPath = os.path.join(labelDir, imageName.replace('.jpg', '.txt'))
    if not os.path.exists(labelPath):
        continue
    image = cv2.imread(imagePath)
    height, width = image.shape[:2]
    with open(labelPath, 'r') as f:
        lines = f.readlines()
    line = lines[0].strip().split()
    
    leftShoulder = lines[5].strip(" \r\n").split(" ")
    leftShoulder = [float(leftShoulder[0]), float(leftShoulder[1])]
    rightShoulder = lines[6].strip(" \r\n").split(" ")
    rightShoulder = [float(rightShoulder[0]), float(rightShoulder[1])]
    lefthand = lines[23].strip(" \r\n").split(" ")
    lefthand = [float(lefthand[0]), float(lefthand[1])]
    righthand = lines[24].strip(" \r\n").split(" ")
    righthand = [float(righthand[0]), float(righthand[1])]
    shoudDist = ((leftShoulder[0] - rightShoulder[0]) ** 2 + (leftShoulder[1] - rightShoulder[1]) ** 2) ** 0.5
    shoudDist = max(shoudDist, 80)
    cropLeftHandXmin = max(0, lefthand[0] - shoudDist * 0.8)
    cropLeftHandXmax = min(width, lefthand[0] + shoudDist * 0.8)
    cropLeftHandYmin = max(0, lefthand[1] - shoudDist * 0.8)
    cropLeftHandYmax = min(height, lefthand[1] + shoudDist * 0.8)
    
    cropRightHandXmin = max(0, righthand[0] - shoudDist * 0.8)
    cropRightHandXmax = min(width, righthand[0] + shoudDist * 0.8)
    cropRightHandYmin = max(0, righthand[1] - shoudDist * 0.8)
    cropRightHandYmax = min(height, righthand[1] + shoudDist * 0.8)
    if index % 1 == 0:
        cv2.rectangle(image, (int(cropLeftHandXmin), int(cropLeftHandYmin)), (int(cropLeftHandXmax), int(cropLeftHandYmax)), (0, 255, 0), 2)
        cv2.rectangle(image, (int(cropRightHandXmin), int(cropRightHandYmin)), (int(cropRightHandXmax), int(cropRightHandYmax)), (0, 255, 0), 2)
        savePath = os.path.join(saveImageDir, imageName)
        cv2.imwrite(savePath, image)
    with open(os.path.join(saveLabelDir, imageName.replace('.jpg', '.txt')), 'w') as f:
        f.write('left_hand %d %d %d %d\n' % (cropLeftHandXmin, cropLeftHandYmin, cropLeftHandXmax, cropLeftHandYmax))
        f.write('right_hand %d %d %d %d\n' % (cropRightHandXmin, cropRightHandYmin, cropRightHandXmax, cropRightHandYmax))