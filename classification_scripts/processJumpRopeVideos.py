import os
import cv2
from tqdm import tqdm
baseDir = "/data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos"
saveImageDir = "/data1/xieyangyang/DataSet/SportsDataset/JumpRopeVideos/videosPartThreeFrames"
os.makedirs(saveImageDir, exist_ok=True)
dirList = os.listdir(baseDir)
saveIndex = -1
for dirName in dirList:
    if dirName not in ["videosPartThree", ]:
        continue
    subDir = os.path.join(baseDir, dirName)
    videoList = os.listdir(subDir)
    # print(len(videoList))
    # continue
    for videoName in tqdm(videoList):
        videoPath = os.path.join(subDir, videoName)
        # print(videoPath)
        cap = cv2.VideoCapture(videoPath)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frameCount):
            ret, frame = cap.read()
            if not ret:
                break
            if i < 300 or i > frameCount // 3:
                continue
            if i % 2 == 0:
                continue
            saveIndex += 1
            saveName = videoName.split(".")[0] + "_" + str(saveIndex).zfill(8) + ".jpg"
            savePath = os.path.join(saveImageDir, saveName)
            cv2.imwrite(savePath, frame)