import math
import numpy as np
from numpy import cos, sin
import cv2

class Colors:
    pose_palette = np.array([[100, 100, 100], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                             [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255],
                             [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
                             [128, 0, 128], [0, 128, 128], [128, 128, 128], [192, 192, 192],
                             [0, 0, 0]], dtype=np.uint8)

class Plot:
    skeleton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]

    limb_color = Colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    @classmethod
    def plot_3axis_Zaxis(cls, img, yaw, pitch, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2):
        # Input is a cv2 image
        # pose_params: (pitch, yaw, roll, tdx, tdy)
        # Where (tdx, tdy) is the translation of the face.
        # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

        p = pitch * np.pi / 180
        y = -(yaw * np.pi / 180)
        r = roll * np.pi / 180
        
        if tdx != None and tdy != None:
            face_x = tdx
            face_y = tdy
        else:
            height, width = img.shape[:2]
            face_x = width / 2
            face_y = height / 2

        # X-Axis (pointing to right) drawn in red
        x1 = size * (cos(y) * cos(r)) + face_x
        y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
        
        # Y-Axis (pointing to down) drawn in green
        x2 = size * (-cos(y) * sin(r)) + face_x
        y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
        
        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(y)) + face_x
        y3 = size * (-cos(y) * sin(p)) + face_y

        # Plot head oritation line in black
        # scale_ratio = 5
        scale_ratio = 2
        base_len = math.sqrt((face_x - x3)**2 + (face_y - y3)**2)
        if face_x == x3:
            endx = tdx
            if face_y < y3:
                if limited:
                    endy = tdy + (y3 - face_y) * scale_ratio
                else:
                    endy = img.shape[0]
            else:
                if limited:
                    endy = tdy - (face_y - y3) * scale_ratio
                else:
                    endy = 0
        elif face_x > x3:
            if limited:
                endx = tdx - (face_x - x3) * scale_ratio
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endx = 0
                endy = tdy - (face_y - y3) / (face_x - x3) * tdx
        else:
            if limited:
                endx = tdx + (x3 - face_x) * scale_ratio
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endx = img.shape[1]
                endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
        # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,0,0), 2)
        # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (255,255,0), 2)
        cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,255,255), thickness)

        # X-Axis pointing to right. drawn in red
        cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),thickness)
        # Y-Axis pointing to down. drawn in green    
        cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,255,0),thickness)
        # Z-Axis (out of the screen) drawn in blue
        cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness)

        return img

    @classmethod
    def plot_angles(cls, image, box, angle):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        pitch, yaw, roll = angle
        pitch = (pitch - 0.5) * 180
        yaw = (yaw - 0.5) * 360
        roll = (roll - 0.5) * 180    
        # print(f"==> pitch: {pitch:.2f}, yaw: {yaw:.2f}, roll: {roll:.2f}")
        image = cls.plot_3axis_Zaxis(
            image, yaw, pitch, roll, 
            tdx=(x1 + x2) / 2, tdy=(y1 + y2) / 2, 
            size=max(y2 - y1, x2 - x1) * 0.8, thickness=1
        )
        return image