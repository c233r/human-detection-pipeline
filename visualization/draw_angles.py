import os
import cv2
import numpy as np
from angle_plot import Plot

def parse_label(label_path, img_width, img_height):
    heads = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 8:
                continue
            cls = int(parts[0])
            if cls != 0:
                continue
            x_center = parts[1] * img_width
            y_center = parts[2] * img_height
            width = parts[3] * img_width
            height = parts[4] * img_height
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            pitch = parts[5]
            yaw = parts[6]
            roll = parts[7]
            heads.append({
                'box': [x1, y1, x2, y2],
                'angle': [pitch, yaw, roll]
            })
    return heads

def draw_on_images(image_dir, label_dir, output_dir, max_images=10):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    count = 0
    for img_file in image_files:
        if count >= max_images:
            break
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        heads = parse_label(label_path, width, height)
        if not heads:
            continue
        for head in heads:
            img = Plot.plot_angles(img, head['box'], head['angle'])
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, img)
        print(f'Saved: {output_path}')
        count += 1
    print(f'Done. Processed {count} images.')

if __name__ == '__main__':
    image_dir = '/data2/caimoyao/dataset/images/train'
    label_dir = '/data2/caimoyao/dataset/labels/train'
    output_dir = '/data2/caimoyao/dataset/output_with_angles'
    draw_on_images(image_dir, label_dir, output_dir, max_images=10)