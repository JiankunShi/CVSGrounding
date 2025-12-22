import os
import json
from datetime import datetime
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from pyquaternion import Quaternion

import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# 配置路径（需要自己改成你的路径）
DATAROOT = '/data_volume_3/nuscenes/v1_0/'   # nuscenes数据集根目录
LABEL_PATH = '/data_volume_1/sjk_data/NuscenesGrounding/nuscenes_scene_labels/scene_day_night_labels.json'  # 之前生成的标签文件

# 加载NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)

# 加载标签
with open(LABEL_PATH, 'r') as f:
    scene_labels = json.load(f)

# 获取所有场景名
scene_names = list(scene_labels.keys())

# 获取每个场景对应的第一张图片路径
def get_first_image_path(scene_name):
    # 找到scene
    for scene in nusc.scene:
        if scene['name'] == scene_name:
            first_sample_token = scene['first_sample_token']
            sample = nusc.get('sample', first_sample_token)
            cam_front_token = sample['data']['CAM_FRONT']
            cam_front = nusc.get('sample_data', cam_front_token)
            img_path = os.path.join(DATAROOT, cam_front['filename'])
            return img_path
    return None

# 预处理：所有场景图片路径列表
scene_images = [get_first_image_path(scene_name) for scene_name in scene_names]


# 创建界面
class SceneViewer:
    def __init__(self, master):
        self.master = master
        master.title("NuScenes Day/Night validation")

        self.index = 0  # 当前场景索引

        # 标签：显示当前场景号和标签
        self.info_label = Label(master, text="", font=('Helvetica', 14))
        self.info_label.pack(pady=10)

        # 图像展示
        self.image_label = Label(master)
        self.image_label.pack()

        # 左右按钮
        self.prev_button = Button(master, text="← Last", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.next_button = Button(master, text="Next →", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=20, pady=10)

        self.update_image()

    def update_image(self):
        # 加载图像
        img_path = scene_images[self.index]
        img = Image.open(img_path)

        # 缩放图片，防止太大
        img = img.resize((800, 450), Image.Resampling.LANCZOS)

        # 转成tkinter能识别的格式
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

        # 更新场景信息
        scene_name = scene_names[self.index]
        label = scene_labels.get(scene_name, "unknow")
        self.info_label.configure(text=f"The {self.index + 1}/{len(scene_names)} scene\n{scene_name}\n judge: {label}")

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.update_image()

    def next_image(self):
        if self.index < len(scene_names) - 1:
            self.index += 1
            self.update_image()

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    viewer = SceneViewer(root)
    root.mainloop()
