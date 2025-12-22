#这个文件是计算 /data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_train_data.pkl数据集(与/data_volume_1/sjk_data/NuscenesGrounding/all_train_data.pkl相同)  和
# /data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data_st_0_val.pkl 数据集的2D和3D IOU
#其中2D IOU是通过GroundingDINO计算的，3D IOU是通过StreamPERT的预训练模型（）计算的

#思路：
# ①获取lidar和camera图片，然后分别在图片和雷达点云下，可视化2D bbox和3D bbox;
# ②将图片和text输入groundingDINO模型，输出2D Bounding box，同时输出真实标签和预测标签的可视化结果
# ③将多视角图像输入streamPERT模型，将预测的所有3D bbox和真实bbox计算3D IOU，将最大的IOU保留作为3DIOUresult，同时输出真是标签和预测标签的可视化结果

import pickle
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import time
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import os
import numpy as np


nusc = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True) #因为几乎每个函数都要用到，所以这里定义全局变量
train_data_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_train_data.pkl'
with open('/data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_train_data.pkl', 'rb') as file:
    train_data = pickle.load(file) #因为几乎每个函数都要用到，所以这里把train_data也定义为定义全局变量

output_dir = '/data_volume_1/sjk_data/NuscenesGrounding/'

#显示带有2D Bounding box的图像
def plot_2d_bbox(data):
    image_bbox_output_path = os.path.join(output_dir, f"image_bbox_{data['token']}.png")
    cam_path = os.path.join(nusc.dataroot, data['cam_file'])
    image = cv2.imread(cam_path)
    x_min, y_min, x_max, y_max = data['2d_bbox']
    caption = data['nlp_desc']
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    # Split the caption into multiple lines if it's too long
    max_width = 40
    lines = [caption[i:i+max_width] for i in range(0, len(caption), max_width)]
    y_offset = y_min - 10
    for line in lines:
        cv2.putText(image, line, (x_min, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset -= 20
    cv2.imwrite(image_bbox_output_path, image)


def draw_3d_bbox(center, size, color=[1, 0, 0]):
    w, l, h = size
    cx, cy, cz = center
    # 8 corners of the box
    dx = w / 2
    dy = l / 2
    dz = h / 2
    corners = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx + dx, cy - dy, cz + dz]
    ])

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def add_text_marker(center, radius=0.2):
    # 使用小球来表示文本位置
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color([0, 1, 0])  # 设置球体为绿色
    return sphere


def show_pointcloud_with_bbox(data):
    # 1. 读取点云
    lidar_path = os.path.join(nusc.dataroot, data['lidar_file'])
    # points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    # [x, y, z, intensity, ring index]
    pc = np.frombuffer(open(lidar_path, "rb").read(), dtype=np.float32)
    pc = pc.reshape(-1, 5)[:, :4]

    x, y, z, intensity = pc.T

    # 设置图像的尺寸1024x1024
    image_size = 500

    # 数据归一化
    # 点的坐标范围大概是100
    pc_range = 100
    x = x / pc_range  # [-1,1]
    y = y / pc_range

    # 缩放到图像大小，并平移到图像中心
    half_image_size = image_size / 2
    x = x * half_image_size + half_image_size
    y = y * half_image_size + half_image_size

    # opencv的图像，可以用numpy进行创建
    image = np.zeros((image_size, image_size, 3), np.uint8)

    for ix, iy, iz in zip(x, y, z):
        ix = int(ix)
        iy = int(iy)

        # 判断是否在图像范围内
        if ix >= 0 and ix < image_size and iy >= 0 and iy < image_size:
            image[iy, ix] = 255, 255, 255

    cv2.imwrite("pointcloud.jpg", image)
    cv2.imshow("image", image)
    cv2.waitKey(0)



def pointcloud_to_image(data, image_size=(28600, 15400), focal_length=500):
    pointcloud_bbox_output_path = os.path.join(output_dir, f"pointcloud_bbox_{data['token']}.png")
    lidar_path = os.path.join(nusc.dataroot, data['lidar_file'])
    # 1. 加载点云
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]  # 只取 X, Y, Z
    points = points[np.abs(points[:, 2]) < 30]  # 去掉Z轴超过30米的点

    # 2. 投影到2D平面
    # 假设点云坐标系的Z轴是与相机平行的，可以使用简单的透视投影
    # 假设摄像头坐标系为：Z轴垂直于平面，X轴和Y轴平行。

    # 透视投影
    image_points = []
    for point in points:
        # 假设相机的原点是(0, 0, 0)，相机的焦距为focal_length
        if point[2] == 0:  # 避免除以零
            continue
        x_proj = int(focal_length * point[0] / point[2])
        y_proj = int(focal_length * point[1] / point[2])

        # 将投影点限制在图像范围内
        if 0 <= x_proj < image_size[0] and 0 <= y_proj < image_size[1]:
            image_points.append([x_proj, y_proj])

    # 3. 检查 image_points 是否为空
    if not image_points:
        print("Warning: No valid points projected onto the image.")
        return

    # 4. 创建一个空白图像，并将投影的点云显示为白色小点
    img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    for point in image_points:
        img[point[1], point[0]] = 255  # 将投影点在图像上显示为白色像素

    # 5. 画 3D Bounding Box 投影
    # 假设你有 3D Box 的参数: 中心点 (cx, cy, cz), 长宽高 (w, l, h)
    # 用于可视化 2D Bounding Box
    cx, cy, cz = [10.0, 5.0, 1.5]  # 示例中心
    w, l, h = [4.0, 2.0, 1.5]  # 示例长宽高

    # 投影 3D bounding box 到 2D
    corners = [
        [cx - w / 2, cy - l / 2, cz - h / 2], [cx + w / 2, cy - l / 2, cz - h / 2],
        [cx + w / 2, cy + l / 2, cz - h / 2], [cx - w / 2, cy + l / 2, cz - h / 2],
        [cx - w / 2, cy - l / 2, cz + h / 2], [cx + w / 2, cy - l / 2, cz + h / 2],
        [cx + w / 2, cy + l / 2, cz + h / 2], [cx - w / 2, cy + l / 2, cz + h / 2]
    ]

    # 投影所有角点到 2D
    bbox_2d = []
    for corner in corners:
        if corner[2] == 0:  # 避免除以零
            continue
        x_proj = int(focal_length * corner[0] / corner[2])
        y_proj = int(focal_length * corner[1] / corner[2])
        if 0 <= x_proj < image_size[0] and 0 <= y_proj < image_size[1]:
            bbox_2d.append([x_proj, y_proj])

    # 6. 检查 2D Bounding Box 投影是否有效
    if len(bbox_2d) < 4:
        print("Warning: 3D Bounding Box did not project correctly into 2D space.")
    else:
        # 绘制 Bounding Box
        bbox_2d = np.array(bbox_2d)
        for i in range(4):
            cv2.line(img, tuple(bbox_2d[i]), tuple(bbox_2d[(i + 1) % 4]), 255, 2)
            cv2.line(img, tuple(bbox_2d[i + 4]), tuple(bbox_2d[((i + 1) % 4) + 4]), 255, 2)
            cv2.line(img, tuple(bbox_2d[i]), tuple(bbox_2d[i + 4]), 255, 2)

    # 7. 保存为 PNG
    cv2.imwrite(pointcloud_bbox_output_path, img)
    print(f'存储路径为{pointcloud_bbox_output_path}')

def get_image_and_point_cloud_by_sample_token(sample_token, data):
    sample = nusc.get('sample', sample_token) #sample为采样的快照，快照下包含 6 个相机图像、1 个 LiDAR 点云、5 个雷达数据
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'] #六张图像
    sensor_tokens = sample['data'] #获取六张图像 + 点云的 sample_data_token
    plot_2d_bbox(data)
    pointcloud_to_image(data)
    show_pointcloud_with_bbox(data)
    print('test')



if __name__ == '__main__':
    for data in train_data:
        sample_token = data['sample_token'] #获取Bounding box的token，进而获得该帧的图像和点云信息
        get_image_and_point_cloud_by_sample_token(sample_token, data)
        break
    print(1)

