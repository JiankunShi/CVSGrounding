#因为MSSG论文，损失函数包括偏航角，偏航角包括全局坐标系下偏航角，雷达坐标系下偏航角和相机坐标系下偏航角，这三个视角可以相互转换
# 因为用的是点云模态，所以这里添加点云坐标系下偏航角到数据集中
import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import os
from tqdm import tqdm

# 路径定义
# input_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_train_data_map_caption_IOU2.pkl'
# output_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_train_data_map_caption_IOU2_yaw.pkl'
input_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption_IOU.pkl'
output_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption_IOU_yaw.pkl'

# 初始化 NuScenes
nusc_train = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)

# 加载原始数据
with open(input_pkl_path, 'rb') as f:
    data_list = pickle.load(f)

# 遍历并添加偏航角字段
for sample in tqdm(data_list, desc="Processing samples"):
    try:
        ann = nusc_train.get('sample_annotation', sample['bbox_token'])

        # 提取 rotation（四元数）
        rotation = ann['rotation']  # [w, x, y, z]
        quat = Quaternion(rotation)

        # 提取 yaw (在 Lidar 坐标系下的偏航角)
        yaw = quat.yaw_pitch_roll[0]  # 返回 (yaw, pitch, roll)，取 yaw

        # 添加新字段
        sample['yaw'] = yaw
        sample['sin_yaw'] = np.sin(yaw)
        sample['cos_yaw'] = np.cos(yaw)

    except Exception as e:
        print(f"Error processing sample with bbox_token={sample.get('bbox_token')}: {e}")
        sample['yaw'] = None
        sample['sin_yaw'] = None
        sample['cos_yaw'] = None

# 保存更新后的数据
with open(output_pkl_path, 'wb') as f:
    pickle.dump(data_list, f)

print(f"处理完成，共处理 {len(data_list)} 条数据，结果保存至：{output_pkl_path}")
