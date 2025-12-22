#之前只将雷达和文本进行融合，后来想到将多视角图片和雷达进行多模态融合，然后再与文本特征融合，但是原始的数据集只有单图像信息，所以这里添加多视角图像信息
#输入mmdet3d_数据集，得到带有多视角图片字典的新数据集
from pyquaternion import Quaternion
import math
import pickle
from nuscenes.nuscenes import NuScenes
import os
import numpy as np
import tqdm

train_xdis_list = []
train_ydis_list = []
val_xdis_list = []
val_ydis_list = []

nusc_train = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)
def get_camera_dicts_for_sample(sample_token):
    """
    输入：一个 sample 的 token
    输出：该 sample 所有相机视角的 dict 列表，每个 dict 包含 img_path, cam2img, cam2ego, sample_data_token, timestamp, lidar2cam
    """
    sample = nusc_train.get('sample', sample_token)
    camera_dicts = {}

    # 相机视角的sensor names，NuScenes有6个相机
    camera_channels = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    for cam_name in camera_channels:
        sample_data_token = sample['data'][cam_name]
        sample_data = nusc_train.get('sample_data', sample_data_token)

        # 图片路径
        img_path = os.path.join(nusc_train.dataroot, sample_data['filename'])

        # 时间戳
        timestamp = sample_data['timestamp']

        # 相机内参矩阵 cam2img，来自于 sample_data['calibrated_sensor']['camera_intrinsic']
        cam_calibrated_sensor = nusc_train.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        cam2img = np.array(cam_calibrated_sensor['camera_intrinsic'])

        # cam2ego，即相机到ego车身坐标系的变换矩阵4x4
        # 从 calibrated_sensor 中拿到 rotation (四元数) 和 translation (xyz)
        rotation = cam_calibrated_sensor['rotation']  # 四元数 [w, x, y, z]
        translation = cam_calibrated_sensor['translation']  # [x, y, z]

        # 四元数转旋转矩阵
        from pyquaternion import Quaternion
        q = Quaternion(rotation)
        rot_matrix = q.rotation_matrix  # 3x3

        # 构造4x4变换矩阵
        cam2ego = np.eye(4)
        cam2ego[:3, :3] = rot_matrix
        cam2ego[:3, 3] = translation

        # lidar2ego（ego2lidar的逆矩阵）
        # 激光雷达的 calibrated_sensor，名字叫 LIDAR_TOP
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sample_data = nusc_train.get('sample_data', lidar_token)
        lidar_calibrated_sensor = nusc_train.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        lidar_rot = Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix
        lidar_trans = lidar_calibrated_sensor['translation']
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar_rot
        lidar2ego[:3, 3] = lidar_trans

        # 计算 lidar2cam = cam2ego^-1 * lidar2ego
        cam2ego_inv = np.linalg.inv(cam2ego)
        lidar2cam = cam2ego_inv @ lidar2ego

        # 保存字典
        cam_dict = {
            'img_path': img_path,
            'cam2img': cam2img,
            'cam2ego': cam2ego,
            'sample_data_token': sample_data_token,
            'timestamp': timestamp,
            'lidar2cam': lidar2cam
        }
        camera_dicts[cam_name] = cam_dict

    return camera_dicts



train_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre.pkl'
train_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView.pkl'
val_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre.pkl'
val_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView.pkl'

#下面是训练集
with open(train_pkl_path, 'rb') as f:
    train_data = pickle.load(f)

all_train_samples = train_data['data_list']

for sample in tqdm.tqdm(all_train_samples):
    sample['images'] = get_camera_dicts_for_sample(sample['sample_token'])

train_data['data_list'] = all_train_samples
with open(train_pkl_output_path, 'wb') as f:
    pickle.dump(train_data, f)


#下面是验证集
with open(val_pkl_path, 'rb') as f:
    val_data = pickle.load(f)

all_val_samples = val_data['data_list']

for sample in tqdm.tqdm(all_val_samples):
    sample['images'] = get_camera_dicts_for_sample(sample['sample_token'])

val_data['data_list'] = all_val_samples
with open(val_pkl_output_path, 'wb') as f:
    pickle.dump(val_data, f)


print('finish!')