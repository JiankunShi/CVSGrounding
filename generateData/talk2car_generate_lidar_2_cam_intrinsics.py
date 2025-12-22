#因为引入了groundingDINO模型，需要把第一阶段得到的lidar坐标系下的3D Bounding box转换为Cam坐标系下的2D Bounding box，便于与groundingDINO预测的2D Bounding box计算分数，
# 但是转换过程中，需要LiDAR -> ego-> global->ego->camera这个顺序，如果直接lidar->camera，会因为不同传感器拍摄时延，导致出现运动误差，所以最保险的方法是第一种
# 这里是根据token，把转换期间需要的内参给存起来，便于后续计算，生成的新字段名为'lidar2CamIntrinsics'
import pickle, tqdm, numpy as np
from functools import lru_cache
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

# ------------------------------------------------------------------
# 1) 仅创建一次 NuScenes 实例（I/O 最耗时）
# ------------------------------------------------------------------
nusc = NuScenes(
    version='v1.0-trainval',
    dataroot='/data_volume_3/nuscenes/v1_0/',
    verbose=False)          # 设成 False 少打日志

CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

@lru_cache
def get_lidar2CamIns(sample_token: str):
    """
    返回一个 dict，键为相机名，值为:
        T_lidar2cam : 4×4 list (float32)
        intrinsic   : 3×3 list
        img_wh      : [w, h]
        filename    : 相机原图相对路径
    """
    sample   = nusc.get('sample', sample_token)

    # ---- LiDAR 标定 & 车身位姿 ----
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ep = nusc.get('ego_pose',        lidar_sd['ego_pose_token'])

    T_lidar_to_ego    = transform_matrix(lidar_cs['translation'],
                                         Quaternion(lidar_cs['rotation']), inverse=False)
    T_ego_to_global   = transform_matrix(lidar_ep['translation'],
                                         Quaternion(lidar_ep['rotation']), inverse=False)
    T_lidar_to_global = T_ego_to_global @ T_lidar_to_ego

    cam_dict = {}
    for cam_name in CAMS:
        cam_sd = nusc.get('sample_data', sample['data'][cam_name])
        cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_ep = nusc.get('ego_pose',        cam_sd['ego_pose_token'])

        # global → cam
        T_global_to_ego_cam = transform_matrix(cam_ep['translation'],
                                               Quaternion(cam_ep['rotation']), inverse=True)
        T_ego_cam_to_cam    = transform_matrix(cam_cs['translation'],
                                               Quaternion(cam_cs['rotation']), inverse=True)
        T_global_to_cam     = T_ego_cam_to_cam @ T_global_to_ego_cam

        # 直接得到 LiDAR → Cam
        T_lidar_to_cam = T_global_to_cam @ T_lidar_to_global

        cam_dict[cam_name] = {
            'T_lidar2cam': np.asarray(T_lidar_to_cam, dtype=np.float32).tolist(),
            'intrinsic'  : [list(row) for row in cam_cs['camera_intrinsic']],
            'img_wh'     : [cam_sd['width'], cam_sd['height']],
            'filename'   : cam_sd['filename'],
        }

    return cam_dict

# ------------------------------------------------------------------
# 4) 读取原 pkl、写入新字段并保存
# ------------------------------------------------------------------
train_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/train_commands_3d_lidarCentre.pkl'
train_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/train_commands_3d_lidarCentre_lidar2CamIns.pkl'

with open(train_pkl_path, 'rb') as f:
    train_data = pickle.load(f)

all_train_samples = train_data['data_list']

for sample in tqdm.tqdm(all_train_samples, desc='Caching lidar→cam params'):
    # 如果已有可跳过，便于增量运行
    if 'lidar2CamIns' not in sample:
        sample['lidar2CamIns'] = get_lidar2CamIns(sample['token'])

# 覆盖写回
train_data['data_list'] = all_train_samples
with open(train_pkl_output_path, 'wb') as f:
    pickle.dump(train_data, f)

print(f'Done! Saved to: {train_pkl_output_path}')

val_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/test_commands_3d_lidarCentre.pkl'
val_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/test_commands_3d_lidarCentre_lidar2CamIns.pkl'
with open(val_pkl_path, 'rb') as f:
    train_data = pickle.load(f)

all_test_samples = train_data['data_list']

for sample in tqdm.tqdm(all_test_samples, desc='Caching lidar→cam params'):
    # 如果已有可跳过，便于增量运行
    if 'lidar2CamIns' not in sample:
        sample['lidar2CamIns'] = get_lidar2CamIns(sample['token'])

# 覆盖写回
train_data['data_list'] = all_test_samples
with open(val_pkl_output_path, 'wb') as f:
    pickle.dump(train_data, f)

print(f'Done! Saved to: {val_pkl_output_path}')