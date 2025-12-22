#这里是根据talk2car数据集，得到talk2car3d数据集
import json
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import os

nusc_train = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)
# nusc_test = NuScenes(version='v1.0-test', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)

def get_target_position_in_lidar_frame(nusc, sample_token, box_token, bbox):
    """
    将目标的中心坐标从 Global 坐标系转换到自车 LIDAR 坐标系（ego frame）。

    参数：
    - nusc: NuScenes 对象
    - sample_token: 样本 token
    - box_token: 目标物体的 sample_annotation token

    返回：
    - np.array: 目标在 LIDAR 坐标系下的位置 [x, y, z]
    """
    # 目标物体的中心在 global 坐标系下
    ann = nusc.get('sample_annotation', box_token)
    target_global = np.array(ann['translation'])

    # 找到对应 sample
    sample = nusc.get('sample', sample_token)

    # 获取 LIDAR_TOP 的 sample_data token
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)

    # 获取 LIDAR 的 ego pose
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # 创建 global → ego (LIDAR) 的变换矩阵
    global2ego = transform_matrix(
        ego_pose['translation'],
        Quaternion(ego_pose['rotation']),
        inverse=True
    )

    # 扩展目标点为齐次坐标
    target_global_homo = np.append(target_global, 1.0)

    # 执行坐标变换
    target_in_lidar = global2ego @ target_global_homo

    global2ego = transform_matrix(
        ego_pose['translation'],
        Quaternion(ego_pose['rotation']),
        inverse=True
    )
    rot_global = Quaternion(bbox['rotation'])
    rot_ego = Quaternion(matrix=global2ego[:3, :3]) * rot_global
    yaw = rot_ego.yaw_pitch_roll[0]

    return target_in_lidar[:3], yaw

def make_transform(rot, trans):
    """把 quaternion & translation 转成 4×4 齐次矩阵."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rot).rotation_matrix
    T[:3, 3]  = trans
    return T

#sample_bbox_token  'cd8cddc5bb964071802db5dca4ca7545'
def process_file(json_path, output_pkl_path,nusc):
    miss_num = 0
    scene_list = []
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_data = []

    for item in tqdm(data['commands']):
        command_text = item['command']
        try:
            bbox = nusc_train.get('sample_annotation', item['box_token'])
            sample_token = bbox['sample_token']
            sample = nusc_train.get('sample', bbox['sample_token'])
            timestamp = sample['timestamp']
            scene = nusc_train.get('scene', sample['scene_token'])
            x_vel, y_vel = nusc.box_velocity(bbox['token'])[:2]
            if scene['name'] not in scene_list:
                scene_list.append(scene['name'])
        except KeyError:
            miss_num += 1
            print(f"[Warning] Sample token not found: {sample_token}")
            continue

        # 从 item 中直接获取 3D Box 数据
        center, yaw = get_target_position_in_lidar_frame(nusc_train, sample_token, bbox['token'], bbox)
        size = bbox['size']
        rotation = bbox['rotation']
        # quaternion = Quaternion(rotation)
        # yaw = quaternion.yaw_pitch_roll[0]
        yaw_sin = float(np.sin(yaw))
        yaw_cos = float(np.cos(yaw))

        # 六个相机 channel 名称按官方顺序列出
        CAM_CHANNELS = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
        ]

        # LIDAR_TOP 作为主参考系（当然你也可以换成别的雷达 / 场景坐标）
        lidar_sd_token = sample['data']['LIDAR_TOP']
        lidar_sd = nusc.get('sample_data', lidar_sd_token)
        lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        lidar_data = nusc.get('sample_data', lidar_sd_token)
        lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

        T_lidar_cs = make_transform(lidar_cs['rotation'], lidar_cs['translation'])  # lidar->ego
        T_ego_lidar = make_transform(lidar_pose['rotation'], lidar_pose['translation'])  # ego -> world

        # ------------ 2. 逐相机提取信息 -----------------
        images = {}

        for cam in CAM_CHANNELS:
            sd_token = sample['data'][cam]
            sd = nusc.get('sample_data', sd_token)
            cam_cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            cam_pose = nusc.get('ego_pose', sd['ego_pose_token'])

            T_cam_cs = make_transform(cam_cs['rotation'], cam_cs['translation'])  # cam -> ego
            T_ego_cam = make_transform(cam_pose['rotation'], cam_pose['translation'])  # ego -> world

            # 公式: LIDAR → Cam = (Cam_cs)^-1 · (Ego_cam)^-1 · Ego_lidar · Lidar_cs
            T_lidar2cam = np.linalg.inv(T_cam_cs) @ np.linalg.inv(T_ego_cam) @ T_ego_lidar @ T_lidar_cs

            images[cam] = dict(
                img_path=os.path.join(nusc.dataroot, sd['filename']),
                cam2img=np.asarray(cam_cs['camera_intrinsic']),  # 3×3
                cam2ego=T_cam_cs,  # 4×4
                lidar2cam=T_lidar2cam,  # 4×4
                sample_data_token=sd_token,
                timestamp=sd['timestamp']
            )

        processed_data.append({
            'token': sample_token,
            'gt_labels_3d':bbox['category_name'].split('.')[-1],
            'nlp_desc': command_text,  # 改字段名
            'timestamp': timestamp,
            'bbox_token': item['box_token'],
            'gt_boxes_3d': {
                'center': list(center),
                'size': size,
                'yaw': yaw,
                'sin_yaw': yaw_sin,
                'cos_yaw': yaw_cos
            },
            'lidar_path': lidar_path,
            'images': images,
            'x_vel':x_vel,
            'y_vel':y_vel
        })

    # 保存为 .pkl
    output_dict = {
        'data_list': processed_data,
        'metainfo': {
            'classes': (
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
            ),
            'dataset': 'SPNuscenes'
        }
    }
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f"[Success] Saved {len(processed_data)} items to {output_pkl_path}")
    print(f'有{miss_num}个token找不到目标')
    print(f'场景数为{len(scene_list)}')
    print(f"commands = {len(data['commands'])}, scene_tokens = {len(data['scene_tokens'])}")

# 分别处理三个文件
process_file('/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/test_commands.json', '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/test_commands_3d.pkl', nusc_train)
process_file('/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/val_commands.json', '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/val_commands_3d.pkl', nusc_train)
process_file('/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/train_commands.json', '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/train_commands_3d.pkl', nusc_train)



print('finish!')