#因为在Grounding时，文本提示通常以自车为中心，所以中心点坐标应该是以自车为中心的全局坐标系，这里是根据物体的全局坐标系和自车的全局坐标系，计算中心坐标
#注意，坐标系问题卡了很久，不同坐标系的translation是不一样的，nuscenes数据集提供的是global坐标系，需要转成ego坐标系，但是bevfusion预训练模型输出的应该是lidar坐标系，所以要global->ego->lidar
#另外，convert_boxes_ego_to_lidar_ori函数是ego->lidar坐标系，但是要注意nuscenes数据集的ego坐标系下偏航角和lidar坐标系下的偏航角应该是一致的，如果转就会偏差很大
#还有，bevfusion预训练模型输出的可能是bottom_center，而不是中心点，所以需要convert_center_to_bottom把中心变为底部
#因为缺包没法运行，所以在Bevfusion_FineTuneByPre_GroundingDINO/generateData/calculateCentre.py运行
from pyquaternion import Quaternion
import math
import pickle
import json
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from mmdet3d.structures import LiDARInstance3DBoxes, Det3DDataSample
import os

nusc_train = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)
# nusc_test = NuScenes(version='v1.0-test', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)

train_xdis_list = []
train_ydis_list = []
val_xdis_list = []
val_ydis_list = []

#转换自车坐标系到lidar坐标系
def convert_boxes_ego_to_lidar_ori(boxes_in_ego: LiDARInstance3DBoxes,
                                ego2lidar: np.ndarray) -> LiDARInstance3DBoxes:
    # 提取框坐标为 numpy（确保在 CPU）
    boxes_tensor = boxes_in_ego.tensor.cpu().numpy()  # shape (N, 7)
    centers_ego = boxes_tensor[:, :3]
    yaws = boxes_tensor[:, 6]

    # 齐次坐标变换
    centers_homo = np.concatenate([centers_ego, np.ones((centers_ego.shape[0], 1))], axis=-1)  # (N, 4)
    centers_lidar = centers_homo @ ego2lidar.T  # ✅ 注意必须是 numpy

    # yaw 角度修正
    # delta_yaw = np.arctan2(ego2lidar[1, 0], ego2lidar[0, 0])
    # yaws_lidar = yaws #+ delta_yaw #去掉delta_yaw就是不变换角度

    # 更新框
    boxes_tensor[:, :3] = centers_lidar[:, :3]
    # boxes_tensor[:, 6] = yaws_lidar

    # 返回新的 LiDARInstance3DBoxes
    boxes_lidar = LiDARInstance3DBoxes(torch.from_numpy(boxes_tensor).to(boxes_in_ego.tensor.device))
    return boxes_lidar

#把中心变为底部
def convert_center_to_bottom(boxes: LiDARInstance3DBoxes) -> LiDARInstance3DBoxes:
    boxes_tensor = boxes.tensor.clone()
    # 假设 z 是中心点，高度是第 5 维（index 2 和 index 5）
    boxes_tensor[:, 2] -= boxes_tensor[:, 5] / 2.0  # z -= height / 2
    return LiDARInstance3DBoxes(boxes_tensor, box_dim=7)

def compute_centre_from_info(info):
    # 物体中心点和自车位置（取 x 和 y）
    obj_x, obj_y = info['3d_center'][:2]
    ego_x, ego_y = info['ego2global_translation'][:2]

    # 计算相对位置
    dx = obj_x - ego_x
    dy = obj_y - ego_y

    # 提取自车 yaw（偏航角，单位是弧度）
    q = info['ego2global_rotation']  # 四元数 [w, x, y, z]
    yaw = Quaternion(q).yaw_pitch_roll[0]  # 提取 yaw

    # 应用逆旋转（从全局坐标转到自车坐标）
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)

    x_local = dx * cos_yaw - dy * sin_yaw
    y_local = dx * sin_yaw + dy * cos_yaw

    return [x_local, y_local]  # 返回 List 类型

def get_target_position_in_lidar_frame(nusc, sample_token, box_token):
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

    return list(target_in_lidar[:3])

#BEVfusion的groundingTruth
def get_single_bbox_from_token_in_ego_frame(nusc, sample_token, bbox_token):
    """
    提取指定 bbox_token 的 3D Bounding Box，并转换为 ego（自车）坐标系下格式。

    参数:
    - nusc: NuScenes 数据库对象
    - sample_token: 当前样本的 token
    - bbox_token: 目标物体的 sample_annotation token

    返回:
    - box_ego: np.array [x, y, z, dx, dy, dz, yaw]
    - label: int 类别 id
    """
    # 获取目标 annotation（包含位置、尺寸、朝向）
    ann = nusc.get('sample_annotation', bbox_token)

    # 中心点在 global 坐标系下
    center_global = np.array(ann['translation'])

    # 获取当前 sample 的 lidar pose（ego pose）
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # global → ego 的变换矩阵
    global2ego = transform_matrix(
        ego_pose['translation'],
        Quaternion(ego_pose['rotation']),
        inverse=True
    )

    # 中心点转换到 ego 坐标
    center_ego = global2ego @ np.append(center_global, 1.0)
    center_ego = center_ego[:3]

    # 尺寸：长、宽、高
    size = np.array(ann['size'])  # [l, w, h]

    # 朝向：global → ego 下的 yaw
    rot_global = Quaternion(ann['rotation'])
    rot_ego = Quaternion(matrix=global2ego[:3, :3]) * rot_global
    yaw = rot_ego.yaw_pitch_roll[0]

    box_ego = np.concatenate([center_ego, size, [yaw]])
    return box_ego

train_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw.pkl'
train_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre.pkl'
val_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw.pkl'
val_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre.pkl'

#下面是训练集
with open(train_pkl_path, 'rb') as f:
    train_data = pickle.load(f)

all_train_samples = train_data['data_list']

for sample in tqdm(all_train_samples):
    # sample['centre_2d_relative_ego'] = compute_centre_from_info(sample)
    # sample['centre_2d_relative_ego'] = get_target_position_in_lidar_frame(nusc_train, sample['sample_token'], sample['bbox_token'])
    bevfusion_box = get_single_bbox_from_token_in_ego_frame(nusc_train, sample['sample_token'], sample['bbox_token'])
    sample['centre_2d_relative_ego'] = list(bevfusion_box[:3])
    sample['yaw'] = bevfusion_box[-1]
    sample['sin_yaw'] = np.sin(bevfusion_box[-1])
    sample['cos_yaw'] = np.cos(bevfusion_box[-1])
    #将ego坐标系的translation转为lidar坐标系的translation，注意，不转yaw
    gt_instances_3d = InstanceData()
    bboxes_tensor = torch.tensor(
        sample['centre_2d_relative_ego'] + sample['3d_size'] + [sample['yaw']]
        , dtype=torch.float32).view([-1, 7])
    gt_instances_3d.bboxes_3d = LiDARInstance3DBoxes(bboxes_tensor)
    # ego 3dbbox转为 lidar 3dbbox
    lidar2ego = transform_matrix(sample['lidar2ego_translation'], Quaternion(sample['lidar2ego_rotation']), inverse=False)
    ego2lidar = np.linalg.inv(lidar2ego)
    print(gt_instances_3d.bboxes_3d)
    gt_instances_3d.bboxes_3d = convert_boxes_ego_to_lidar_ori(gt_instances_3d.bboxes_3d, ego2lidar)
    sample['translation_in_lidar_center'] = gt_instances_3d.bboxes_3d[:3]
    print(gt_instances_3d.bboxes_3d)
    gt_instances_3d.bboxes_3d = convert_center_to_bottom(gt_instances_3d.bboxes_3d)
    sample['translation_in_lidar_bottom'] = gt_instances_3d.bboxes_3d[:3]
    print(gt_instances_3d.bboxes_3d)
    print('-'*60)
    train_xdis_list.append(sample['centre_2d_relative_ego'][0])
    train_ydis_list.append(sample['centre_2d_relative_ego'][1])

train_data['data_list'] = all_train_samples
with open(train_pkl_output_path, 'wb') as f:
    pickle.dump(train_data, f)


#下面是验证集
with open(val_pkl_path, 'rb') as f:
    val_data = pickle.load(f)

all_val_samples = val_data['data_list']

for sample in tqdm(all_val_samples):
    # sample['centre_2d_relative_ego'] = compute_centre_from_info(sample)
    # sample['centre_2d_relative_ego'] = get_target_position_in_lidar_frame(nusc_train, sample['sample_token'], sample['bbox_token'])
    bevfusion_box = get_single_bbox_from_token_in_ego_frame(nusc_train, sample['sample_token'], sample['bbox_token'])
    sample['centre_2d_relative_ego'] = list(bevfusion_box[:3])
    sample['yaw'] = bevfusion_box[-1]
    sample['sin_yaw'] = np.sin(bevfusion_box[-1])
    sample['cos_yaw'] = np.cos(bevfusion_box[-1])
    val_xdis_list.append(sample['centre_2d_relative_ego'][0])
    val_ydis_list.append(sample['centre_2d_relative_ego'][1])

val_data['data_list'] = all_val_samples
with open(val_pkl_output_path, 'wb') as f:
    pickle.dump(val_data, f)

max_train_xdis_list = max(train_xdis_list)
min_train_xdis_list = min(train_xdis_list)
max_train_ydis_list = max(train_ydis_list)
min_train_ydis_list = min(train_ydis_list)
max_val_xdis_list = max(val_xdis_list)
min_val_xdis_list = min(val_xdis_list)
max_val_ydis_list = max(val_ydis_list)
min_val_ydis_list = min(val_ydis_list)

print("max_train_xdis_list =", max_train_xdis_list)
print("min_train_xdis_list =", min_train_xdis_list)
print("max_train_ydis_list =", max_train_ydis_list)
print("min_train_ydis_list =", min_train_ydis_list)
print("max_val_xdis_list =", max_val_xdis_list)
print("min_val_xdis_list =", min_val_xdis_list)
print("max_val_ydis_list =", max_val_ydis_list)
print("min_val_ydis_list =", min_val_ydis_list)

print('finish!')

#下面是mini数据集的距离
# max_train_xdis_list = 102.65642811760904
# min_train_xdis_list = -52.206583052876965
# max_train_ydis_list = 78.17446564703397
# min_train_ydis_list = -63.22467513770758
# max_val_xdis_list = 74.79444805496303
# min_val_xdis_list = -35.98041912749324
# max_val_ydis_list = 40.01003042478965
# min_val_ydis_list = -46.81902560270157

#下面是all数据集的距离
# max_train_xdis_list = 105.98215052160852
# min_train_xdis_list = -77.86184144569518
# max_train_ydis_list = 80.69394106029014
# min_train_ydis_list = -64.97977787914739
# max_val_xdis_list = 74.79444805496303
# min_val_xdis_list = -91.25729746339178
# max_val_ydis_list = 80.91937926690991
# min_val_ydis_list = -61.51928268998427

#后面可以统计一下，每段距离有多少物体
#all_val_samples[1]['centre_2d_relative_ego'][0]
