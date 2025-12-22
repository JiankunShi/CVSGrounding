#为了与MSSG论文metric一致，所以要在talk2car数据集中添加字段categroy_name，形式与SPNuscenes一致
import pickle, tqdm, numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)


PKL_IN  = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/train_commands_3d_lidarCentre_lidar2CamIns2.pkl'
PKL_OUT = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/train_commands_3d_lidarCentre_lidar2CamIns2_addCategoryName.pkl'

with open(PKL_IN, 'rb') as f:
    data = pickle.load(f)

samples  = data['data_list']

for sample in samples:
    bbox_token = sample['bbox_token']
    box = nusc.get('sample_annotation', bbox_token)
    sample['category_name'] = box['category_name']

with open(PKL_OUT, 'wb') as f:
    pickle.dump(data, f)

print(f'Result saved to: {PKL_OUT}')

PKL_IN  = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/test_commands_3d_lidarCentre_lidar2CamIns2.pkl'
PKL_OUT = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/test_commands_3d_lidarCentre_lidar2CamIns2_addCategoryName.pkl'

with open(PKL_IN, 'rb') as f:
    data = pickle.load(f)

samples  = data['data_list']

for sample in samples:
    bbox_token = sample['bbox_token']
    box = nusc.get('sample_annotation', bbox_token)
    sample['category_name'] = box['category_name']

with open(PKL_OUT, 'wb') as f:
    pickle.dump(data, f)

print(f'Result saved to: {PKL_OUT}')