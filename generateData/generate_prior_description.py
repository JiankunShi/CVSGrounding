#既然能输入文本描述，所以希望通过文本描述提供先验信息，进而增强模型预测性能，例如车子边界框最大和最小的长宽高分别是多少，自车的可视范围是多少。
from pyquaternion import Quaternion
import math
import pickle

train_xdis_list = []
train_ydis_list = []
val_xdis_list = []
val_ydis_list = []

def getPriorSizeAndDistance(info):
    category = ''
    max_size_x = 0
    max_size_y = 0
    max_size_z = 0
    min_size_x = 0
    min_size_y = 0
    min_size_z = 0
    max_distance_x = 0
    max_distance_y = 0
    max_distance_z = 0
    min_distance_x = 0
    min_distance_y = 0
    min_distance_z = 0


train_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw.pkl'
train_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre.pkl'
val_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw.pkl'
val_pkl_output_path = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre.pkl'

#下面是训练集
with open(train_pkl_path, 'rb') as f:
    train_data = pickle.load(f)

all_train_samples = train_data['data_list']

for sample in all_train_samples:
    sample['centre_2d_relative_ego'] = compute_centre_from_info(sample)
    train_xdis_list.append(sample['centre_2d_relative_ego'][0])
    train_ydis_list.append(sample['centre_2d_relative_ego'][1])

train_data['data_list'] = all_train_samples
with open(train_pkl_output_path, 'wb') as f:
    pickle.dump(train_data, f)


#下面是验证集
with open(val_pkl_path, 'rb') as f:
    val_data = pickle.load(f)

all_val_samples = val_data['data_list']

for sample in all_val_samples:
    sample['centre_2d_relative_ego'] = compute_centre_from_info(sample)
    val_xdis_list.append(sample['centre_2d_relative_ego'][0])
    val_ydis_list.append(sample['centre_2d_relative_ego'][1])

val_data['data_list'] = all_val_samples
with open(val_pkl_output_path, 'wb') as f:
    pickle.dump(val_data, f)


print('finish!')