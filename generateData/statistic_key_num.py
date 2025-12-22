#因为要确定distance和motion描述的阈值，所以这里统计分别有几种distance和motion描述
#因为heatmap一直学不到中心，感觉是分辨率太低了，因为之前统计过甚至有100多米距离的物体，这里统计一下，每10米的距离有多少个样本
import pickle
import os
import matplotlib.pyplot as plt

def StatisticKeyNum(input_path, data_type):
    # 加载数据
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    all_samples = data['data_list']

    depth_dict = {}
    motion_dict = {}
    max_not_moving = 0
    max_slowly_moving = 0
    max_quickly_moving = 0
    for sample in all_samples:
        if sample['motion_caption']['motion_caption'] not in motion_dict:
            motion_dict[sample['motion_caption']['motion_caption']] = 0
        else:
            motion_dict[sample['motion_caption']['motion_caption']] += 1

        # if sample['motion_caption']['motion_caption'] == 'not moving' and sample['motion_caption']['x_vel'] > max_not_moving:
        #     max_not_moving = sample['motion_caption']['x_vel']
        # elif sample['motion_caption']['motion_caption'] == 'not moving' and sample['motion_caption']['y_vel'] > max_not_moving:
        #     max_not_moving = sample['motion_caption']['y_vel']
        # elif sample['motion_caption']['motion_caption'] == 'moving slowly' and sample['motion_caption']['y_vel'] > max_slowly_moving:
        #     max_slowly_moving = sample['motion_caption']['x_vel']
        # elif sample['motion_caption']['motion_caption'] == 'moving slowly' and sample['motion_caption']['y_vel'] > max_slowly_moving:
        #     max_slowly_moving = sample['motion_caption']['y_vel']
        # elif sample['motion_caption']['motion_caption'] == 'moving quickly' and sample['motion_caption']['y_vel'] > max_quickly_moving:
        #     max_quickly_moving = sample['motion_caption']['x_vel']
        # elif sample['motion_caption']['motion_caption'] == 'moving quickly' and sample['motion_caption']['y_vel'] > max_quickly_moving:
        #     max_quickly_moving = sample['motion_caption']['y_vel']

        if sample['motion_caption']['motion_caption'] == 'not moving' and sample['motion_caption']['bev_vel'] > max_not_moving:
            max_not_moving = sample['motion_caption']['bev_vel']
        elif sample['motion_caption']['motion_caption'] == 'moving slowly' and sample['motion_caption']['bev_vel'] > max_slowly_moving:
            max_slowly_moving = sample['motion_caption']['bev_vel']
        elif sample['motion_caption']['motion_caption'] == 'moving quickly' and sample['motion_caption']['bev_vel'] > max_quickly_moving:
            max_quickly_moving = sample['motion_caption']['bev_vel']

        if sample['depth_caption']['depth_caption'] not in depth_dict:
            depth_dict[sample['depth_caption']['depth_caption']] = 0
        else:
            depth_dict[sample['depth_caption']['depth_caption']] += 1
    print(data_type)
    print(f"max_not_moving: {max_not_moving}")
    print(f"max_slowly_moving: {max_slowly_moving}")
    print(f"max_quickly_moving: {max_quickly_moving}")
    print(f"motion_dict: {motion_dict}")
    print(f"depth_dict: {depth_dict}")

if __name__ == "__main__":
    # 路径设置
    train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior2_addDiffDimDesc.pkl"
    val_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior2_addDiffDimDesc.pkl"

    StatisticKeyNum(train_input_path, 'train')
    StatisticKeyNum(val_input_path, 'test')