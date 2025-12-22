#Mono3DVG效果太好了，感觉是因为文本提供了深度信息和高度信息，导致定位时非常容易，所以这里在数据集中，提供距离和高度信息
import pickle
import os
import matplotlib.pyplot as plt
import tqdm
import random
from statistic_distance_num import StatisticDistanceNum

def generate_detail_description(input_path):
    # 加载数据
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    all_samples = data['data_list']
    filtered_samples = []  # 用于保存筛选后的样本
    max_abs_values = []

    for sample in all_samples:
        # values = sample['centre_2d_relative_ego'][:2]
        values = sample['lidar_gt_center_bottom_3d_box'][:2]
        if len(values) == 2:
            max_abs = max(abs(values[0]), abs(values[1]))
            max_abs_values.append(max_abs)

            # 取 max(abs(x), abs(y)) < 10 的样本
            if max_abs < distance:
                filtered_samples.append(sample)
        else:
            print('长度不为2')

    # 保存筛选后的数据集
    data['data_list'] = filtered_samples
    with open(filtered_output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"筛选后样本数：{len(filtered_samples)}，保存路径：{filtered_output_path}")


if __name__ == "__main__":
    distance = 6.5
    # 输入路径
    train_input_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5.pkl"
    val_input_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0p5.pkl"

    # 筛选后数据集保存路径
    detail_train_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_detailDesc.pkl"
    detail_val_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0p5_detailDesc.pkl"

    generate_detail_description(train_input_path)
    generate_detail_description(val_input_path)