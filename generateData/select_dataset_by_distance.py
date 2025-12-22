#因为heatmap一直学不到中心，感觉是分辨率太低了，所以这里挑选数据集
import pickle
import os
import matplotlib.pyplot as plt
import tqdm
import random
from statistic_distance_num import StatisticDistanceNum

def SelectDistanceNum(input_path, filtered_output_path, distance):
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

#下面这个函数是按概率选择范围内的物体
def selectObjectByDistanceAndProbability(input_path, filtered_output_path, distance, probability, statistic_path):
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
            if max_abs < distance and random.random()<=probability:
                filtered_samples.append(sample)
        else:
            print('长度不为2')

    # 保存筛选后的数据集
    data['data_list'] = filtered_samples
    with open(filtered_output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"筛选后样本数：{len(filtered_samples)}，保存路径：{filtered_output_path}")

    #统计筛选后数据按距离分布情况
    StatisticDistanceNum(filtered_output_path, statistic_path)

if __name__ == "__main__":
    distance = 6.5
    # 输入路径
    train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView.pkl"
    val_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView.pkl"

    # 筛选后数据集保存路径
    filtered_train_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView6p5.pkl"
    filtered_val_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView6p5.pkl"

    SelectDistanceNum(train_input_path, filtered_train_path, distance)
    SelectDistanceNum(val_input_path, filtered_val_path, distance)

    distance = 30
    probability = 0.5
    # 输入路径
    train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView.pkl"
    val_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView.pkl"

    # 筛选后数据集保存路径
    filtered_train_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5.pkl"
    filtered_val_path = f"/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0p5.pkl"
    statistic_train_path = f"/data_volume_1/sjk_data/NuscenesGrounding/statistic_mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5.png"
    statistic_val_path = f"/data_volume_1/sjk_data/NuscenesGrounding/statistic_mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0p5.png"

    selectObjectByDistanceAndProbability(train_input_path, filtered_train_path, distance, probability, statistic_train_path)
    selectObjectByDistanceAndProbability(val_input_path, filtered_val_path, distance, probability, statistic_val_path)



#保存路径：/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre6p5.pkl   筛选后样本数：14582
#保存路径：/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre6p5.pkl   筛选后样本数：2304