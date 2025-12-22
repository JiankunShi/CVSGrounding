#因为heatmap一直学不到中心，感觉是分辨率太低了，因为之前统计过甚至有100多米距离的物体，这里统计一下，每10米的距离有多少个样本
import pickle
import os
import matplotlib.pyplot as plt

def StatisticDistanceNum(input_path, output_path):
    # 加载数据
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    all_samples = data['data_list']

    # 提取最大绝对值
    max_abs_values = []
    for sample in all_samples:
        values = sample['lidar_gt_center_bottom_3d_box']
        max_abs = max(abs(values[0]), abs(values[1]))
        max_abs_values.append(max_abs)

    # 划分区间（10为一个区间）
    max_value = max(max_abs_values)
    bins = list(range(0, int(max_value) + 10, 10))

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bins, patches = ax.hist(max_abs_values, bins=bins, edgecolor='black')

    # 添加柱子上方的数字
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, height + 0.5, str(int(count)),
                    ha='center', va='bottom', fontsize=10)

    # 设置标题和标签
    ax.set_title("Distribution of Max Abs lidar_gt_center_bottom_3d_box")
    ax.set_xlabel("Max(abs(x), abs(y)) Range")
    ax.set_ylabel("Number of Samples")

    # 不显示网格
    ax.grid(False)

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"图像已保存至: {output_path}")

if __name__ == "__main__":
    # 路径设置
    train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre.pkl"
    val_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre.pkl"
    train_output_path = "/data_volume_1/sjk_data/NuscenesGrounding/Statistics_distance_mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre.png"
    val_output_path = "/data_volume_1/sjk_data/NuscenesGrounding/Statistics_distance_mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre.png"

    StatisticDistanceNum(train_input_path, train_output_path)
    StatisticDistanceNum(val_input_path, val_output_path)

    train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/train_commands_3d_lidarCentre.pkl"
    test_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/test_commands_3d_lidarCentre.pkl"
    train_output_path = "/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/Statistics_distance_train_commands_3d_lidarCentre.png"
    test_output_path = "/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/Statistics_distance_test_commands_3d_lidarCentre.png"
    StatisticDistanceNum(train_input_path, train_output_path)
    StatisticDistanceNum(test_input_path, test_output_path)