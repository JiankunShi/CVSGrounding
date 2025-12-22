#这个代码是在SPNuscnens论文第三章数据统计的完整代码，我们统计了场景数，帧数，各个维度文本平均长度和自然语言文本平均长度（柱状图），自然语言文本词云（词云图），10个类别百分比（饼图）
#先把训练集和测试集拼接起来，得到总数据集，然后存成mmdet_all_data_map_caption_IOU03_yaw_egoCentre.pkl
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#统计场景数，输入samples，统计里面的场景数
def statistic_scene_frame_by_print(samples):
    # 初始化场景和帧的统计
    scene_count = 0
    frame_count = 0
    scene_id_dict = {}
    frame_id_dict = {}

    # 遍历所有样本，统计场景和帧
    for sample in samples:
        # 获取场景ID和帧ID
        scene_id = sample['scene_id']  # 假设每个样本中有一个 'scene_token' 来标识场景
        frame_id = scene_id + f'_frame_id_{sample["frame_id"]}'

        if scene_id not in scene_id_dict:
            scene_id_dict[scene_id] = 1
            scene_count += 1
        else:
            scene_id_dict[scene_id] += 1
        if frame_id not in frame_id_dict:
            frame_id_dict[frame_id] = 1
            frame_count += 1
        else:
            frame_id_dict[frame_id] += 1

    # 计算每个场景的平均帧数
    average_frames_per_scene = frame_count / scene_count if scene_count > 0 else 0

    # 输出统计结果
    print(f"场景数: {scene_count}")
    print(f"帧数: {frame_count}")
    print(f"平均每个场景的帧数: {average_frames_per_scene:.2f}")

def statistic_category10per_by_pie(samples):
    """
    根据samples，统计10个类别百分比，并用饼状图表示
    :return:
    """
    categories_list = ['Car', 'Truck', 'Trailer', 'Pedestrian', 'Bus', 'Barrier', 'Motorcycle', 'Bicycle', 'Trafficcone', 'Vehicle.Construction']
    categories_dict = {}
    for sample in samples:
        category_name = sample['attribute_caption']['category']
        for category_name_ori in categories_list:
            if category_name_ori.lower() in category_name:
                if category_name_ori in categories_dict:
                    categories_dict[category_name_ori] += 1
                else:
                    categories_dict[category_name_ori] = 1

    labels = list(categories_dict.keys())
    sizes = list(categories_dict.values())
    # print(labels)
    # 颜色
    colors = plt.cm.tab20.colors[:len(labels)]

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制饼状图
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        radius=1.0
    )

    # 中心空心
    centre_circle = plt.Circle((0, 0), 0.45, fc='white')
    fig.gca().add_artist(centre_circle)

    # 保证圆形
    ax.axis('equal')

    # 总数
    total = sum(sizes)

    for wedge, size, label in zip(wedges, sizes, labels):
        # 当前扇形中心角度
        angle = (wedge.theta2 + wedge.theta1) / 2.0

        # 坐标
        x = 0.75 * np.cos(np.deg2rad(angle))
        y = 0.75 * np.sin(np.deg2rad(angle))

        # 标签内容
        percent = size / total * 100
        # print(f'size={size} total={total}')
        # if percent < 3:
        #     continue
        print(f'label:percent = {label}:{percent}')
        # text = f"{label} {percent:.1f}%"
        text = f"{label}"

        # 保证 rotation 在 [-90°, 90°]
        rotation = angle
        while rotation > 180:
            rotation -= 360
        while rotation < -180:
            rotation += 360

        if rotation > 90:
            rotation -= 180
        elif rotation < -90:
            rotation += 180

        ax.text(
            x, y, text,
            ha='center', va='center',
            rotation=rotation,
            rotation_mode='anchor',
            fontsize=20
        )

    # 标题
    # plt.title('Categories Distribution', fontsize=32)

    # 保存
    save_dir = '/data_volume_1/sjk_data/NuscenesGrounding'
    save_filename = 'statistic_category10per_pie.png'
    save_path = os.path.join(save_dir, save_filename)
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    print(f'饼状图已保存到: {save_path}')

def statistic_average_length_eight_dim(samples):
    dim_dict = {"Action": 0,"Category": 0,"Color": 0,"Depth": 0,"Geospatial": 0,"Motion": 0,"Relationship": 0,"Direction": 0,"Holistic_nlp": 0}
    for sample in samples:
        dim_dict['Color'] += len(sample['attribute_caption']['attribute_caption'].split(' '))
        dim_dict['Action'] += len(sample['behavior_phrase'].split(' '))
        dim_dict['Category'] += len(sample['attribute_caption']['category'].split(' '))
        dim_dict['Depth'] += len(sample['depth_caption']['depth_caption'].split(' '))
        dim_dict['Geospatial'] += len(sample['map_caption']['map_caption'].split(' '))
        dim_dict['Motion'] += len(sample['motion_caption']['motion_caption'].split(' '))
        dim_dict['Relationship'] += len(sample['relation_caption'].split(' '))
        dim_dict['Direction'] += len(sample['localization_caption']['localization_caption'].split(' '))
        dim_dict['Holistic_nlp'] += len(sample['nlp_desc'].split(' '))
    # 准备数据
    labels = list(dim_dict.keys())
    values = list(dim_dict.values())
    avg_values = [round(v / len(samples), 2) for v in values]

    # 设置科研论文配色（ColorBrewer Set2，温和且易区分）
    colors = sns.color_palette("Set2", n_colors=len(labels))

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制柱状图
    bars = ax.bar(labels, avg_values, color=colors)

    # 设置坐标轴标签和标题
    ax.set_ylabel('Length', fontsize=15)
    # ax.set_title('Dimension Description Average Length', fontsize=14)

    # 添加数值标注在柱子上
    for bar in bars:
        height = bar.get_height()
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=15)

    # 去掉多余边框，使风格更简洁
    sns.despine()

    # 旋转x轴标签以防重叠
    plt.xticks(ha='center', fontsize=11)
    plt.yticks(ha='right', fontsize=13)
    # plt.xlabel('X轴标签', fontsize=15)  # 设置X轴标签字体大小
    # plt.ylabel('Y轴标签', fontsize=15)  # 设置Y轴标签字体大小

    # 调整布局
    plt.tight_layout()

    # 保存路径
    save_dir = '/data_volume_1/sjk_data/NuscenesGrounding'
    save_filename = 'dim_distribution_bar.png'
    save_path = os.path.join(save_dir, save_filename)
    os.makedirs(save_dir, exist_ok=True)

    # 保存为高分辨率 PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'柱状图已保存到: {save_path}')

# 输入路径
train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior.pkl"
val_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior.pkl"
# 输出路径
all_output_data_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior.pkl"
# 读取train数据集
with open(train_input_path, 'rb') as f:
    train_data = pickle.load(f)
# 读取val数据集
with open(val_input_path, 'rb') as f:
    val_data = pickle.load(f)
# 获取数据列表
train_samples = train_data['data_list']
val_samples = val_data['data_list']
# 合并数据
all_samples = train_samples + val_samples
# 创建合并后的数据字典
all_data = {
    'data_list': all_samples
}
# 保存合并后的数据到指定路径
# with open(all_output_data_path, 'wb') as f:
#     pickle.dump(all_data, f)
# print(f"数据已成功合并并保存到 {all_output_data_path}")
# statistic_scene_frame_by_print(train_samples)
# statistic_scene_frame_by_print(val_samples)
# statistic_scene_frame_by_print(all_samples)
# statistic_category10per_by_pie(all_samples)
statistic_average_length_eight_dim(all_samples)
print('statistic finish!')
