#下面的代码是通过IOU，挑选出能够被groundingDINO识别的数据集
import pickle

# 路径定义
input_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_train_data_map_caption_IOU_yaw.pkl'
output_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_train_data_map_caption_IOU03_yaw.pkl'
input_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption_IOU_yaw.pkl'
output_pkl_path = '/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption_IOU03_yaw.pkl'

# 加载数据
with open(input_pkl_path, 'rb') as f:
    data_list = pickle.load(f)

# 过滤 IOU_2D > 0.3 的样本
filtered_data = [sample for sample in data_list if sample.get('IOU_2D', 0) > 0.3]

# 保存筛选后的数据
with open(output_pkl_path, 'wb') as f:
    pickle.dump(filtered_data, f)

print(f"筛选完成，共筛选出 {len(filtered_data)} 条 IOU_2D > 0.3 的样本，保存至：{output_pkl_path}")
