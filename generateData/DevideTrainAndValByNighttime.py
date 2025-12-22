#这个代码是将已用大语言模型标注好的训练集和测试集，通过新标注的标签，将新的标签daytime,nighttime, rainy,summy,normal(无雨黑夜)插入到字典里，
# 并生成四个数据集，all_val_data_daytime.pkl(185247个数据), all_val_data_nighttime.pkl(4754个数据), all_train_data_daytime.pkl(911821个数据)和all_train_data_nighttime.pkl(61779个数据)
# 因为有些数据没有标注背景信息，所以将有背景信息的挑出来为all_val_data_map_caption.pkl和all_train_data_map_caption.pkl
# 根据有背景信息的数据，标注黑夜和白天，又得到下面四个数据集
#all_val_data_map_caption_daytime.pkl(134887), all_val_data_map_caption_nighttime.pkl(4035), all_train_data_map_caption_daytime.pkl(694120), all_train_data_map_caption_nighttime.pkl(45859)
import json
import pickle
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import pandas as pd
import os

scene_id_dict_train = {}
scene_id_dict_val = {}

with open('/data_volume_1/sjk_data/NuscenesGrounding/all_train_data_map_caption.pkl', 'rb') as file:
    data_train = pickle.load(file)
for data in data_train:
    if data['scene_id'] not in scene_id_dict_train:
        scene_id_dict_train[data['scene_id']] = 1
    else:
        scene_id_dict_train[data['scene_id']] += 1
print(f'scene_id_dict_train = {sum(scene_id_dict_train.values())}')

with open('/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption.pkl', 'rb') as file:
    data_test = pickle.load(file)
for data in data_test:
    if data['scene_id'] not in scene_id_dict_val:
        scene_id_dict_val[data['scene_id']] = 1
    else:
        scene_id_dict_val[data['scene_id']] += 1
print(f'scene_id_dict_val = {sum(scene_id_dict_val.values())}')

# 读取Excel
df = pd.read_excel('/data_volume_1/sjk_data/NuscenesGrounding/nuscenes_scene_labels/scene_day_night_rainy_labels.xlsx', engine='openpyxl')

# 假设第一列是 key，其他列作为 value
# 先取列名
first_col = df.columns[0]
other_cols = df.columns[1:]

# 转成字典
result_dict = df.set_index(first_col)[other_cols].to_dict(orient='index')

# 打印一下看看
print(result_dict)

# 创建保存新数据的列表
train_daytime = []
train_nighttime = []
test_daytime = []
test_nighttime = []

# 处理 data_train
for item in data_train:
    scene_id = item['scene_id']  # 假设每个元素是字典，并且有'scene_id'字段
    if scene_id in result_dict:
        # 添加dayoftime和weather
        item['dayoftime'] = result_dict[scene_id]['dayoftime']
        item['weather'] = result_dict[scene_id]['weather']

        # 根据dayoftime划分
        if item['dayoftime'] == 'daytime':
            train_daytime.append(item)
        elif item['dayoftime'] == 'nighttime':
            train_nighttime.append(item)
    else:
        print(f"警告：{scene_id}在result_dict中找不到")

# 处理 data_test
for item in data_test:
    scene_id = item['scene_id']  # 同样假设有'scene_id'
    if scene_id in result_dict:
        # 添加dayoftime和weather
        item['dayoftime'] = result_dict[scene_id]['dayoftime']
        item['weather'] = result_dict[scene_id]['weather']

        # 根据dayoftime划分
        if item['dayoftime'] == 'daytime':
            test_daytime.append(item)
        elif item['dayoftime'] == 'nighttime':
            test_nighttime.append(item)
    else:
        print(f"警告：{scene_id}在result_dict中找不到")


# 保存为pkl文件的函数
def save_as_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"保存成功: {filename}")


# 创建保存目录
save_dir = '/data_volume_1/sjk_data/NuscenesGrounding/'
os.makedirs(save_dir, exist_ok=True)

# 保存四个pkl文件
save_as_pkl(test_daytime, os.path.join(save_dir, 'all_val_data_map_caption_daytime.pkl'))
save_as_pkl(test_nighttime, os.path.join(save_dir, 'all_val_data_map_caption_nighttime.pkl'))
save_as_pkl(train_daytime, os.path.join(save_dir, 'all_train_data_map_caption_daytime.pkl'))
save_as_pkl(train_nighttime, os.path.join(save_dir, 'all_train_data_map_caption_nighttime.pkl'))



print('finish')