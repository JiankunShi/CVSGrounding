#因为数据比较杂，所以在这里重新统计一下各个数据，包括含义和量
"""
all_caption_nuscenes_data.pkl   该数据是通过StreamPERT的nuscenes2d_temporal_infos_train.pkl和TOD3Cap的final_caption_bbox_token.json根据bounding box token匹配生成的，字典拼接
all_caption_nuscenes_data_val.pkl 该数据是通过StreamPERT的nuscenes2d_temporal_infos_val.pkl和TOD3Cap的final_caption_bbox_token.json根据bounding box token匹配生成的，字典拼接
因为nuscenes数据集的测试集，没有提供标注信息，所以这两个数据集都没有测试集标注
all_train_data.pkl是对all_caption_nuscenes_data.pkl数据集通过大模型，将不同字段的信息，融合成一段描述，因为数据量太多，所以分别生成描述，最后拼接为all_train_data.pkl，在mergeAllData.py文件中合并的
GLM4_9B_process_nuscenes_data_st_0_val.pkl是对all_caption_nuscenes_data.pkl数据集通过大模型，将不同字段的信息，融合成一段描述的，是验证集，也是测试集
后面也包括下面的代码
由于TOD3Cap提供的数据有1165741条，streamPERT提供的训练集有28130帧，包含974146个Bounding box，验证集6019帧，包含192041个Bounding box，加起来一共1166187条，二者相差446条
可视化的时候突然发现很多场景信息为空，这里统计一下个数，最后发现训练集973600个数据中，有233621个没有场景信息，739979有场景信息，验证集190001个样本中，由51079个没有场景信息，138922个有场景信息
"""
import json
import pickle
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

nusc_train = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)
# nusc_test = NuScenes(version='v1.0-test', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)
# scene_id_dict1 = {}
# scene_id_dict2 = {}
# scene_id_dict3 = {}
# scene_id_dict4 = {}
# scene_id_dict_nuscenes2d_temporal_infos_test = {}
# scene_id_dict_nuscenes2d_temporal_infos_train = {}
# scene_id_dict_nuscenes2d_temporal_infos_val = {}
# cnt_scene_id_dict1 = 0
# cnt_scene_id_dict2 = 0
# cnt_scene_id_dict3 = 0
# cnt_scene_id_dict_nuscenes2d_temporal_infos_test = 0
# cnt_scene_id_dict_nuscenes2d_temporal_infos_train = 0
# cnt_scene_id_dict_nuscenes2d_temporal_infos_val = 0
#
# cnt_gt_name_nuscenes2d_temporal_infos_test = 0
# cnt_gt_name_nuscenes2d_temporal_infos_train = 0
# cnt_gt_name_nuscenes2d_temporal_infos_val = 0
# cnt_bounding_box_nuscenes2d_temporal_infos_test = 0
# cnt_bounding_box_nuscenes2d_temporal_infos_train = 0
# cnt_bounding_box_nuscenes2d_temporal_infos_val = 0
#
# with open('/data_volume_1/sjk_data/tod3cap/final_caption_bbox_token.json', 'r', encoding = 'utf-8') as file:
#     data_infos  = json.load(file)
# for data_id in data_infos:
#     data = data_infos[data_id]
#     if data['scene_id'] not in scene_id_dict1:
#         scene_id_dict1[data['scene_id']] = 1
#     else:
#         scene_id_dict1[data['scene_id']] += 1
# print(f'scene_id_dict1 = {sum(scene_id_dict1.values())}')
#
#
# with open('/data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_train_data.pkl', 'rb') as file:
#     data2 = pickle.load(file)
# for data in data2:
#     if data['scene_id'] not in scene_id_dict2:
#         scene_id_dict2[data['scene_id']] = 1
#     else:
#         scene_id_dict2[data['scene_id']] += 1
# print(f'scene_id_dict2 = {sum(scene_id_dict2.values())}')
#
# with open('/data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_caption_nuscenes_data.pkl', 'rb') as file:
#     data3 = pickle.load(file)
# for data in data3:
#     if data['scene_id'] not in scene_id_dict3:
#         scene_id_dict3[data['scene_id']] = 1
#     else:
#         scene_id_dict3[data['scene_id']] += 1
# print(f'scene_id_dict3 = {sum(scene_id_dict3.values())}')
#
# with open('/data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_caption_nuscenes_data_val.pkl', 'rb') as file:
#     data4 = pickle.load(file)
# for data in data4:
#     if data['scene_id'] not in scene_id_dict4:
#         scene_id_dict4[data['scene_id']] = 1
#     else:
#         scene_id_dict4[data['scene_id']] += 1
# print(f'scene_id_dict4 = {sum(scene_id_dict4.values())}')
#
# with open('/data_volume_1/sjk_data/nuscenes_streamPETR/nuscenes2d_temporal_infos_train.pkl', 'rb') as file:
#     nuscenes2d_temporal_infos_train = pickle.load(file)
# for data in nuscenes2d_temporal_infos_train['infos']:
#     scene = nusc_train.get('scene', data['scene_token'])
#     scene_id = scene['name']
#     cnt_gt_name_nuscenes2d_temporal_infos_train += len(data['gt_boxes'])
#     cnt_bounding_box_nuscenes2d_temporal_infos_train += len(data['gt_names'])
#     if scene_id not in scene_id_dict_nuscenes2d_temporal_infos_train:
#         scene_id_dict_nuscenes2d_temporal_infos_train[scene_id] = 1
#     else:
#         scene_id_dict_nuscenes2d_temporal_infos_train[scene_id] += 1
# print(f'scene_id_dict_nuscenes2d_temporal_infos_train = {sum(scene_id_dict_nuscenes2d_temporal_infos_train.values())}')
#
# with open('/data_volume_1/sjk_data/nuscenes_streamPETR/nuscenes2d_temporal_infos_val.pkl', 'rb') as file:
#     nuscenes2d_temporal_infos_val = pickle.load(file)
# for data in nuscenes2d_temporal_infos_val['infos']:
#     scene = nusc_train.get('scene', data['scene_token'])
#     scene_id = scene['name']
#     cnt_gt_name_nuscenes2d_temporal_infos_val += len(data['gt_boxes'])
#     cnt_bounding_box_nuscenes2d_temporal_infos_val += len(data['gt_names'])
#     if scene_id not in scene_id_dict_nuscenes2d_temporal_infos_val:
#         scene_id_dict_nuscenes2d_temporal_infos_val[scene_id] = 1
#     else:
#         scene_id_dict_nuscenes2d_temporal_infos_val[scene_id] += 1
# print(f'scene_id_dict_nuscenes2d_temporal_infos_val = {sum(scene_id_dict_nuscenes2d_temporal_infos_val.values())}')
#
# with open('/data_volume_1/sjk_data/nuscenes_streamPETR/nuscenes2d_temporal_infos_test.pkl', 'rb') as file:
#     nuscenes2d_temporal_infos_test = pickle.load(file)
# for data in nuscenes2d_temporal_infos_test['infos']:
#     scene = nusc_test.get('scene', data['scene_token'])
#     scene_id = scene['name']
#     sample = nusc_test.get('sample', data['token'])
#     cnt_gt_name_nuscenes2d_temporal_infos_test += len(data['gt_boxes'])
#     cnt_bounding_box_nuscenes2d_temporal_infos_test += len(data['gt_names'])
#     if scene_id not in scene_id_dict_nuscenes2d_temporal_infos_test:
#         scene_id_dict_nuscenes2d_temporal_infos_test[scene_id] = 1
#     else:
#         scene_id_dict_nuscenes2d_temporal_infos_test[scene_id] += 1
# print(f'scene_id_dict_nuscenes2d_temporal_infos_test = {sum(scene_id_dict_nuscenes2d_temporal_infos_test.values())}')
#
# print(data)
#
#
#
# #由于TOD3Cap提供的数据有1165741条，streamPERT提供的训练集有28130帧，包含974146个Bounding box，验证集6019帧，包含192041个Bounding box，加起来一共1166187条，二者相差446条
# #下面将Bounding box相同的，合成一个数据集
# cnt_match_train = 0
# cnt_match_val = 0
# for bbox_id_tod3cap in tqdm(data_infos):
#     for sample in nuscenes2d_temporal_infos_train['infos']:
#         sample_ = nusc_train.get('sample', sample['token'])


#可视化的时候突然发现很多场景信息为空，这里统计一下个数，最后发现训练集973600个数据中，有233621个没有场景信息，验证集190001个样本中，由51079个没有场景信息
map_empty_num_train = 0
map_empty_num_val = 0

# Step 1: 读取 pkl 文件
# with open('/data_volume_1/sjk_data/NuscenesGrounding/all_train_data.pkl', 'rb') as file:
#     data_list = pickle.load(file)
# # Step 2: 过滤掉 map_caption['map_caption'] 为空的样本
# filtered_data = []
# removed_count = 0  # 统计被移除的数量
# for sample in data_list:
#     if 'map_caption' in sample and 'map_caption' in sample['map_caption']:
#         if len(sample['map_caption']['map_caption']) > 0:
#             filtered_data.append(sample)
#         else:
#             removed_count += 1
#     else:
#         removed_count += 1  # 也统计结构缺失的
#
# print(f"原始样本数量: {len(data_list)}")
# print(f"保留样本数量: {len(filtered_data)}")
# print(f"移除空 map_caption 的样本数量: {removed_count}")
# # Step 3: 可选 - 保存过滤后的数据
# with open('/data_volume_1/sjk_data/NuscenesGrounding/all_train_data_map_caption.pkl', 'wb') as file:
#     pickle.dump(filtered_data, file)


#下面是处理验证集的无场景描述数据

# Step 1: 读取 pkl 文件
# with open('/data_volume_1/sjk_data/NuscenesGrounding/all_val_data.pkl', 'rb') as file:
#     data_list = pickle.load(file)
# # Step 2: 过滤掉 map_caption['map_caption'] 为空的样本
# filtered_data = []
# removed_count = 0  # 统计被移除的数量
# for sample in data_list:
#     if 'map_caption' in sample and 'map_caption' in sample['map_caption']:
#         if len(sample['map_caption']['map_caption']) > 0:
#             filtered_data.append(sample)
#         else:
#             removed_count += 1
#     else:
#         removed_count += 1  # 也统计结构缺失的
# print(f"原始样本数量: {len(data_list)}")
# print(f"保留样本数量: {len(filtered_data)}")
# print(f"移除空 map_caption 的样本数量: {removed_count}")
# # Step 3: 可选 - 保存过滤后的数据
# with open('/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption.pkl', 'wb') as file:
#     pickle.dump(filtered_data, file)
#
# for index, data in enumerate(filtered_data):
#     if index % 100 == 0:
#         print(data)
#




print('finish!!!')