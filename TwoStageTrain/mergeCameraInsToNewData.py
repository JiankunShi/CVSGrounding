import pickle

# 定义文件路径
mini_train_file = '/data_volume_1/sjk_data/NuscenesGrounding/train_one_stage_output.pkl'
other_train_file = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2.pkl'
output_file = '/data_volume_1/sjk_data/NuscenesGrounding/merge_train_one_stage_output.pkl'

def get_dir(data_info):
    res = None
    if data_info['localization_caption']['localization_caption'] == 'in the front of ego car':
        res = 'CAM_FRONT'
    elif data_info['localization_caption']['localization_caption'] == 'in the front right of ego car':
        res = 'CAM_FRONT_RIGHT'
    elif data_info['localization_caption']['localization_caption'] == 'in the front left of ego car':
        res = 'CAM_FRONT_LEFT'
    elif data_info['localization_caption']['localization_caption'] == 'in the back of ego car':
        res = 'CAM_BACK'
    elif data_info['localization_caption']['localization_caption'] == 'in the back left of ego car':
        res = 'CAM_BACK_LEFT'
    elif data_info['localization_caption']['localization_caption'] == 'in the back right of ego car':
        res = 'CAM_BACK_RIGHT'
    return res

# 读取第一个数据集（mini_train_one_stage_output.pkl）
with open(mini_train_file, 'rb') as f:
    mini_train_data = pickle.load(f)

# 读取第二个数据集（mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2.pkl）
with open(other_train_file, 'rb') as f:
    other_train_data = pickle.load(f)
other_train_data = other_train_data['data_list']
# 创建一个字典来存储第二个数据集中的 'lidar2CamIns' 和 'proj_bbox'，以 bbox_token 为键
other_data_dict = {}
for item in other_train_data:
    bbox_token = item['bbox_token']
    lidar2CamIns = item['lidar2CamIns']
    proj_bbox = item['proj_bbox']
    lidar_gt_center_bottom_3d_box = item['lidar_gt_center_bottom_3d_box']
    lidar2CamIns = item['lidar2CamIns'][get_dir(item)]
    category_name = item['attribute_caption']['category']
    other_data_dict[bbox_token] = {'lidar2CamIns': lidar2CamIns, 'proj_bbox': proj_bbox, 'lidar_gt_center_bottom_3d_box':lidar_gt_center_bottom_3d_box,'category_name':category_name}

# 合并数据：遍历第一个数据集，将匹配到的 'lidar2CamIns' 和 'proj_bbox' 添加到其中
merged_results = []
for result in mini_train_data:
    bbox_token = result['bbox_token']

    # 如果在第二个数据集里找到匹配的 bbox_token
    if bbox_token in other_data_dict:
        # 获取匹配的 'lidar2CamIns' 和 'proj_bbox'
        lidar2CamIns = other_data_dict[bbox_token]['lidar2CamIns']
        proj_bbox = other_data_dict[bbox_token]['proj_bbox']
        lidar_gt_center_bottom_3d_box = other_data_dict[bbox_token]['lidar_gt_center_bottom_3d_box']
        category_name = other_data_dict[bbox_token]['category_name']
        # 将这些字段添加到第一个数据集的结果中
        result['lidar2CamIns'] = lidar2CamIns
        result['proj_bbox'] = proj_bbox
        result['lidar_gt_center_bottom_3d_box'] = lidar_gt_center_bottom_3d_box
        result['category_name'] = category_name

        # 将合并后的结果添加到最终列表
        merged_results.append(result)

# 将合并后的结果保存到新的文件
with open(output_file, 'wb') as f:
    pickle.dump(merged_results, f)

print(f"合并后的数据已保存到 {output_file}")
