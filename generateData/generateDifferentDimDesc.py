#由于要测不同角度描述对模型性能的影响，这里要把8个维度，用拼接的方法，分别减去一个，然后得到新的描述，下面是新描述的字段
#except_appearance, except_behavior,except_category, except_depth, except_geograph, except_geographic, except_motion, except_realtionship, except_spatial
import pickle
import os
import numpy as np

def addDifferentDimensionDescription(input_data_path, out_data_path):
    with open(input_data_path, 'rb') as f:
        data_infos = pickle.load(f)
    samples = data_infos['data_list']
    for sample in samples:
        concatenate_desc_all7 = sample['attribute_caption']['category'] + ';' + sample['attribute_caption']['attribute_caption'] + ';' \
                           + sample['depth_caption']['depth_caption'] + ';' + sample['localization_caption']['localization_caption'] + ';' \
                           + sample['motion_caption']['motion_caption'] + ';' + sample['map_caption']['map_caption'] + ';' + \
                           sample['relation_caption']
        concatenate_desc_all7 = concatenate_desc_all7.replace('none','')
        sample['concat_all7'] = concatenate_desc_all7
        sample['concate_all8'] = concatenate_desc_all7 + ';' + sample['behavior_phrase']
        sample['concate_except_category_name'] = concatenate_desc_all7.replace(sample['attribute_caption']['category'], '')
        for cate in sample['attribute_caption']['category'].split('.'):
            sample['concate_except_category_name'] = sample['concate_except_category_name'].replace(cate, 'object')
        sample['concate_except_attribute_caption'] = concatenate_desc_all7.replace(sample['attribute_caption']['attribute_caption'], '')
        sample['concate_except_depth_caption'] = concatenate_desc_all7.replace(sample['depth_caption']['depth_caption'], '')
        sample['concate_except_localization_caption'] = concatenate_desc_all7.replace(sample['localization_caption']['localization_caption'], '')
        sample['concate_except_motion_caption'] = concatenate_desc_all7.replace(sample['motion_caption']['motion_caption'], '')
        sample['concate_except_map_caption'] = concatenate_desc_all7.replace(sample['map_caption']['map_caption'], '')
        sample['concate_except_relation_caption'] = concatenate_desc_all7.replace(sample['relation_caption'], '')
    data_infos['data_list'] = samples
    with open(out_data_path, 'wb') as f:
        pickle.dump(data_infos, f)

train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior.pkl"
val_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior.pkl"

train_output_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior2_addDiffDimDesc.pkl"
val_output_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior2_addDiffDimDesc.pkl"

addDifferentDimensionDescription(train_input_path, train_output_path)
addDifferentDimensionDescription(val_input_path, val_output_path)

print('finish!')