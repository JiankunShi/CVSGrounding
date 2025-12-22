#这个文件是生成数据集，将多种描述拼接，而非生成自然语言
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import pickle
import time
import torch
start_time = time.time()
# #数据准备
#nuscenes_pkl_file = "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_caption_nuscenes_data_val.pkl"
nuscenes_pkl_file = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0p5.pkl"

with open(nuscenes_pkl_file, 'rb') as f:
    nuscenes_data_ori = pickle.load(f)

st_index = 0
nuscenes_data = nuscenes_data_ori['data_list'][st_index:]

for index, data in tqdm.tqdm(enumerate(nuscenes_data)):
    concatenate_desc = data['attribute_caption']['attribute_caption']+';'+data['depth_caption']['depth_caption']+';'+data['localization_caption']['localization_caption']+';'+data['motion_caption']['motion_caption']+';'+data['map_caption']['map_caption']+';'+data['relation_caption']
    camera_rec_desc =  'The '+ data['attribute_caption']['attribute_caption'] + ' is '+ data['map_caption']['map_caption']
    concate_except_env_desc = data['attribute_caption']['category']+';'+data['depth_caption']['depth_caption']+';'+data['localization_caption']['localization_caption']+';'+data['motion_caption']['motion_caption']+';'+data['relation_caption']
    nuscenes_data[index]['concatenate_desc'] = concatenate_desc
    nuscenes_data[index]['camera_rec_desc'] = camera_rec_desc
    nuscenes_data[index]['concate_except_env_desc'] = camera_rec_desc

    # print(camera_rec_desc)
nuscenes_data_ori['data_list'] = nuscenes_data
with open('/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc.pkl', 'wb') as f:
    pickle.dump(nuscenes_data_ori, f)

print('花费总时间为{}'.format(time.time()-start_time))


nuscenes_pkl_file = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5.pkl"

with open(nuscenes_pkl_file, 'rb') as f:
    nuscenes_data_ori = pickle.load(f)

st_index = 0
nuscenes_data = nuscenes_data_ori['data_list'][st_index:]

for index, data in tqdm.tqdm(enumerate(nuscenes_data)):
    concatenate_desc = data['attribute_caption']['attribute_caption']+';'+data['depth_caption']['depth_caption']+';'+data['localization_caption']['localization_caption']+';'+data['motion_caption']['motion_caption']+';'+data['map_caption']['map_caption']+';'+data['relation_caption']
    camera_rec_desc =  'The '+ data['attribute_caption']['attribute_caption'] + ' is '+ data['map_caption']['map_caption']
    concate_except_env_desc = data['attribute_caption']['category']+';'+data['depth_caption']['depth_caption']+';'+data['localization_caption']['localization_caption']+';'+data['motion_caption']['motion_caption']+';'+data['relation_caption']
    nuscenes_data[index]['concatenate_desc'] = concatenate_desc
    nuscenes_data[index]['camera_rec_desc'] = camera_rec_desc
    nuscenes_data[index]['concate_except_env_desc'] = camera_rec_desc

    # print(camera_rec_desc)
nuscenes_data_ori['data_list'] = nuscenes_data
with open('/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc.pkl', 'wb') as f:
    pickle.dump(nuscenes_data_ori, f)

print('花费总时间为{}'.format(time.time()-start_time))

#下面是看checkpoint种有多少数据

# nuscenes_pkl_file = "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data.pkl"
# with open(nuscenes_pkl_file, 'rb') as f:
#     nuscenes_data = pickle.load(f)
# print(test)

# The adult human pedestrian is located in the front right of the ego car, moving slowly and is seen farther than the eye can see. They are positioned to the left of a black, shiny, and sleek bicycle.
# 236141it [119:58:00,  1.83s/it]