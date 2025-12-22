#目前获得的训练集和验证集分别为36415个和6327个,类别分别为21个和18个，为了减少调优模型的时间，分别按照类别采样，将数据量缩小为原来的1/10，类别数不变，因为类别最少采集1类
import pickle
import random
from collections import defaultdict


def sample_mmdet3d_pkl_by_category(input_path, output_path, sample_ratio=0.2, sample_key='data_list', metainfo_key='metainfo'):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    if sample_key not in data:
        raise KeyError(f"Key '{sample_key}' not found in data dict.")

    all_samples = data[sample_key]

    # 按类别分组
    category_dict = defaultdict(list)
    for sample in all_samples:
        category = sample['attribute_caption']['category']
        category_dict[category].append(sample)

    # 按类别采样
    sampled_samples = []
    for cat, samples in category_dict.items():
        n_samples = max(1, int(len(samples) * sample_ratio))  # 每类至少采1个
        sampled_cat = random.sample(samples, n_samples)
        sampled_samples.extend(sampled_cat)

    # 更新 metainfo 中 classes 为采样后所有类别（去重且排序）
    new_classes = sorted(set([sample['attribute_caption']['category'] for sample in sampled_samples]))

    # 替换数据
    new_data = data.copy()
    new_data[sample_key] = sampled_samples
    new_metainfo = new_data[metainfo_key].copy()
    new_metainfo['classes'] = tuple(new_classes)
    new_data[metainfo_key] = new_metainfo
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)

    print(f"按类别采样完成，原始样本数：{len(all_samples)}，采样后样本数：{len(sampled_samples)}")

sample_mmdet3d_pkl_by_category('/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw.pkl', '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_mini_val_data_map_caption_IOU03_yaw.pkl')
sample_mmdet3d_pkl_by_category('/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw.pkl', '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_mini_train_data_map_caption_IOU03_yaw.pkl')
print('finish!')