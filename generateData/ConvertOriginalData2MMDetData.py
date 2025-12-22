#因为直接生成的.pkl数据，不符合mmdet3d框架的输入规范，所以这里重新包装一下
import pickle

# 1. 读取你的原始列表数据
with open('/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption_IOU03_yaw.pkl', 'rb') as f:
    original_list = pickle.load(f)

    cam_views = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    for index, sample in enumerate(original_list):
        original_list[index]['lidar_path'] = original_list[index]['lidar_path'].replace('/data_volumn/nuscenes', '/data_volume_3/nuscenes')
        for cam_view in cam_views:
             original_list[index]['cams'][cam_view]['data_path'] = original_list[index]['cams'][cam_view]['data_path'].replace('/data_volumn/nuscenes', '/data_volume_3/nuscenes')

# 2. 构造包装后的结构
new_format = {
    'data_list': original_list,
    'metainfo': {
        'classes': (
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ),
        'dataset': 'SPNuscenes'
    }
}

# 3. 保存为新的 .pkl 文件供配置文件使用
with open('/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw.pkl', 'wb') as f:
    pickle.dump(new_format, f)


print("转换成功，输出文件为 /data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw.pkl")
