#该代码是根据nuscenes的时间，生成标签，判断是白天还是黑夜，后来发现识别有误，因为阴天或者阴凉处也会被识别为黑天，所以人工识别，发现751（共850）场景后都是黑夜，
#而且在excel中标出了sunny，rainy，normal（黑夜无雨）天气条件！！！！！路径为/data_volume_1/sjk_data/NuscenesGrounding/nuscenes_scene_labels/scene_day_night_rainy_labels.xlsx
import cv2
import os
import numpy as np
from nuscenes.nuscenes import NuScenes
import json
import pandas as pd
# 加载 NuScenes 数据
# 注意根据你的实际路径修改 dataroot
nusc = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)

output_dir = '/data_volume_1/sjk_data/NuscenesGrounding/nuscenes_scene_labels'
os.makedirs(output_dir, exist_ok=True)

scene_labels = {}

# location -> 时区偏移（单位小时）
location_timezone = {
    'boston-seaport': -5,  # UTC-5
    'singapore-onenorth': 8,  # UTC+8
    'singapore-queenstown': 8
}


def calculate_brightness(image_path):
    """计算图像的亮度"""
    img = cv2.imread(image_path)
    if img is None:
        return 0
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算所有像素的平均亮度
    brightness = np.mean(gray)
    return brightness


for scene in nusc.scene:
    scene_token = scene['token']
    scene_name = scene['name']

    # 获取 first_sample
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    # 获取 CAM_FRONT 图片对应的 sample_data
    cam_front_token = sample['data']['CAM_FRONT']
    cam_front = nusc.get('sample_data', cam_front_token)

    # 获取图像路径
    image_path = os.path.join(nusc.dataroot, cam_front['filename'])

    # 计算图像亮度
    brightness = calculate_brightness(image_path)

    # 设定亮度阈值（你可以根据实际情况调整）
    brightness_threshold = 100  # 亮度阈值，低于此值判定为夜晚

    if brightness > brightness_threshold:
        label = 'daytime'
    else:
        label = 'nighttime'

    scene_labels[scene_name] = label
    print(f"{scene_name}: {label} （brightness={brightness:.2f}）")

#人工检查发现从第751个场景（共850个场景）后，全是夜间，之前全是白天
for num, scene_id in enumerate(scene_labels):
    if num >= 751:
        scene_labels[scene_id]='nighttime'
    else:
        scene_labels[scene_id] = 'daytime'

# 保存为 JSON
json_path = os.path.join(output_dir, 'scene_day_night_labels.json')
with open(json_path, 'w') as f:
    json.dump(scene_labels, f, indent=4)
print(f"标签已保存到 {json_path}")

# 保存为 XLSX
xlsx_path = os.path.join(output_dir, 'scene_day_night_labels.xlsx')
df = pd.DataFrame(list(scene_labels.items()), columns=['scene_name', 'label'])
df.to_excel(xlsx_path, index=False)
print(f"标签也已保存到 {xlsx_path}")
