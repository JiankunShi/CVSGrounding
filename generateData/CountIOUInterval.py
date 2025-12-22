#自己的grounding数据集，用GroundingDINO计算2DIOU，然后统计在[0,1]区间中，每个置信度区间的个数，共分了10个区间，最后生成图像
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
font_prop = FontProperties(fname=font_path)
# 设置matplotlib的默认字体
rcParams['font.family'] = font_prop.get_name()
# 设置字体大小，修改标题、坐标轴标签和图例
# rcParams['font.size'] = 16  # 设置全局字体大小
rcParams['axes.titlesize'] = 15  # 设置坐标轴标题字体大小
rcParams['legend.fontsize'] = 15  # 设置图例字体大小

# 加载 .pkl 文件
with open('/data_volume_1/sjk_data/NuscenesGrounding/all_train_data_map_caption_IOU.pkl', 'rb') as f:
    data = pickle.load(f)

# 提取 IOU_2D 值
iou_values = []
for item in data:
    if 'IOU_2D' in item:
        iou_values.append(item['IOU_2D'])
        # iou = item['IOU_2D']
        # if isinstance(iou, (float, int)):  # 确保是——数值型
        #     iou_values.append(iou)

# 定义区间数，例如10个区间：0-0.1, 0.1-0.2, ..., 0.9-1.0
num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)

# 使用 numpy 统计每个区间中的个数
counts, bin_edges = np.histogram(iou_values, bins=bins)

# 绘图
plt.figure(figsize=(8, 5))
bin_labels = [f'{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}' for i in range(len(bin_edges) - 1)]
plt.bar(bin_labels, counts, width=0.6, align='center')

plt.xlabel('IOU_2D 区间')
plt.ylabel('数量')
plt.title('IOU_2D 分布柱状图')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/data_volume_1/sjk_data/NuscenesGrounding/count_all_train_data_map_caption_IOU.png')
