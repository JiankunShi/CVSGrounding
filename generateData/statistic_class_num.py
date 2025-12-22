#下面是统计数据集的类别和每个类别的数目，结果保存到PNG图片
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# 1. 加载数据
with open('/data_volume_1/sjk_data/NuscenesGrounding/all_val_data_map_caption_IOU03_yaw.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. 提取类别信息
categories = []
for sample in data:
    try:
        category_str = sample['attribute_caption']['category']
        category = category_str.split('.')[-1]
        categories.append(category)
    except KeyError:
        print("缺少字段，跳过一个样本")

# 3. 统计类别数量
category_counts = Counter(categories)

# 4. 按类别排序（可选）
sorted_items = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*sorted_items)

# 5. 绘制柱状图
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values)

# 6. 添加数字标注
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, str(height), ha='center', va='bottom', fontsize=10)

# 7. 美化图表
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 8. 显示图表
output_path = '/data_volume_1/sjk_data/NuscenesGrounding/Statistics_Categories_Info.png'
plt.savefig(output_path, dpi=300)
print(f"图表已保存至：{output_path}")
plt.show()