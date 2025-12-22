#这个代码是生成词云图
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 读取数据文件
train_file = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2.pkl'
with open(train_file, 'rb') as f:
    train_data = pickle.load(f)

# 获取所有的文本
text_data = ""
for data in train_data['data_list']:
    text_data += data['nlp_desc'] + " "

# 创建词云
wordcloud = WordCloud(stopwords=None, width=800, height=400, background_color='white').generate(text_data)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# 保存词云图到指定目录
output_path = '/data_volume_1/sjk_data/NuscenesGrounding/SPNuscenes_trainData_wordcloud_image_stopwordsNone.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

# 显示成功提示
print(f"词云图已保存到 {output_path}")

