#由于运行时间过长，所以设置数据断点保存，在/data_volume_1/sjk_data/nuscenes_caption_streamPERT/路径下有多个pkl文件，需要将其整合为1个
import pickle

# 定义文件名列表
file_names = [
    "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data.pkl",
    "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data_st_47600.pkl",
    "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data_st_55600_add38000.pkl",
    "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data_st_93600_add644000.pkl",
    "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data_st_737600_add236164.pkl"
]

# 用于存储所有数据的列表
all_data = []

# 读取每个pkl文件并拼接数据
for file_name in file_names:
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        data = data[:-1]
        all_data += data

# 假设数据是可以直接拼接的（如列表或字典），将它们拼接成一个最终数据结构
final_data = all_data  # 根据你的数据类型调整拼接的方式

#这里是根据Stream中训练集和tod3cap共同组成的数据集，然后通过GLM-4得到descriptions，最终拼接得到完整的all_train_data
with open("/data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_train_data.pkl", 'wb') as f:
    pickle.dump(final_data, f)

print("数据已成功拼接并保存为all_train_data.pkl")
