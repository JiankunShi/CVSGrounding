#这个代码是根据tod3cap训练集，将其字典通过GLM-4-9B-Chat大模型，得到一句完整的自然语言，对物体进行描述，注意，由于环境问题，需要在GLM-4-main的myTest中运行
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import pickle
import time
import torch
#
# #数据准备
nuscenes_pkl_file = "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/all_caption_nuscenes_data.pkl"
with open(nuscenes_pkl_file, 'rb') as f:
    nuscenes_data = pickle.load(f)

#模型准备
device = "cuda:1"
tokenizer = AutoTokenizer.from_pretrained("/data_volumn_3/glm-4-9b-chat/", trust_remote_code=True, device_map = 'auto')#
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "/data_volumn_3/glm-4-9b-chat/",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # device_map = 'auto'
).to(device).eval()

# 定义生成文本时的参数
gen_kwargs = {
    "max_length": 2500,  # 设置生成文本的最大长度
    "do_sample": True,  # 是否从可能的下一个词中随机选择
    "top_k": 1  # 从概率最高的k个词中选择
}

# key_list = ['attribute_caption', 'depth_caption', 'localization_caption', 'motion_caption', 'map_caption', 'relation_caption']
sample_token_data = []

#如果重新运行，需要把已经有描述的物体添加进去
# open pkl,
# sample_token_data.append()
st_index = 737600
nuscenes_data = nuscenes_data[st_index:]

for index, data in tqdm.tqdm(enumerate(nuscenes_data)):
    query_english = 'Please generate a concise English description based on the following dictionary.' \
                    'Please include all the information in the dictionary. Please do not add additional descriptions outside the dictionary. Notice that the subject is the ' + data['attribute_caption']['attribute_caption']+':'
    # if data['attribute_caption']['attribute_caption'] != 'none':
    #     query_english += ' attribute_caption:' + data['attribute_caption']['attribute_caption']
    if data['attribute_caption']['category'] != 'none':
        query_english += ', category:' + data['attribute_caption']['category']
    if data['depth_caption']['depth_caption'] != 'none':
        query_english += ', depth_caption:' + data['depth_caption']['depth_caption']
    if data['localization_caption']['localization_caption'] != 'none':
        query_english += ', localization_caption:' + data['localization_caption']['localization_caption']
    if data['motion_caption']['motion_caption'] != 'none':
        query_english += ', motion_caption:' + data['motion_caption']['motion_caption']
    if data['map_caption']['map_caption'] != 'none':
        query_english += ', map_caption:' + data['map_caption']['map_caption']
    if data['relation_caption'] != 'none':
        query_english += ', relation_caption' + data['relation_caption']

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": query_english}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    # 使用torch.no_grad()上下文管理器来禁用梯度计算，这在推理时可以减少内存使用
    with torch.no_grad():
        # 使用模型的generate方法生成文本
        outputs = model.generate(**inputs, **gen_kwargs)

        # 截取生成的文本，去除开头的提示部分
        outputs = outputs[:, inputs['input_ids'].shape[1]:]

        prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 使用分词器的decode方法将生成的词ID解码回文本，并打印出来
        data['nlp_desc'] = prompt
        sample_token_data.append(data)

        print(prompt)

    if index % 2000 == 0:
        with open('/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data_st_' + str(st_index) + '.pkl', 'wb') as f:
            pickle.dump(sample_token_data, f)

        print('当前运行到'+str(index)+'个，花费总时间为{}'.format(time.time()-start_time))


#下面是看checkpoint种有多少数据

# nuscenes_pkl_file = "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/GLM4_9B_process_nuscenes_data.pkl"
# with open(nuscenes_pkl_file, 'rb') as f:
#     nuscenes_data = pickle.load(f)
# print(test)

# The adult human pedestrian is located in the front right of the ego car, moving slowly and is seen farther than the eye can see. They are positioned to the left of a black, shiny, and sleek bicycle.
# 236141it [119:58:00,  1.83s/it]