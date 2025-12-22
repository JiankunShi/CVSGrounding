import os
import torch
from torchviz import make_dot
from mmengine.config import Config
from mmdet3d.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS

# 注册自定义
from mmdet3d.models.data_preprocessors.SPNuscenes_data_preprocessor import SPNuscenesDataPreprocessor
MMENGINE_MODELS.register_module(module=SPNuscenesDataPreprocessor)

# 设置环境变量，确保 dot 能找到
os.environ["PATH"] += os.pathsep + "/home/jiankunshi/anaconda3/envs/py39/bin"

# 路径配置
config_path = '/home/jiankunshi/python/Bevfusion_FineTuneByPre_GroundingDINO/configs/conf_mssg_bert_bevfusion_lidar.py'
save_dir = '/home/jiankunshi/python/Bevfusion_FineTuneByPre_GroundingDINO/tools/visualization/visualizationData'
os.makedirs(save_dir, exist_ok=True)

# 模型加载
cfg = Config.fromfile(config_path)
model = MODELS.build(cfg.model).cuda().eval()

# 1. pts_voxel_encoder 可视化
num_voxels = 1000
max_points_per_voxel = 10
num_features = 5
voxels = torch.randn(num_voxels, max_points_per_voxel, num_features).cuda()
num_points = torch.randint(1, max_points_per_voxel + 1, (num_voxels,)).cuda()
z = torch.randint(1, 40, (num_voxels, 1))           # ⚠️ 避免 0 和 40（边缘容易被丢弃）
y = torch.randint(0, 1440, (num_voxels, 1))
x = torch.randint(0, 1440, (num_voxels, 1))
batch_idx = torch.zeros((num_voxels, 1), dtype=torch.int)

coors = torch.cat([batch_idx, z, y, x], dim=1).int().cuda()

voxel_encoder = model.pts_voxel_encoder
out_voxel = voxel_encoder(voxels, num_points, coors)

dot = make_dot(out_voxel, params=dict(list(voxel_encoder.named_parameters())))
dot.render(os.path.join(save_dir, '1_voxel_encoder'), format='png')

# 2. pts_middle_encoder 可视化
# from mmdet3d.utils import scatter
coors = torch.randint(0, 40, (1000, 4)).cuda().int()
middle_encoder = model.pts_middle_encoder
out_middle = middle_encoder(out_voxel, coors, batch_size=1)

dot = make_dot(out_middle, params=dict(list(middle_encoder.named_parameters())))
dot.render(os.path.join(save_dir, '2_middle_encoder'), format='png')

# 3. pts_backbone 可视化
backbone = model.pts_backbone
x = out_middle  # 中间输出是 dense 特征图，直接输入 backbone
out_backbone = backbone(x)

dot = make_dot(out_backbone, params=dict(list(backbone.named_parameters())))
dot.render(os.path.join(save_dir, '3_pts_backbone'), format='png')

# 4. pts_neck 可视化
neck = model.pts_neck
out_neck = neck(out_backbone)

dot = make_dot(out_neck[0], params=dict(list(neck.named_parameters())))
dot.render(os.path.join(save_dir, '4_pts_neck'), format='png')

# 5. fusion_PC_Text 可视化
fusion = model.fusion_PC_Text
lang_feat = torch.randn(1, 768).cuda()
pc_feat = torch.randn(1, 512, 180, 180).cuda()
out_fusion = fusion(lang_feat, pc_feat)

dot = make_dot(out_fusion, params=dict(list(fusion.named_parameters())))
dot.render(os.path.join(save_dir, '5_fusion_PC_Text'), format='png')

# 6. bbox_head 可视化（输入 BEV 特征图 + dummy 的 token embedding）
bbox_head = model.bbox_head
bev_feat = out_neck[0]
lang_feat = torch.randn(1, 768).cuda()  # [CLS] 向量

# 构造假数据
batch_size = 1
input_dict = dict(
    bev_feat=bev_feat,
    text_feat=lang_feat,
    img_metas=[dict(box_type_3d='LiDAR')],
    gt_labels_3d=None,
    gt_bboxes_3d=None,
    gt_scores_3d=None
)

# 6. bbox_head 可视化（用 fusion_PC_Text 输出作为 feats）
bbox_head = model.bbox_head
fusion = model.fusion_PC_Text

# 构造假输入
bev_feat = torch.randn(1, 512, 180, 180).cuda()  # BEV 特征图
lang_feat = torch.randn(1, 768).cuda()           # BERT 输出

# 融合语言+BEV特征，作为 bbox_head 的输入
feats = fusion(lang_feat, bev_feat)

# 构造 batch_input_metas（伪装成真实推理元信息）
batch_input_metas = [{'box_type_3d': 'LiDAR'}]

with torch.no_grad():
    out = bbox_head.predict(feats, batch_input_metas)

# 调试：打印类型
print("类型：", type(out))

if isinstance(out, dict):
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"Visualizing from key: {k}")
            vis_tensor = v
            break
    else:
        raise ValueError("No tensor found in dict output.")
elif isinstance(out, (list, tuple)) and isinstance(out[0], dict):
    for k, v in out[0].items():
        if isinstance(v, torch.Tensor):
            print(f"Visualizing from key: {k}")
            vis_tensor = v
            break
    else:
        raise ValueError("No tensor found in list[dict] output.")
elif isinstance(out, torch.Tensor):
    vis_tensor = out
else:
    raise ValueError(f"Unexpected output type: {type(out)}")

# 可视化
dot = make_dot(vis_tensor, params=dict(list(bbox_head.named_parameters())))
dot.render(os.path.join(save_dir, '6_bbox_head'), format='png')

# ====== 构建整体结构图 ======
print("正在构建完整结构图...")

# ===== 工具函数：构造合法体素输入 =====
def generate_safe_sparse_input(batch_size=1, sparse_shape=(1440, 1440, 41), feature_dim=5, max_points_per_voxel=10):
    z_max = sparse_shape[2] - 2  # 留出顶部
    y_max = sparse_shape[1] - 2
    x_max = sparse_shape[0] - 2

    coors = []
    feats = []
    for z in range(2, z_max, 4):
        for y in range(2, y_max, 40):
            for x in range(2, x_max, 40):
                coors.append([0, z, y, x])  # batch_idx, z, y, x
                feats.append(torch.randn(max_points_per_voxel, feature_dim))

    coors = torch.tensor(coors, dtype=torch.int32).cuda()
    feats = torch.stack(feats).cuda()  # (N, max_points, C)
    num_points = torch.randint(1, max_points_per_voxel + 1, (feats.shape[0],)).cuda()

    return feats, num_points, coors

# ===== 构造前向流程 =====
voxels, num_points, coors = generate_safe_sparse_input()

# Step 1: voxel encoder
feats_voxel = model.pts_voxel_encoder(voxels, num_points, coors)

# Step 2: middle encoder
feats_middle = model.pts_middle_encoder(feats_voxel, coors, batch_size=1)

# Step 3: backbone
feats_backbone = model.pts_backbone(feats_middle)

# Step 4: neck
feats_neck = model.pts_neck(feats_backbone)[0]  # 取第一个输出

# Step 5: language + fusion
lang_feat = torch.randn(1, 768).cuda()
fused_feat = model.fusion_PC_Text(lang_feat, feats_neck)

# Step 6: bbox head predict（不要 no_grad！）
batch_input_metas = [{'box_type_3d': 'LiDAR'}]
bbox_output = model.bbox_head.predict(fused_feat.requires_grad_(), batch_input_metas)

# Step 7: 提取 heatmap 用于可视化
if isinstance(bbox_output, dict) and 'hm' in bbox_output:
    vis_tensor = bbox_output['hm']
    print(f"[总图可视化] 使用字段: 'hm', shape = {vis_tensor.shape}")
else:
    raise ValueError("bbox_head.predict 返回格式异常，无法生成整图")

# Step 8: 可视化为结构图
dot = make_dot(vis_tensor, params=dict(model.named_parameters()))
dot.render(os.path.join(save_dir, 'full_model_graph'), format='png')

print("✅ 成功生成结构图：full_model_graph.png")
