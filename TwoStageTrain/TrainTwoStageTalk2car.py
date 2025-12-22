import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from typing import Any, List, Sequence
import os
from shapely.geometry import Polygon
import time
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_proj = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)
text_proj2 = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)
def fuse_text_with_queries(
                           query_feat: torch.Tensor,  # [B, N, C]
                           text_feat: torch.Tensor,
                           text_proj,
                           cross_attn):  # [B, C_t] or [B, L, C_t]
    if text_feat.dim() == 2:  # [B, C_t] → [B, 1, C_t]
        text_feat = text_feat.unsqueeze(1)
    kv = text_proj(text_feat)  # [B, L, C]
    fused, _ = cross_attn(query_feat, kv, kv)  # Q=query
    return fused
cross_attn  = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=8,
                batch_first=True)
max_iou_list = []
topk = 128
epochs = 20

def build_lidar_box(raw_boxes):
    """
    raw_box: [x_btm, y_btm, z_btm, length, width, height, yaw]
    转成 nuScenes Box (几何中心、[w,l,h])。
    """
    raw_boxes_list = raw_boxes.detach().clone().tolist()
    result = []
    for raw_box in raw_boxes_list:
        x_btm, y_btm, z_btm = raw_box[:3]
        l_len, w_len, h = raw_box[3:6]
        yaw = raw_box[6]
        center = [x_btm, y_btm, z_btm + h / 2]  # 底面中心 → 几何中心
        size = [w_len, l_len, h]  # [w,l,h]
        result.append(Box(center=center,
                          size=size,
                          orientation=Quaternion(axis=[0, 0, 1], radians=yaw)))
    return result

#  project_box_to_cam 函数
def project_box_to_cam(box_lidar_list, cam_ins):
    """
    直接用 4×4 齐次矩阵把 8 个角点从 LiDAR 坐标系送到相机坐标系，
    然后用内参投影到像素平面，返回裁剪后的 [x1,y1,x2,y2] 或 None。
    """
    boxes2d_list = []
    for box_lidar in box_lidar_list:
        # --- 取关键参数 ---
        T = np.asarray(cam_ins['T_lidar2cam'], dtype=np.float32)  # 4×4
        K = np.asarray(cam_ins['intrinsic'], dtype=np.float32)  # 3×3
        w_img, h_img = cam_ins['img_wh']

        # --- LiDAR 8 个角点 → 齐次坐标 (4×8) ---
        corners_lidar = box_lidar.corners()  # 3×8
        corners_homo = np.vstack([corners_lidar, np.ones((1, 8))])

        # --- 变换到相机坐标系 ---
        corners_cam = T @ corners_homo  # 4×8
        z = corners_cam[2, :]
        if (z <= 0.1).any():  # 在背面 / 相机原点附近
            boxes2d_list.append(None)
            continue

        # --- 投影到像素 ---
        pts_2d = K @ corners_cam[:3, :]  # 3×8
        pts_2d[:2, :] /= pts_2d[2, :]

        x_min, y_min = pts_2d[0, :].min(), pts_2d[1, :].min()
        x_max, y_max = pts_2d[0, :].max(), pts_2d[1, :].max()

        # --- 裁剪到图像尺寸 ---
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w_img - 1, x_max), min(h_img - 1, y_max)

        if x_min >= x_max or y_min >= y_max:  # 被完全裁掉
            boxes2d_list.append(None)
            continue
        boxes2d_list.append([int(round(x_min)), int(round(y_min)),
                             int(round(x_max)), int(round(y_max))])
    return boxes2d_list

def iou_2d(boxA_list, boxB):
    """
    boxA/B: [x1,y1,x2,y2]
    """
    results_list = []
    if boxB[0]+boxB[1]+boxB[2]+boxB[3] == 0:
        for boxA in boxA_list:
            if boxA == None:
                results_list.append(-1)
            else:
                results_list.append(0)
        return results_list
    # return [0] * self.topk
    for boxA in boxA_list:
        if boxA == None:
            results_list.append(-1)
            continue
        xA = max(boxA[0], boxB[0].item())
        yA = max(boxA[1], boxB[1].item())
        xB = min(boxA[2], boxB[2].item())
        yB = min(boxA[3], boxB[3].item())

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter = inter_w * inter_h
        if inter == 0:
            results_list.append(0)
            continue
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        results_list.append(inter / float(areaA + areaB - inter))
    return results_list

def calculate_groundingDINO_score(
        prop_boxes_k = None,
        cam_dir_Ins=None,
        dino_pred_boxes=None):
    lidar_box = build_lidar_box(prop_boxes_k)
    proj_2d = project_box_to_cam(lidar_box, cam_dir_Ins)
    iou_val = iou_2d(proj_2d, dino_pred_boxes)

    return iou_val

def preprocess_dino_pred_boxes(dino_pred_boxes_list):
    # 如果 'dino_pred_boxes_list' 是空的，填充为 [[0, 0, 0, 0]]
    if len(dino_pred_boxes_list) == 0:
        return torch.tensor([0, 0, 0, 0], dtype=torch.float32)

    # 如果 'dino_pred_boxes_list' 中有多个 tensor，只保留第一个
    return dino_pred_boxes_list[0]

def gt_idx_from_all(prop_boxes, gt_box):
    prop_boxes = LiDARInstance3DBoxes(prop_boxes, box_dim=7)
    ious = LiDARInstance3DBoxes.overlaps(prop_boxes, gt_box.to(device)).squeeze(1)
    max_iou, gt_idx = ious.max(dim=0)
    # print(f'强制补GT,max_iou = {max_iou}')
    max_iou_list.append(max_iou)
    return gt_idx

def grounding_loss(
        boxes_dict,  # list  ·len = B·   每个元素含 'bboxes' [N,9] & 'scores' [N]
        proposal_feat=None,
        batch_gt_instances_3d=None,  # list  ·len = B·   InstanceData（只有 1 个 GT）
        dino_pred_boxes_list=None,
        cam_dir_Ins_list=None,
        text_features=None,
        device='cuda',
):
    batch_logits = []  # 收集每张图片的 logits
    batch_targets = []  # 收集每张图片的正样本索引
    gt_boxes_3d = batch_gt_instances_3d
    B = len(boxes_dict)
#加上下面两行代码，有一点提升，LiDARInstance3DBoxes_miou_yaw': 0.47465, 'recall_yaw_@0.25': 0.6023, 'recall_yaw@0.5': 0.5886
#       proposal_feat_fused = self.fuse_text_with_queries(proposal_feat, text_features)
#     proposal_feat = proposal_feat + proposal_feat_fused
    proposal_feat_fused = fuse_text_with_queries(proposal_feat, text_features, text_proj, cross_attn)
    proposal_feat = proposal_feat + proposal_feat_fused  # 直接相加
    for b in range(B):
        # --------------- 提取 per-image 数据 ---------------
        prop_scores = boxes_dict[b]['scores']  # [Ni]
        prop_boxes = boxes_dict[b]['bboxes'][:,:7]  # [Ni,7]
        prop_feats = proposal_feat[b]  # [Ni,C]
        gt_box = gt_boxes_3d[b][:7]  # [1,7]
        text_feat = text_features[b]  # [C]

        # --------------- (1) Top-k by detector score ---------------
        topk_scores, topk_idx = torch.topk(prop_scores, topk, largest=True, sorted=False)
        prop_feats_k = prop_feats[topk_idx]  # [k,C]
        prop_scores_k = prop_scores[topk_idx]  # [k]
        prop_boxes_k = prop_boxes[topk_idx]  # [k,7]
        groundingDINO_score_list = calculate_groundingDINO_score(prop_boxes_k = prop_boxes_k,
                                                                 cam_dir_Ins = cam_dir_Ins_list[b],
                                                                 dino_pred_boxes = dino_pred_boxes_list[b])
        prop_boxes_k = LiDARInstance3DBoxes(prop_boxes_k, box_dim=7)
        # --------------- (2) IoU 选 gt_idx ---------------
        ious = LiDARInstance3DBoxes.overlaps(prop_boxes_k, gt_box.to(device)).squeeze(1)  # [k]
        max_iou, rel_idx = ious.max(dim=0)  # rel_idx : 0…k-1
        if max_iou >= 0.1:
            gt_pos = rel_idx.item()  # 正样本已在 top-k
            # print(f'终于不补了,max_iou = {max_iou}')
            max_iou_list.append(max_iou)
        else:
            # 强制补 GT (追加到末尾)
            prop_feats_k = torch.cat([prop_feats_k, prop_feats[gt_idx_from_all(prop_boxes, gt_box)].unsqueeze(0)], dim=0)
            prop_scores_k = torch.cat([prop_scores_k, torch.tensor([1.], device=prop_scores_k.device)])
            gt_pos = prop_feats_k.size(0) - 1
            groundingDINO_score_list.append(0)
        text_feat_proj = text_proj2(text_feat)
        # --------------- (3) 文本-视觉相似度 logits ---------------
        logits = F.cosine_similarity(prop_feats_k, text_feat_proj.expand_as(prop_feats_k), dim=-1) / 0.1  #除以0.1是温度系数，防止值过于均匀，等价于“拉大间隔”

        # --------------- (4) optional: 融合置信度 ---------------
        groundingDINO_score_tensor = torch.tensor(groundingDINO_score_list).cuda(logits.device)
        logits = logits + 0.1 * torch.logit(prop_scores_k.detach(), eps=1e-6)
        logits = logits + logits.max() * groundingDINO_score_tensor
        # if random.random() < 0.05:
        #     print(f'groundingDINO_score_list[top20] = {sorted(groundingDINO_score_list, reverse=True)[:20]}, dino_pred_boxes_list[b] = {len(dino_pred_boxes_list[b])}')
        batch_logits.append(logits)  # 长度不同，先收集 list
        batch_targets.append(gt_pos)  # int

    # --------------- (5) Pad → Tensor [B, Kmax] ---------------
    Kmax = max(l.size(0) for l in batch_logits)
    logits_pad = torch.full((B, Kmax), -1e9, device=prop_feats.device)  # -∞ 填充
    for b, l in enumerate(batch_logits):
        logits_pad[b, :l.size(0)] = l

    targets = torch.tensor(batch_targets, dtype=torch.long, device=logits_pad.device)  # [B]

    # --------------- (6) Cross-entropy loss ---------------
    loss = F.cross_entropy(logits_pad, targets)
    # print(f'mean_sum_max_iou = {sum(self.max_iou_list) / len(self.max_iou_list)}')
    return loss


# 假设数据的字典结构已经解析好
class NuScenesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = self.data[idx]
        return {
            'bbox_token': result['bbox_token'],
            'boxes_bboxes': result['boxes_bboxes'],
            'boxes_scores': result['boxes_scores'],
            '3d_proposal_feature': result['3d_proposal_feature'],
            'text_features': result['text_features'],
            'dino_pred_boxes_list': result['dino_pred_boxes_list'],
            # 'lidar2CamIns': lidar2CamIns,
            'lidar2CamIns': result['lidar2CamIns'],
            'category_name': result['category_name'],
            'lidar_gt_center_bottom_3d_box':torch.tensor(result['lidar_gt_center_bottom_3d_box'],
                             dtype=torch.float32)
        }



def tensor_to_python(x: Any):
    """
    递归把 Tensor → float / int / list
    """
    if torch.is_tensor(x):
        if x.dim() == 0:          # 0-维：标量
            # .item() 得到 Python float/int
            return x.item()
        else:                     # 1-维及以上：用 .tolist()
            return x.tolist()

    if isinstance(x, (list, tuple)):
        return type(x)(tensor_to_python(e) for e in x)

    # 其它类型保持原样（str、dict、Path…）
    return x


def unpack_lidar2CamIns(collated: dict) -> List[dict]:
    """
    把 DataLoader 里“拼在一起”的 lidar2CamIns 拆回原始 list[dict]，
    并且把里面所有 Tensor 都转成 Python 数字 / list。
    """
    batch_size = len(collated['filename'])  # 依旧用 filename 判 batch

    def _slice(obj: Any, idx: int):
        # 1) Tensor：按 dim-0 切，再转 Python
        if torch.is_tensor(obj):
            return tensor_to_python(obj[idx])

        # 2) list/tuple：递归
        if isinstance(obj, (list, tuple)):
            # "批量列表"：元素非 list/Tensor → 直接取 idx
            if len(obj) == batch_size and not isinstance(obj[0], (list, tuple, torch.Tensor)):
                return obj[idx]
            return type(obj)(_slice(o, idx) for o in obj)

        # 3) 其它类型（str/标量…）
        return obj

    samples = []
    for i in range(batch_size):
        sample_dict = {k: _slice(v, i) for k, v in collated.items()}
        # 再次走一遍 tensor_to_python，确保 dict 深处的 Tensor 也转掉
        samples.append(tensor_to_python(sample_dict))

    return samples

def save_checkpoint(epoch: int, test_loss: float):
    ckpt_path = '/home/jiankunshi/python/Bevfusion_FineTuneByPre_GroundingDINO/TwoStageTrain/save_model/' + f"GT2DBox_epoch{epoch:02d}.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": text_proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "test_loss": test_loss,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

def grounding_evaluation(
        boxes_dict,  # list  ·len = B·   每个元素含 'bboxes' [N,9] & 'scores' [N]
        proposal_feat=None,
        batch_gt_instances_3d=None,  # list  ·len = B·   InstanceData（只有 1 个 GT）
        dino_pred_boxes_list=None,
        cam_dir_Ins_list=None,
        text_features=None,
        device='cuda',
):
    B = len(boxes_dict)
    boxes_list = []
    gt_boxes_3d = batch_gt_instances_3d
    proposal_feat_fused = fuse_text_with_queries(proposal_feat, text_features, text_proj, cross_attn)
    proposal_feat = proposal_feat + proposal_feat_fused  # 直接相加
    for b in range(B):
        prop_scores = boxes_dict[b]['scores']  # [Ni]
        prop_boxes = boxes_dict[b]['bboxes'][:, :7]  # [Ni,7]
        prop_feats = proposal_feat[b]  # [Ni,C]
        text_feat = text_features[b]  # [C]

        # --------------- (1) Top-k by detector score ---------------
        topk_scores, topk_idx = torch.topk(prop_scores, topk, largest=True, sorted=False)
        prop_feats_k = prop_feats[topk_idx]  # [k,C]
        prop_scores_k = prop_scores[topk_idx]  # [k]
        prop_boxes_k = prop_boxes[topk_idx]  # [k,7]
        groundingDINO_score_list = calculate_groundingDINO_score(prop_boxes_k=prop_boxes_k,
                                                                      cam_dir_Ins=cam_dir_Ins_list[b],
                                                                      dino_pred_boxes=dino_pred_boxes_list[b])
        # --------------- (2) IoU 选 gt_idx ---------------
        text_feat_proj = text_proj2(text_feat)
        # --------------- (3) 文本-视觉相似度 logits ---------------
        logits = F.cosine_similarity(prop_feats_k, text_feat_proj.expand_as(prop_feats_k), dim=-1) / 0.1  # 除以0.1是温度系数，防止值过于均匀，等价于“拉大间隔”

        # --------------- (4) optional: 融合置信度 ---------------
        groundingDINO_score_tensor = torch.tensor(groundingDINO_score_list).cuda(logits.device)
        logits = logits + 0.1 * torch.logit(prop_scores_k.detach(), eps=1e-6)
        logits = logits + logits.max() * groundingDINO_score_tensor
        # if random.random() < 0.05:
        #     print(f'groundingDINO_score_list[top20] = {sorted(groundingDINO_score_list, reverse=True)[:20]}, dino_pred_boxes_list[b] = {len(dino_pred_boxes_list[b])}')

        probs = torch.softmax(logits, dim=0)  # [k]
        best_idx_rel = probs.argmax()  # 0…k-1
        best_prob = probs[best_idx_rel]

        # 计算sum_max_iou
        # batch_gt_instances_3d = []
        # for data_sample in batch_data_samples:
        #     batch_input_metas.append(data_sample.metainfo)
        #     batch_gt_instances_3d.append(data_sample.gt_instances_grounding_3d)
        # gt_boxes_3d = prepare_gt_boxes(batch_gt_instances_3d, batch_feats[0].device)
        gt_box = gt_boxes_3d[b][:7]  # [1,7]
        prop_boxes = LiDARInstance3DBoxes(prop_boxes, box_dim=7)
        ious = LiDARInstance3DBoxes.overlaps(prop_boxes, gt_box.to(device)).squeeze(1)
        max_iou, gt_idx = ious.max(dim=0)
        # print(f'强制补GT,max_iou = {max_iou}')
        max_iou_list.append(max_iou)
        # print(f'mean_sum_max_iou = {sum(self.max_iou_list) / len(self.max_iou_list)}')
        # -------- (6) 回退策略 (可选) --------
        # if fallback_top1 and best_prob < 0.3:  # τ = 0.3 可调
        #     pred_box = prop_boxes_k[0]  # detector 置信度最高的框
        #     best_prob = topk_scores[0]
        # else:
        #     pred_box = prop_boxes_k[best_idx_rel]
        pred_box = prop_boxes_k[best_idx_rel]
        pred_box = LiDARInstance3DBoxes(pred_box.unsqueeze(0), box_dim=7)

        boxes_list.append(pred_box.to(device))
    return boxes_list

def compute_iou_tensor(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    box_a, box_b: Tensor of shape (6,), each representing [x, y, z, dx, dy, dz]
    Returns:
        IoU (float Tensor)
    """
    # Compute max corners
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = torch.min(max_a, max_b)

    # Compute min corners
    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = torch.max(min_a, min_b)

    # Check for non-overlap
    overlap = (min_max - max_min).clamp(min=0)
    intersection = overlap.prod()

    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection

    iou = intersection / union if union > 0 else torch.tensor(0.0, device=box_a.device)
    return iou

def rect_corners_xy(
        center_xy: torch.Tensor,  # (..., 2)
        dims_xy: torch.Tensor,  # (..., 2)
        yaw: torch.Tensor  # (...,)
) -> torch.Tensor:
    """
    计算 2D 旋转矩形四角点，返回 (..., 4, 2) tensor（顺时针）。
    """
    # 确保至少有 batch 维度
    if center_xy.ndim == 1:
        center_xy = center_xy.unsqueeze(0)  # (1, 2)
        dims_xy = dims_xy.unsqueeze(0)
        yaw = yaw.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    # 半长半宽
    h = dims_xy / 2.0  # (..., 2)
    # (4, 2) 局部模板
    corners_local = torch.tensor(
        [[1, 1],
         [-1, 1],
         [-1, -1],
         [1, -1]], dtype=center_xy.dtype, device=center_xy.device).unsqueeze(0)  # (1, 4, 2)
    corners_local = corners_local * h.unsqueeze(1)  # (..., 4, 2)

    # 构造旋转矩阵
    cos_y = torch.cos(yaw)[..., None, None]  # (..., 1, 1)
    sin_y = torch.sin(yaw)[..., None, None]
    rot = torch.cat([
        torch.cat([cos_y, -sin_y], dim=-1),
        torch.cat([sin_y, cos_y], dim=-1)
    ], dim=-2)  # (..., 2, 2)

    # 旋转 + 平移
    global_xy = (corners_local @ rot) + center_xy.unsqueeze(1)  # (..., 4, 2)

    if squeeze_out:
        global_xy = global_xy.squeeze(0)

    return global_xy

def pairwise_polygon_inter_area(
        poly1_xy: torch.Tensor,  # (..., 4, 2)
        poly2_xy: torch.Tensor  # (..., 4, 2)
) -> torch.Tensor:
    """
    用 Shapely 逐元素求交面积。为简单起见，先展平成 (N, 4, 2)。
    """
    # reshape => (-1, 4, 2)
    p1 = poly1_xy.reshape(-1, 4, 2).cpu().numpy()
    p2 = poly2_xy.reshape(-1, 4, 2).cpu().numpy()

    inter_areas = []
    for a, b in zip(p1, p2):
        poly_a = Polygon(a)
        poly_b = Polygon(b)
        inter_areas.append(poly_a.intersection(poly_b).area)
    return torch.tensor(inter_areas, dtype=poly1_xy.dtype, device=poly1_xy.device).reshape(poly1_xy.shape[:-2])

def intersection_height(
        z1: torch.Tensor, dz1: torch.Tensor,
        z2: torch.Tensor, dz2: torch.Tensor
) -> torch.Tensor:
    """Z 方向重叠高度 (...,)"""
    top1, bot1 = z1 + dz1 / 2.0, z1 - dz1 / 2.0
    top2, bot2 = z2 + dz2 / 2.0, z2 - dz2 / 2.0
    return torch.clamp_min(torch.minimum(top1, top2) - torch.maximum(bot1, bot2), 0.0)

def box3d_iou_yaw_tensor(
        box1: torch.Tensor,  # (..., 7)  [x,y,z,dx,dy,dz,yaw]
        box2: torch.Tensor  # 同形状或可 broadcast 至 box1
) -> torch.Tensor:
    """
    批量 3D IoU 计算（不可微）。

    返回形状 broadcast 后的 (...,) IoU 张量。
    """
    # 广播到同形
    box1, box2 = torch.broadcast_tensors(box1, box2)  # (..., 7)

    # 拆分字段
    x1, y1, z1, dx1, dy1, dz1, yaw1 = box1[:7].unbind(-1)
    x2, y2, z2, dx2, dy2, dz2, yaw2 = box2[:7].unbind(-1)

    # 1) XY 平面交面积
    corners1 = rect_corners_xy(torch.stack((x1, y1), -1),
                                torch.stack((dx1, dy1), -1),
                                yaw1)
    corners2 = rect_corners_xy(torch.stack((x2, y2), -1),
                                torch.stack((dx2, dy2), -1),
                                yaw2)
    inter_area = pairwise_polygon_inter_area(corners1, corners2)

    # 2) Z 重叠
    inter_h = intersection_height(z1, dz1, z2, dz2)

    # 3) 交 & 并
    inter_vol = inter_area * inter_h
    vol1 = dx1 * dy1 * dz1
    vol2 = dx2 * dy2 * dz2
    union_vol = vol1 + vol2 - inter_vol

    # 避免除零
    iou = torch.where(union_vol > 0, inter_vol / union_vol, torch.zeros_like(union_vol))
    return iou


def my_compute_metrics(pred_7dim_list ,gt_7dim_list, category_name_list):
    # 这里计算IoU、Accuracy、Recall等指标
    ious = []
    ious_yaw = []
    ious_LiDARInstance3DBoxes_list = []
    for i in range(len(gt_7dim_list)):
        if isinstance(pred_7dim_list[i], LiDARInstance3DBoxes):
            pred_7dim_tensor = pred_7dim_list[i].tensor.squeeze(0)
            pred_7dim_LiDARInstance3DBoxes = pred_7dim_list[i]
        else:
            pred_7dim_tensor = pred_7dim_list[i]
            pred_7dim_LiDARInstance3DBoxes = LiDARInstance3DBoxes(pred_7dim_list[i].unsqueeze(0), box_dim=7)

        if isinstance(gt_7dim_list[i], LiDARInstance3DBoxes):
            gt_7dim_tensor = gt_7dim_list[i].tensor.squeeze(0)
            gt_7dim_LiDARInstance3DBoxes = gt_7dim_list[i]
        else:
            gt_7dim_tensor = gt_7dim_list[i]
            gt_7dim_LiDARInstance3DBoxes = LiDARInstance3DBoxes(gt_7dim_list[i].unsqueeze(0), box_dim=7)

        iou = compute_iou_tensor(pred_7dim_tensor, gt_7dim_tensor)
        iou_yaw = box3d_iou_yaw_tensor(pred_7dim_tensor, gt_7dim_tensor)
        ious.append(iou)
        ious_yaw.append(iou_yaw)
        ious_LiDARInstance3DBoxes_list.append(LiDARInstance3DBoxes.overlaps(pred_7dim_LiDARInstance3DBoxes, gt_7dim_LiDARInstance3DBoxes))
    ious_tensor = torch.stack(ious)  # or torch.tensor(ious) if ious are scalars
    metrics_iou = {
        'mean_iou': float(ious_tensor.mean()),
        'recall@0.25': float((ious_tensor >= 0.25).float().mean()),
        'recall@0.5': float((ious_tensor >= 0.5).float().mean()),
    }
    ious_yaw_tensor = torch.stack(ious_yaw)  # 如果 ious_yaw 是 list of tensor
    metrics_iou_yaw = {
        'mean_iou_yaw': float(ious_yaw_tensor.mean()),
        'recall_yaw@0.25': float((ious_yaw_tensor >= 0.25).float().mean()),
        'recall_yaw@0.5': float((ious_yaw_tensor >= 0.5).float().mean()),
    }
    ious_LiDARInstance3DBoxes_tensor = torch.stack(ious_LiDARInstance3DBoxes_list)  # 如果 ious_yaw 是 list of tensor
    metrics_LiDARInstance3DBoxes_iou_yaw = {
        'LiDARInstance3DBoxes_mean_iou_yaw': float(ious_LiDARInstance3DBoxes_tensor.mean()),
        'LiDARInstance3DBoxes_recall_yaw_@0.25': float((ious_LiDARInstance3DBoxes_tensor >= 0.25).float().mean()),
        'LiDARInstance3DBoxes_recall_yaw@0.5': float((ious_LiDARInstance3DBoxes_tensor >= 0.5).float().mean()),
    }
    metric_result = {**metrics_iou, **metrics_iou_yaw, **metrics_LiDARInstance3DBoxes_iou_yaw}
    categories_thr_list = ['car', 'truck', 'vehicle.construction', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'trafficcone']
    threshold_A_dict = {'car': 0.5,'truck': 0.5,'vehicle.construction': 0.5,'bus': 0.5,'trailer': 0.5,'barrier': 0.25,'motorcycle': 0.25,'bicycle': 0.25,'pedestrian': 0.25,'trafficcone':0.25}
    threshold_B_dict = {'car': 0.7,'truck': 0.7,'vehicle.construction': 0.7,'bus': 0.7,'trailer': 0.7,'barrier': 0.5,'motorcycle': 0.5,'bicycle': 0.5,'pedestrian': 0.3,'trafficcone':0.3}
    pos_threshold_A_num = 0
    neg_threshold_A_num = 0
    pos_threshold_B_num = 0
    neg_threshold_B_num = 0
    for index, category_name_ori in enumerate(category_name_list):
        for index2, category_name in enumerate(categories_thr_list):
            if category_name in category_name_ori and ious_LiDARInstance3DBoxes_tensor[index] >= threshold_A_dict[category_name]:
                pos_threshold_A_num += 1
            elif category_name in category_name_ori and ious_LiDARInstance3DBoxes_tensor[index] < threshold_A_dict[category_name]:
                neg_threshold_A_num += 1
            if category_name in category_name_ori and ious_LiDARInstance3DBoxes_tensor[index] >= threshold_B_dict[category_name]:
                pos_threshold_B_num += 1
            elif category_name in category_name_ori and ious_LiDARInstance3DBoxes_tensor[index] < threshold_B_dict[category_name]:
                neg_threshold_B_num += 1
    metrics_threshold_A = pos_threshold_A_num / (pos_threshold_A_num + neg_threshold_A_num)
    metrics_threshold_B = pos_threshold_B_num / (pos_threshold_B_num + neg_threshold_B_num)

    print(f'metrics_iou = {metrics_iou}')
    print(f'metrics_iou_yaw = {metrics_iou_yaw}')
    print(f'metrics_LiDARInstance3DBoxes_iou_yaw = {metrics_LiDARInstance3DBoxes_iou_yaw}')
    print(f'metrics_threshold_A = {metrics_threshold_A}, metrics_threshold_B = {metrics_threshold_B}, 有{len(category_name_list)-pos_threshold_A_num-neg_threshold_A_num}个未参与类别计算')

@torch.no_grad()
def test_model(test_loader):
    text_proj.eval()
    text_proj2.eval()
    cross_attn.eval()
    running_loss = 0.0
    outputs_list = []
    gt_instances_3d_list = []
    category_name_list = []
    for batch_idx, batch_data in enumerate(test_loader):
        # 获取批次数据
        # bbox_tokens = batch_data['bbox_token'].to(device)
        boxes_bboxes = batch_data['boxes_bboxes'].to(device)
        boxes_scores = batch_data['boxes_scores'].to(device)
        proposal_features = batch_data['3d_proposal_feature'].to(device)
        text_features = batch_data['text_features'].to(device)
        dino_pred_boxes = batch_data['dino_pred_boxes_list']
        lidar2CamIns = batch_data['lidar2CamIns']
        gt_instances_3d = batch_data['lidar_gt_center_bottom_3d_box']
        category_names = batch_data['category_name']
        boxes_dict, batch_gt_instances_3d, dino_pred_boxes_list, cam_dir_Ins_list = [], [], [], []
        # print(category_names)
        for categroy_name in category_names:
            category_name_list.append(categroy_name)
        for index, (box_score, box_box) in enumerate(zip(boxes_scores, boxes_bboxes)):
            boxes_dict.append({'bboxes': boxes_bboxes[index], 'scores': boxes_scores[index]})
        for gt_3d_box in gt_instances_3d:
            # gt_3d_center = gt_3d_box[:3]
            # gt_3d_size = gt_3d_box[3:6]
            # sin_yaw = np.sin(gt_3d_box[6])
            # cos_yaw = np.cos(gt_3d_box[6])
            gt_bboxes_3d = LiDARInstance3DBoxes(gt_3d_box[None, :], box_dim=7, origin=(0.5, 0.5, 0.5))
            batch_gt_instances_3d.append(gt_bboxes_3d.to(device))
            gt_instances_3d_list.append(gt_3d_box.clone().to(device))
        # print(f'batch_gt_instances_3d[0].equal(gt_instances_3d_list[0]) = {torch.equal(batch_gt_instances_3d[0].tensor,gt_instances_3d_list[0])}')
        # print(batch_gt_instances_3d[0].tensor)
        # print(gt_instances_3d_list[0])
        # print(gt_3d_box.shape)
        # print(torch.stack(gt_instances_3d_list).shape)
        for dino_pred_box in dino_pred_boxes:
            dino_pred_boxes_list.append(dino_pred_box)
        lidar2CamIns_list = unpack_lidar2CamIns(lidar2CamIns)
        outputs = grounding_evaluation(boxes_dict=boxes_dict,  # list[B]
                              proposal_feat=proposal_features,
                              batch_gt_instances_3d=batch_gt_instances_3d,
                              dino_pred_boxes_list=dino_pred_boxes_list,
                              cam_dir_Ins_list=lidar2CamIns_list,
                              text_features=text_features)
        # print(f'len(outputs) = {len(outputs)}')
        outputs_list+=outputs
        # print(f'batch_gt_instances_3d[0].equal(gt_instances_3d_list[0]) = {torch.equal(batch_gt_instances_3d[0].tensor,gt_instances_3d_list[0])}')
        # print(batch_gt_instances_3d[0].tensor)
        # print(gt_instances_3d_list[0])
        # print(f'len(outputs_list) = {len(outputs_list)}')
        # print(f'batch_gt_instances_3d[0].equal(gt_instances_3d_list[0]) = {batch_gt_instances_3d[0].equal(gt_instances_3d_list[0])}')
        print(f'batch_idx = {batch_idx}')

    gt_instances_3d_tensor = torch.stack(gt_instances_3d_list)
    my_compute_metrics(outputs_list, gt_instances_3d_tensor, category_name_list)


def train_model(train_loader, test_loader):
    running_loss = 0
    # 训练循环
    st_time_epoch = time.time()
    for epoch in range(epochs):
        text_proj.train()
        text_proj2.train()
        cross_attn.train()
        gt_instances_3d_list = []
        for batch_idx, batch_data in enumerate(train_loader):
            # 获取批次数据
            # bbox_tokens = batch_data['bbox_token'].to(device)
            boxes_bboxes = batch_data['boxes_bboxes'].to(device)
            boxes_scores = batch_data['boxes_scores'].to(device)
            proposal_features = batch_data['3d_proposal_feature'].to(device)
            text_features = batch_data['text_features'].to(device)
            dino_pred_boxes = batch_data['dino_pred_boxes_list']
            lidar2CamIns = batch_data['lidar2CamIns']
            gt_instances_3d = batch_data['lidar_gt_center_bottom_3d_box']
            boxes_dict,batch_gt_instances_3d,dino_pred_boxes_list,cam_dir_Ins_list = [],[],[],[]

            for index, (box_score, box_box) in enumerate(zip(boxes_scores,boxes_bboxes)):
                boxes_dict.append({'bboxes':boxes_bboxes[index], 'scores':boxes_scores[index]})
            for gt_3d_box in gt_instances_3d:
                # gt_3d_center = gt_3d_box[:3]
                # gt_3d_size = gt_3d_box[3:6]
                # sin_yaw = np.sin(gt_3d_box[6])
                # cos_yaw = np.cos(gt_3d_box[6])
                gt_bboxes_3d = LiDARInstance3DBoxes(gt_3d_box[None, :], box_dim=7, origin=(0.5, 0.5, 0.5))
                batch_gt_instances_3d.append(gt_bboxes_3d)
            for dino_pred_box in dino_pred_boxes:
                dino_pred_boxes_list.append(dino_pred_box)
            lidar2CamIns_list = unpack_lidar2CamIns(lidar2CamIns)

            optimizer.zero_grad(set_to_none=True)
            loss = grounding_loss(boxes_dict=boxes_dict,  # list[B]
                    proposal_feat=proposal_features,
                    batch_gt_instances_3d=batch_gt_instances_3d,
                    dino_pred_boxes_list = dino_pred_boxes_list,
                    cam_dir_Ins_list = lidar2CamIns_list,
                    text_features=text_features)
            loss.backward()
            # Gradient clipping (max_norm = 30, norm_type = 2)
            # torch.nn.utils.clip_grad_norm_(text_proj.parameters(), max_norm=30.0, norm_type=2)
            running_loss += loss.item() * batch_size

            optimizer.step()
            print(f'epoch = {epoch}, batch_idx = {batch_idx}, loss = {loss.item()}')

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch+1) % 5 == 0:
            save_checkpoint(epoch, epoch_loss)
        print(f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        test_model(test_loader)
        print(f'一个epoch花费时间为:{time.time() - st_time_epoch}')


GT_box_flag = True
None_cnt_train = 0
None_cnt_test = 0

st_time = time.time()
# 加载数据
train_file_path = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/merge_train_one_stage_output.pkl'
test_file_path = '/data_volume_1/sjk_data/NuscenesGrounding/talk2car_dataset/merge_test_one_stage_output.pkl'
with open(train_file_path, 'rb') as f:
    train_data = pickle.load(f)
for tmp_data in train_data:
    tmp_data['dino_pred_boxes_list'] = preprocess_dino_pred_boxes(tmp_data['dino_pred_boxes_list'])
    if GT_box_flag:
        if tmp_data['proj_bbox']:
            tmp_data['dino_pred_boxes_list'] = torch.tensor(tmp_data['proj_bbox'])
        else:
            None_cnt_train = None_cnt_train+1
            tmp_data['dino_pred_boxes_list'] = torch.tensor([0,0,0,0])
print(f'None_cnt_train = {None_cnt_train}')

with open(test_file_path, 'rb') as f:
    test_data = pickle.load(f)
for tmp_data in test_data:
    tmp_data['dino_pred_boxes_list'] = preprocess_dino_pred_boxes(tmp_data['dino_pred_boxes_list'])
    if GT_box_flag:
        if tmp_data['proj_bbox']:
            tmp_data['dino_pred_boxes_list'] = torch.tensor(tmp_data['proj_bbox'])
        else:
            None_cnt_test = None_cnt_test + 1
            tmp_data['dino_pred_boxes_list'] = torch.tensor([0, 0, 0, 0])
print(f'None_cnt_test = {None_cnt_test}')

text_proj = text_proj.to(device)
text_proj2 = text_proj2.to(device)
cross_attn = cross_attn.to(device)
lr = 1e-4
weight_decay = 1e-2
optimizer = torch.optim.AdamW(
    text_proj.parameters(),
    lr=lr,
    weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


# 定义模型
input_dim = 768  # 假设文本特征和proposal feature的维度是768

# 创建数据集与数据加载器
batch_size = 256
train_dataset = NuScenesDataset(train_data)
test_dataset = NuScenesDataset(test_data)
# data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

train_model(train_loader, test_loader)

# checkpoint = torch.load('/home/jiankunshi/python/Bevfusion_FineTuneByPre_GroundingDINO/TwoStageTrain/save_model/epoch20_5304_4658_talk2car.pth')
# checkpoint = torch.load('/home/jiankunshi/python/Bevfusion_FineTuneByPre_GroundingDINO/TwoStageTrain/save_model/GT2DBox_epoch20_7727_6648_talk2car.pth')
# text_proj.load_state_dict(checkpoint['model_state_dict'])
# test_model(test_loader)


# ckpt = torch.load('/home/jiankunshi/python/Bevfusion_FineTuneByPre_GroundingDINO/work_dirs/conf_two_stage_grounding_by_bevfusion_lidar_detection_camera_grounding/epoch_5.pth')
# source_state = ckpt['state_dict']   # 根据实际情况改键名
# target_state = text_proj.state_dict()
# #
# # # 过滤：只留下在 text_proj 中也存在且形状一致的参数
# filtered_state = {
#     # k = k.replace("text_proj.", "")
#     k: v for k, v in source_state.items()
#     if k in target_state and v.shape == target_state[k].shape
# }
# #
# # # 更新并加载
# target_state.update(filtered_state)
# text_proj.load_state_dict(target_state)
#
# print("已成功加载:", filtered_state.keys())
# test_model(test_loader)

print(f'总花费时间为:{time.time() - st_time}')
#
# 下面代码是根据维度检索文本映射层
# state_dict = (
#     ckpt['state_dict']                # 常见：MMEngine / Lightning
#     if isinstance(ckpt, dict) and 'state_dict' in ckpt
#     else ckpt                         # 直接是 plain state_dict
# )
# target_shape = torch.Size([256, 768])
# matches = [name for name, tensor in state_dict.items()
#            if tensor.shape == target_shape]
#
# print(f'共有 {len(matches)} 个参数满足 {target_shape}:')
# for n in matches:
#     print('  -', n)