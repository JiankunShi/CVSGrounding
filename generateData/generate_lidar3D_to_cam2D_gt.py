#之前用的是sample的bbox_2d做groundingTruth，但是这个值和lidar坐标系下3DBoundingbox转相机坐标系下2d Bounding box不太一致，mean_IOU为0.91左右，
# 所以这里直接生成lidar坐标系下3DBoundingbox转相机坐标系下2d Bounding box作为最终的groundingTruth
#因为会出现极个别不可见的情况，所以这时用smp['proj_bbox'] = gt_2d

import pickle, tqdm, numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


# ---------- 路径 ----------
PKL_IN  = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns.pkl'
PKL_OUT = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2.pkl'


# ---------- 1. 通用工具 ----------
def build_lidar_box(raw_box):
    """
    raw_box: [x_btm, y_btm, z_btm, length, width, height, yaw]
    转成 nuScenes Box (几何中心、[w,l,h])。
    """
    x_btm, y_btm, z_btm = raw_box[:3]
    l_len, w_len, h = raw_box[3:6]
    yaw = raw_box[6]
    center = [x_btm, y_btm, z_btm + h / 2]     # 底面中心 → 几何中心
    size   = [w_len, l_len, h]                 # [w,l,h]
    return Box(center=center,
               size=size,
               orientation=Quaternion(axis=[0, 0, 1], radians=yaw))


def project_box_to_cam(box_lidar, cam_ins):
    """
    直接用 4×4 齐次矩阵把 8 个角点从 LiDAR 坐标系送到相机坐标系，
    然后用内参投影到像素平面，返回裁剪后的 [x1,y1,x2,y2] 或 None。
    """
    # --- 取关键参数 ---
    T = np.asarray(cam_ins['T_lidar2cam'], dtype=np.float32)   # 4×4
    K = np.asarray(cam_ins['intrinsic'] , dtype=np.float32)    # 3×3
    w_img, h_img = cam_ins['img_wh']

    # --- LiDAR 8 个角点 → 齐次坐标 (4×8) ---
    corners_lidar = box_lidar.corners()                       # 3×8
    corners_homo  = np.vstack([corners_lidar, np.ones((1, 8))])

    # --- 变换到相机坐标系 ---
    corners_cam = T @ corners_homo                            # 4×8
    z = corners_cam[2, :]
    if (z <= 0.1).any():                                      # 在背面 / 相机原点附近
        return None

    # --- 投影到像素 ---
    pts_2d = K @ corners_cam[:3, :]                           # 3×8
    pts_2d[:2, :] /= pts_2d[2, :]

    x_min, y_min = pts_2d[0, :].min(), pts_2d[1, :].min()
    x_max, y_max = pts_2d[0, :].max(), pts_2d[1, :].max()

    # --- 裁剪到图像尺寸 ---
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w_img - 1, x_max), min(h_img - 1, y_max)

    if x_min >= x_max or y_min >= y_max:                      # 被完全裁掉
        return None

    return [int(round(x_min)), int(round(y_min)),
            int(round(x_max)), int(round(y_max))]



def iou_2d(boxA, boxB):
    """
    boxA/B: [x1,y1,x2,y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def get_cam_from_path(path: str):
    """
    根据文件路径推断所属相机名
    """
    for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_FRONT',
                'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',  'CAM_BACK']:
        if cam in path:
            return cam
    raise ValueError(f'Cannot infer camera from path: {path}')


# ---------- 2. 主流程 ----------
with open(PKL_IN, 'rb') as f:
    data = pickle.load(f)

samples  = data['data_list']
iou_vals = []

for smp in tqdm.tqdm(samples, desc='Project & IoU'):
    # 2-D Ground-truth
    gt_2d = smp['2d_bbox']           # [x1,y1,x2,y2]

    # 相机名
    cam = get_cam_from_path(smp['cam_file'])

    # LiDAR 3-D Box
    lidar_box = build_lidar_box(smp['lidar_gt_center_bottom_3d_box'])

    # 投影
    cam_ins   = smp['lidar2CamIns'][cam]
    proj_2d   = project_box_to_cam(lidar_box, cam_ins)

    smp['proj_bbox'] = proj_2d

    # IoU
    if proj_2d is not None:
        iou_val = iou_2d(proj_2d, gt_2d)
    #     smp['proj_iou'] = iou_val
        iou_vals.append(iou_val)
    else:
        print(gt_2d)
        smp['proj_bbox'] = gt_2d
        # smp['proj_iou'] = None   # 不可见 / 被裁剪

# ---------- 3. 汇总并写回 ----------
mean_iou = np.mean(iou_vals) if iou_vals else 0.0
print(f'Valid projections: {len(iou_vals)} |  Mean IoU = {mean_iou:.4f}')

with open(PKL_OUT, 'wb') as f:
    pickle.dump(data, f)

print(f'Result saved to: {PKL_OUT}')


# ---------- 路径 ----------
PKL_IN  = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns.pkl'
PKL_OUT = '/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_val_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2.pkl'


# ---------- 1. 通用工具 ----------
def build_lidar_box(raw_box):
    """
    raw_box: [x_btm, y_btm, z_btm, length, width, height, yaw]
    转成 nuScenes Box (几何中心、[w,l,h])。
    """
    x_btm, y_btm, z_btm = raw_box[:3]
    l_len, w_len, h = raw_box[3:6]
    yaw = raw_box[6]
    center = [x_btm, y_btm, z_btm + h / 2]     # 底面中心 → 几何中心
    size   = [w_len, l_len, h]                 # [w,l,h]
    return Box(center=center,
               size=size,
               orientation=Quaternion(axis=[0, 0, 1], radians=yaw))


def project_box_to_cam(box_lidar, cam_ins):
    """
    直接用 4×4 齐次矩阵把 8 个角点从 LiDAR 坐标系送到相机坐标系，
    然后用内参投影到像素平面，返回裁剪后的 [x1,y1,x2,y2] 或 None。
    """
    # --- 取关键参数 ---
    T = np.asarray(cam_ins['T_lidar2cam'], dtype=np.float32)   # 4×4
    K = np.asarray(cam_ins['intrinsic'] , dtype=np.float32)    # 3×3
    w_img, h_img = cam_ins['img_wh']

    # --- LiDAR 8 个角点 → 齐次坐标 (4×8) ---
    corners_lidar = box_lidar.corners()                       # 3×8
    corners_homo  = np.vstack([corners_lidar, np.ones((1, 8))])

    # --- 变换到相机坐标系 ---
    corners_cam = T @ corners_homo                            # 4×8
    z = corners_cam[2, :]
    if (z <= 0.1).any():                                      # 在背面 / 相机原点附近
        return None

    # --- 投影到像素 ---
    pts_2d = K @ corners_cam[:3, :]                           # 3×8
    pts_2d[:2, :] /= pts_2d[2, :]

    x_min, y_min = pts_2d[0, :].min(), pts_2d[1, :].min()
    x_max, y_max = pts_2d[0, :].max(), pts_2d[1, :].max()

    # --- 裁剪到图像尺寸 ---
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w_img - 1, x_max), min(h_img - 1, y_max)

    if x_min >= x_max or y_min >= y_max:                      # 被完全裁掉
        return None

    return [int(round(x_min)), int(round(y_min)),
            int(round(x_max)), int(round(y_max))]



def iou_2d(boxA, boxB):
    """
    boxA/B: [x1,y1,x2,y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def get_cam_from_path(path: str):
    """
    根据文件路径推断所属相机名
    """
    for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_FRONT',
                'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',  'CAM_BACK']:
        if cam in path:
            return cam
    raise ValueError(f'Cannot infer camera from path: {path}')


# ---------- 2. 主流程 ----------
with open(PKL_IN, 'rb') as f:
    data = pickle.load(f)

samples  = data['data_list']
iou_vals = []

for smp in tqdm.tqdm(samples, desc='Project & IoU'):
    # 2-D Ground-truth
    gt_2d = smp['2d_bbox']           # [x1,y1,x2,y2]

    # 相机名
    cam = get_cam_from_path(smp['cam_file'])

    # LiDAR 3-D Box
    lidar_box = build_lidar_box(smp['lidar_gt_center_bottom_3d_box'])

    # 投影
    cam_ins   = smp['lidar2CamIns'][cam]
    proj_2d   = project_box_to_cam(lidar_box, cam_ins)

    smp['proj_bbox'] = proj_2d

    # IoU
    if proj_2d is not None:
        iou_val = iou_2d(proj_2d, gt_2d)
        # smp['proj_iou'] = iou_val
        iou_vals.append(iou_val)
    else:
        print(gt_2d)
        smp['proj_bbox'] = gt_2d
        # smp['proj_iou'] = 1000000   # 不可见 / 被裁剪

# ---------- 3. 汇总并写回 ----------
mean_iou = np.mean(iou_vals) if iou_vals else 0.0
print(f'Valid projections: {len(iou_vals)} |  Mean IoU = {mean_iou:.4f}')

with open(PKL_OUT, 'wb') as f:
    pickle.dump(data, f)

print(f'Result saved to: {PKL_OUT}')
