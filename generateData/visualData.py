#为了验证生成的final数据可靠性，需要将其可视化
#这个函数是测试读取
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image
from pyquaternion import Quaternion
import time
import mmcv
import pickle

def get_matrix(calibrated_data, inverse=False):
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data['rotation']).rotation_matrix
    output[:3, 3] = calibrated_data['translation']
    if inverse:
        return np.linalg.inv(output)
    return output

# Function to plot 2D bounding box
def plot_2d_bbox(image, bbox, caption):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    # Split the caption into multiple lines if it's too long
    max_width = 40
    lines = [caption[i:i+max_width] for i in range(0, len(caption), max_width)]
    y_offset = y_min - 10
    for line in lines:
        cv2.putText(image, line, (x_min, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset -= 20

# Visualize the data
def visualize_annotation_by_2D_box(nusc, data, token):
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    annotation = data[token]
    sample_token = annotation['sample_token']
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    if 'Rainy' in scene['description'] or 'rainy' in scene['description']:
        annotation['weather'] = 'Rainy'
    elif 'Night' in scene['description'] or 'night' in scene['description']:
        annotation['weather'] = 'Night'
    elif 'Sunny' in scene['description'] or 'sunny' in scene['description']:
        annotation['weather'] = 'Sunny'
    elif 'Snowy' in scene['description'] or 'snowy' in scene['description']:
        annotation['weather'] = 'Snowy'
    else:
        annotation['weather'] = 'Manual view'
    cam = annotation['cam_file'].split('/')[1]
    # Get camera data
    cam_data = nusc.get('sample_data', sample['data'][cam])
    cam_path = os.path.join(nusc.dataroot, cam_data['filename'])
    image = cv2.imread(cam_path)

    # Plot 2D bounding box
    bbox_2d = annotation['2d_bbox']
    attribute_caption = annotation['attribute_caption']['attribute_caption']
    depth_caption = annotation['depth_caption']['depth_caption']
    localization_caption = annotation['localization_caption']['localization_caption']
    motion_caption = annotation['motion_caption']['motion_caption']
    map_caption = annotation['map_caption']['map_caption']

    # Extract orientation and speed
    orientation = annotation['localization_caption']['localization_theta']
    speed = annotation['motion_caption']['bev_vel']

    # Combine captions
    combined_caption = (f"{attribute_caption}, {depth_caption}, {localization_caption}, "
                        f"{motion_caption}, {map_caption}, Orientation: {orientation:.2f} degrees, Speed: {speed:.2f} m/s")
    plot_2d_bbox(image, bbox_2d, combined_caption)

    # Save the image with 2D bounding box
    output_dir = "/data_volume_1/sjk_data/tod3cap/visualization/monoViewAndPrompt/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{token}_2d_3d_bbox.png")

    # Plot 3D bounding box
    for cam in cameras:
        camera_token = sample['data'][cam]
        box = nusc.get_box(token)
        corners = box.corners().T
        global_corners = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
        camera_ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        #ego_pose 本身就是基于global而言的
        global_to_ego = get_matrix(camera_ego_pose, True)# global->ego

        camera_calibrated = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_to_camera = get_matrix(camera_calibrated, True) # camera -> ego
        camera_intrinsic = np.eye(4)
        camera_intrinsic[:3, :3] = camera_calibrated['camera_intrinsic']

        #global -> camera_ego_pose -> image
        global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego
        image_based_corners = global_corners @ global_to_image.T
        image_based_corners[:, :2] /= image_based_corners[:, [2]]
        image_based_corners = image_based_corners.astype(np.int32)#为什么变为

        #画线
        ix, iy = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 1, 2, 3, 0, 5, 6, 7, 4]
        line_flag = True
        for p0, p1 in zip(image_based_corners[ix], image_based_corners[iy]):
            if p0[2] <= 0 or p1[2] <= 0: continue
            if p0[0] > image.shape[1] or p0[1] > image.shape[0] or p1[0] > image.shape[1] or p1[1] > image.shape[0]:
                line_flag = False
                break
            if p0[0] < 0 or p1[0] < 0 or p0[1] < 0 or p1[1] < 0:
                line_flag = False
                break
            cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (0, 0, 255), 2, 16)
        cv2.imwrite(output_path, image)

# Function to plot text
def plot_text(image, bbox, caption):
    x_min, y_min, x_max, y_max = bbox
    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    cv2.putText(image, caption, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to plot 3D bounding box
def plot_3d_bbox(ax, box: Box, view, color='red'):
    corners = view_points(box.corners(), view, normalize=True)[:2, :]
    for i in range(4):
        ax.plot([corners[0, i], corners[0, (i + 1) % 4]],
                [corners[1, i], corners[1, (i + 1) % 4]], color=color)
        ax.plot([corners[0, i + 4], corners[0, (i + 1) % 4 + 4]],
                [corners[1, i + 4], corners[1, (i + 1) % 4 + 4]], color=color)
        ax.plot([corners[0, i], corners[0, i + 4]],
                [corners[1, i], corners[1, i + 4]], color=color)

# Visualize the data
def visualize_annotation_by_3D_box(nusc, annotation, token):
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    sample_token = annotation['sample_token']
    sample = nusc.get('sample', sample_token)
    cam = annotation['cam_file'].split('/')[1]
    # Get camera data
    cam_data = nusc.get('sample_data', sample['data'][cam])
    cam_path = os.path.join(nusc.dataroot, cam_data['filename'])
    image = cv2.imread(cam_path)

    # Plot 2D bounding box
    bbox_2d = annotation['2d_bbox']
    attribute_caption = annotation['attribute_caption']['attribute_caption']
    depth_caption = annotation['depth_caption']['depth_caption']
    localization_caption = annotation['localization_caption']['localization_caption']
    motion_caption = annotation['motion_caption']['motion_caption']
    map_caption = annotation['map_caption']['map_caption']

    # Combine captions
    combined_caption = f"{attribute_caption}, {depth_caption}, {localization_caption}, {motion_caption}, {map_caption}"
    plot_text(image, bbox_2d, combined_caption)

    # Save the image with 2D bounding box
    output_dir = "/data_volume_1/sjk_data/tod3cap/visualization/monoViewAndPrompt/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{token}_3d_bbox.png")

    # Plot 3D bounding box
    for cam in cameras:
        camera_token = sample['data'][cam]
        box = nusc.get_box(token)
        corners = box.corners().T
        global_corners = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
        camera_ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        #ego_pose 本身就是基于global而言的
        global_to_ego = get_matrix(camera_ego_pose, True)# global->ego

        camera_calibrated = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_to_camera = get_matrix(camera_calibrated, True) # camera -> ego
        camera_intrinsic = np.eye(4)
        camera_intrinsic[:3, :3] = camera_calibrated['camera_intrinsic']

        #global -> camera_ego_pose -> image
        global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego
        image_based_corners = global_corners @ global_to_image.T
        image_based_corners[:, :2] /= image_based_corners[:, [2]]
        image_based_corners = image_based_corners.astype(np.int32)#为什么变为

        #画线
        ix, iy = [0,1,2,3,0,1,2,3,4,5,6,7],[4,5,6,7,1,2,3,0,5,6,7,4]
        line_flag = True
        for p0, p1 in zip(image_based_corners[ix], image_based_corners[iy]):
            if p0[2] <= 0 or p1[2] <= 0: continue
            if p0[0] > image.shape[1] or p0[1] > image.shape[0] or p1[0] > image.shape[1] or p1[1] >image.shape[0]:
                line_flag = False
                break
            if p0[0] < 0 or p1[0] < 0 or p0[1] < 0 or p1[1] < 0:
                line_flag = False
                break
            cv2.line(image, (p0[0], p0[1]), (p1[0],p1[1]), (0, 0, 255), 2, 16)
        cv2.imwrite(output_path, image)

def visualize_nuscenes_pkl_file_by_3D_box(nusc, data, annotation):
    cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    sample_token = annotation['sample_token']
    sample = nusc.get('sample', sample_token)
    cam = annotation['cam_file'].split('/')[1]
    # Get camera data
    cam_data = nusc.get('sample_data', sample['data'][cam])
    cam_path = os.path.join(nusc.dataroot, cam_data['filename'])
    image = cv2.imread(cam_path)

    # Plot 2D bounding box
    bbox_2d = annotation['2d_bbox']
    attribute_caption = annotation['attribute_caption']['attribute_caption']
    depth_caption = annotation['depth_caption']['depth_caption']
    localization_caption = annotation['localization_caption']['localization_caption']
    motion_caption = annotation['motion_caption']['motion_caption']
    map_caption = annotation['map_caption']['map_caption']

    # Combine captions
    combined_caption = f"{attribute_caption}, {depth_caption}, {localization_caption}, {motion_caption}, {map_caption}"
    plot_text(image, bbox_2d, combined_caption)

    # Save the image with 2D bounding box
    output_dir = "/data_volume_1/sjk_data/nuscenes_caption_streamPERT/visual_predict_3D_box/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{annotation['bbox_token']}_3d_bbox.png")

    # Plot 3D bounding box
    for cam in cameras:
        camera_token = sample['data'][cam]
        box = nusc.get_box(annotation['bbox_token'])
        # a_nusc_get_box = nusc.get_box(annotation['bbox_token'])
        # lidar_token = nusc.get('sample', annotation['sample_token'])['data']['LIDAR_TOP']
        # lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        #注意，上面得到的a_nusc_get_box值和tod3cap的box值一致，但是和通过lidar_token得到的box值和旋转角度不一致，但是size一致
        #上述不一致的原因不明。
        corners = box.corners().T
        global_corners = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
        camera_ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # ego_pose 本身就是基于global而言的
        global_to_ego = get_matrix(camera_ego_pose, True)  # global->ego

        camera_calibrated = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_to_camera = get_matrix(camera_calibrated, True)  # camera -> ego
        camera_intrinsic = np.eye(4)
        camera_intrinsic[:3, :3] = camera_calibrated['camera_intrinsic']

        # global -> camera_ego_pose -> image
        global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego
        image_based_corners = global_corners @ global_to_image.T
        image_based_corners[:, :2] /= image_based_corners[:, [2]]
        image_based_corners = image_based_corners.astype(np.int32)  # 为什么变为

        # 画线
        ix, iy = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 1, 2, 3, 0, 5, 6, 7, 4]
        line_flag = True
        for p0, p1 in zip(image_based_corners[ix], image_based_corners[iy]):
            if p0[2] <= 0 or p1[2] <= 0: continue
            if p0[0] > image.shape[1] or p0[1] > image.shape[0] or p1[0] > image.shape[1] or p1[1] > image.shape[0]:
                line_flag = False
                break
            if p0[0] < 0 or p1[0] < 0 or p0[1] < 0 or p1[1] < 0:
                line_flag = False
                break
            cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (0, 0, 255), 2, 16)
        cv2.imwrite(output_path, image)

# Example usage
if __name__ == '__main__':
    # vis_flag = 1 #表示可视化源data的数据集
    vis_flag = 2 #表示可视化model预测的数据集
    #可视化
    if vis_flag == 1:
        # Load NuScenes dataset
        nusc = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)
        # Load annotation data
        # annotation_file = "/data_volume_1/sjk_data/tod3cap/final_caption_bbox_token.json"
        annotation_file = "/home/jiankunshi/python/NuscenesGrounding/generateData/final_data.pkl"
        with open(annotation_file, 'rb') as f:
            # data = json.load(f)
            data = pickle.load(f)
        for sample in data:
            token = sample['sample_token']
            st_time = time.time()
            # visualize_annotation_by_2D_box(nusc, data, token) #2D 可视化
            print(f'当前可视化的数据描述为：{sample["nlp_desc"]}')
            visualize_annotation_by_3D_box(nusc, sample, token) #3D 可视化
            print('Generating an object took: ' + str(time.time() - st_time))
    elif vis_flag == 2:
        # Load NuScenes dataset
        nusc = NuScenes(version='v1.0-trainval', dataroot='/data_volume_3/nuscenes/v1_0/', verbose=True)

        # Load annotation data
        nuscenes_pkl_file = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns.pkl"
        nuscenes_data = mmcv.load(nuscenes_pkl_file, file_format='pkl')
        for token in nuscenes_data:
            visualize_nuscenes_pkl_file_by_3D_box(nusc, nuscenes_data, token)  # 3D 可视化
