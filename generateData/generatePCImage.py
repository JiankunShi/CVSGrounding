#这个代码是根据点云文件路径，可视化彩色点云，并存储为PNG格式
import pickle
from pathlib import Path
import numpy as np
import open3d as o3d
from matplotlib import cm

# ---------- 数据加载 & 上色 ---------- #
def load_nuscenes_lidar_bin(bin_path) -> np.ndarray:
    """nuScenes LIDAR_TOP .bin → (N,5) float32"""
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)

def colorize_by_height(xyz: np.ndarray,
                       cmap=cm.get_cmap("viridis")) -> np.ndarray:
    """按 z 值上色 → uint8 RGB"""
    z = xyz[:, 2]
    norm = (z - z.min()) / (z.ptp() + 1e-6)
    rgb = (cmap(norm)[:, :3] * 255).astype(np.uint8)
    return rgb

# ---------- 离屏渲染 & 保存 ---------- #
def save_pointcloud_png(xyz: np.ndarray,
                        colors: np.ndarray,
                        out_path,
                        width: int = 1920,
                        height: int = 1080,
                        fov: float = 60.0) -> None:
    """使用 OffscreenRenderer 保存 PNG"""
    # PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # 视角参数
    center = pcd.get_center()
    extent = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    eye_offset = np.array([0, -1.5 * extent, 0.3 * extent])
    up_vec = [0, 0, 1]

    # 1. 创建渲染器（不使用 with）
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    try:
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        renderer.scene.add_geometry("pcd", pcd, material)

        cam = renderer.scene.camera
        # set_projection：新版本 5 参数；旧版本 4 参数
        try:
            cam.set_projection(fov, width / height, 0.1, 1000.0,
                               o3d.visualization.rendering.Camera.FovType.Vertical)
        except TypeError:
            cam.set_projection(fov, width / height, 0.1, 1000.0)

        cam.look_at(center, center + eye_offset, up_vec)

        img = renderer.render_to_image()
        o3d.io.write_image(str(Path(out_path)), img)
        print(f"[✓] PNG 已保存：{Path(out_path).resolve()}")
    finally:
        # 2. 兼容性释放
        if hasattr(renderer, "release"):
            renderer.release()
        del renderer  # 触发析构

# ---------- 入口 ---------- #
def main():
    train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior.pkl"
    with open(train_input_path, 'rb') as f:
        data_infos = pickle.load(f)
    samples = data_infos['data_list']
    for sample in samples[52:54]:
        # TODO: 修改为你的文件路径
        bin_path = sample['lidar_path']
        out_png  = f"/data_volume_1/sjk_data/NuscenesGrounding/visualization/pc_{sample['sample_token']}.png"

        pts    = load_nuscenes_lidar_bin(bin_path)
        xyz    = pts[:, :3]
        colors = colorize_by_height(xyz)

        save_pointcloud_png(xyz, colors, out_png)
        print(f'save {sample["sample_token"]}')

if __name__ == "__main__":
    main()
