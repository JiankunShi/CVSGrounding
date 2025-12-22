import numpy as np
from mayavi import mlab
import pickle

if __name__ == '__main__':
    train_input_path = "/data_volume_1/sjk_data/NuscenesGrounding/mmdet_all_train_data_map_caption_IOU03_yaw_egoCentre_multiView30_pro0P5_concatDesc_lidar2CamIns2_addBehavior.pkl"
    with open(train_input_path, 'rb') as f:
        data_infos = pickle.load(f)
    samples = data_infos['data_list']
    for sample in samples[52:54]:
        bin_file = sample['lidar_path']
        pointcloud = np.fromfile(bin_file, dtype=np.float32, count=-1).reshape([-1, 4])

        x = pointcloud[:, 0]  # x position of point
        y = pointcloud[:, 1]  # y position of point
        z = pointcloud[:, 2]  # z position of point
        r = pointcloud[:, 3]  # reflectance value of point

        d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

        vals = 'height'
        if vals == "height":
            col = z
        else:
            col = d

        fig = mlab.figure(bgcolor=(0, 0, 0), size=(700, 500))
        mlab.points3d(x, y, z,
                      d,  # Values used for Color
                      mode="point",
                      colormap='spectral',  # 'bone', 'copper', 'gnuplot', 'spectral', 'summer'
                      # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                      figure=fig)
        mlab.show()
