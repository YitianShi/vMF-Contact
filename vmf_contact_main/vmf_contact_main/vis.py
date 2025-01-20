import numpy as np
import open3d as o3d
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
file = f"{current_dir}/pcd.npy"
data = np.load(file)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
o3d.visualization.draw_geometries([pcd])
