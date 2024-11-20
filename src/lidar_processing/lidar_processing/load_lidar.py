import numpy as np
import open3d as o3d

def load_lidar_data(file_path="/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/velodyne_points/data/0000000000.bin"):
    # Each point is (x, y, z, intensity) in float32
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def visualize_lidar(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z
    o3d.visualization.draw_geometries([cloud])

# Example usage
def main():
    lidar_file = "/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/velodyne_points/data/0000000000.bin"
    lidar_points = load_lidar_data(lidar_file)
    print(f"Loaded {lidar_points.shape[0]} points")
    visualize_lidar(lidar_points)

if __name__ == "__main__":
    main()