import numpy as np
import open3d as o3d

def load_lidar_data(file_path="/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/velodyne_points/data/0000000000.bin"):
    """
    Load lidar data from a single-line file.
    
    :param file_path: str
        Path to the lidar file.
    :return: numpy array
        Array containing the parsed lidar data.
    """
    # Each point is (x, y, z, intensity) in float32
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def visualize_lidar(points):
    """
    Vizualize the lidar points
    
    :param points: np array
        Array of lidar Points
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z
    o3d.visualization.draw_geometries([cloud])

# Example usage
def main():
    img_no = "001"
    lidar_file = "/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/velodyne_points/data/0000000000.bin"
    lidar_file = "/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/velodyne_points/data/0000000"+img_no+".bin"

    lidar_points = load_lidar_data(lidar_file)
    print(f"Loaded {lidar_points.shape[0]} points")
    visualize_lidar(lidar_points)

if __name__ == "__main__":
    main()