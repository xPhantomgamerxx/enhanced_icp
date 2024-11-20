import numpy as np
from load_lidar import *

def parse_cam_to_cam(file_path="/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/calib/calib_cam_to_cam.txt"):
    """Parses the cam_to_cam calibration file from a file path."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = np.array([float(x) for x in value.strip().split()])
                # Reshape matrices
                if key.startswith("K_") or key.startswith("R_") or key.startswith("P_rect_"):
                    data[key] = value.reshape(3, 3 if "K_" in key or "R_" in key else 4)
                elif key.startswith("S_") or key.startswith("S_rect_"):
                    data[key] = value.reshape(2)
                else:
                    data[key] = value
    return data

def parse_imu_to_velo(file_path="/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/calib/calib_imu_to_velo.txt"):
    """Parses the imu_to_velo calibration file from a file path."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                data[key] = np.array([float(x) for x in value.strip().split()])
    R = data["R"].reshape(3, 3)
    T = data["T"]
    # Build the 4x4 transformation matrix
    imu_to_velo = np.eye(4)
    imu_to_velo[:3, :3] = R
    imu_to_velo[:3, 3] = T
    return imu_to_velo

def parse_velo_to_cam(file_path="/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/calib/calib_velo_to_cam.txt"):
    """Parses the velo_to_cam calibration file from a file path."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                data[key] = np.array([float(x) for x in value.strip().split()])
    R = data["R"].reshape(3, 3)
    T = data["T"]
    # Build the 4x4 transformation matrix
    velo_to_cam = np.eye(4)
    velo_to_cam[:3, :3] = R
    velo_to_cam[:3, 3] = T
    return velo_to_cam

def main():
    cam_to_cam_data = parse_cam_to_cam()
    imu_to_velo_matrix = parse_imu_to_velo()
    velo_to_cam_matrix = parse_velo_to_cam()

    # Access parsed data        
    K_02 = cam_to_cam_data["K_02"]
    P_rect_02 = cam_to_cam_data["P_rect_02"]
    print("K_02:\n", K_02)
    print("P_rect_02:\n", P_rect_02)
    print("IMU to Velodyne:\n", imu_to_velo_matrix)
    print("Velodyne to Camera:\n", velo_to_cam_matrix)

if __name__ == "__main__":
    main()