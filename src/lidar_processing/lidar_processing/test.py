from load_lidar import *
from calib_parse import *
from process_lidar import *
import numpy as np
import cv2

def main():
    # File paths
    img_no = "001"
    lidar_file = "/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/velodyne_points/data/0000000"+img_no+".bin"
    lidar_file = "/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/velodyne_points/data/0000000"+img_no+".bin"
    image_file = "/home/user/lidar_camera_slam_ws/datasets/KITTI/2011_09_26/drive_long/image_02/data/0000000"+img_no+".png"
    image_file = "/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/image_02/data/0000000"+img_no+".png"

    # Parse calibration files
    cam_to_cam_data = parse_cam_to_cam("/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/calib/calib_cam_to_cam.txt")
    T_velo_to_cam = parse_velo_to_cam("/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/calib/calib_velo_to_cam.txt")

    # Access parsed data
    K_02 = cam_to_cam_data["K_02"]
    P_rect_02 = cam_to_cam_data["P_rect_02"]
    R_rect_00 = cam_to_cam_data["R_rect_00"]

    # Load LiDAR points
    lidar_points = load_lidar_data(lidar_file)

    # Convert LiDAR points to homogeneous coordinates
    lidar_points_homogeneous = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))

    # Project LiDAR points to image plane
    pixel_coords, valid_mask = project_lidar_to_image(
        lidar_points_homogeneous, T_velo_to_cam, R_rect_00, P_rect_02
    )

    # Load image
    image = cv2.imread(image_file)

    # Visualize projected points on image
    visualize_projected_points(image, pixel_coords)

def visualize_projected_points(image, pixel_coords):
    """Visualize projected LiDAR points on the image."""
    import matplotlib.pyplot as plt

    # Convert image to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot image
    plt.imshow(image_rgb)

    # Iterate over filtered pixel coordinates (no need for valid_mask here)
    for pixel in pixel_coords:
        u, v = int(pixel[0]), int(pixel[1])
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            plt.plot(u, v, 'r.', markersize=0.5)

    # plt.title("LiDAR Points Projected onto Image")
    plt.show()



if __name__ == "__main__":
    main()
