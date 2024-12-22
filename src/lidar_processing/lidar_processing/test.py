from calib_parse import *
from icp import *
from load_imu import *
from load_lidar import *
from main import *
from process_image import *
from process_lidar import *
from process_imu import *
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import cv2
import matplotlib.pyplot as plt


def test_pipeline():
    # Configuration
    timestep_1 = "0000000000"  # First timestep
    timestep_2 = "0000000010"  # Second timestep
    base_path = "/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI"

    # File paths
    lidar_file_1 = f"{base_path}/velodyne_points/data/{timestep_1}.bin"
    lidar_file_2 = f"{base_path}/velodyne_points/data/{timestep_2}.bin"
    image_file_1 = f"{base_path}/image_02/data/{timestep_1}.png"
    image_file_2 = f"{base_path}/image_02/data/{timestep_2}.png"
    oxts_file_1 = f"{base_path}/oxts/data/{timestep_1}.txt"
    oxts_file_2 = f"{base_path}/oxts/data/{timestep_2}.txt"

    # Load calibration data
    print("Loading calibration data...")
    velo_to_cam = parse_velo_to_cam()
    cam_to_cam = parse_cam_to_cam()
    imu_to_velo = parse_imu_to_velo()
    rectification = cam_to_cam["R_rect_02"]
    projection = cam_to_cam["P_rect_02"]

    # Load LiDAR data
    print("Loading LiDAR data...")
    points_t1 = load_lidar_data(lidar_file_1)
    points_t2 = load_lidar_data(lidar_file_2)
    print(f"Points T1: {points_t1.shape}, Points T2: {points_t2.shape}")

    # Load OXTS data
    print("Loading OXTS data...")
    transform_t1 = oxts_to_global_transformation(oxts_file_1)
    transform_t2 = oxts_to_global_transformation(oxts_file_2)
    # transform_t1 = transform_t1 @imu_to_velo
    # transform_t2 = transform_t2 @imu_to_velo
    # print(f"Transform 1 {transform_t1}")
    # print(f"Transform 2 {transform_t2}")

    ground_truth_transform = np.linalg.inv(transform_t1) @ transform_t2

    # Load CNN model for feature extraction
    cnn_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    # Enrich LiDAR points with semantic features
    print("Enriching LiDAR points with semantic features...")
    _, _, enriched_t1 = enrich_lidar_points(lidar_file_1, image_file_1, {"velo_to_cam": velo_to_cam, "rectification": rectification, "projection": projection}, cnn_model)
    _, _, enriched_t2 = enrich_lidar_points(lidar_file_2, image_file_2, {"velo_to_cam": velo_to_cam, "rectification": rectification, "projection": projection}, cnn_model)
    pca_points_1 = perform_pca(enriched_t1, n_components=128)
    pca_points_2 = perform_pca(enriched_t2, n_components=128)
    # print("ENRICHED", enriched_t1.shape)

    # Run ICP
    print("Running ICP...")
    # predicted_transform, _, _ = icp(enriched_t1, enriched_t2, init_pose=np.eye(4), max_iterations=100, tolerance=0.001, use_semantic_features=True, simple=True)
    predicted_transform, _, _ = icp(points_t1, points_t2, init_pose=np.eye(4), max_iterations=100, tolerance=0.001, use_semantic_features=False, simple=True)

    error_transform = ground_truth_transform - predicted_transform

    print(f"Transform between T1 and T2:\n{ground_truth_transform}")
    print(f"Predicted Transform:\n{predicted_transform}")
    print(f"Transform Error:\n{error_transform}")

    # Add homogeneous coordinates to points_t1
    points_t1_homogeneous = np.hstack((points_t1[:, :3], np.ones((points_t1.shape[0], 1))))

    # Apply the predicted transform
    aligned_points = (predicted_transform @ points_t1_homogeneous.T).T
    gt_translation_points = (ground_truth_transform @ points_t1_homogeneous.T).T
    # aligned_points= (np.linalg.inv(predicted_transform) @ points_t1_homogeneous.T).T

    # Drop the homogeneous coordinate
    aligned_points = aligned_points[:, :3]

    # Visualize ICP results
    visualize_icp_results(np.array([[0,0,0]]), points_t2[:, :3], aligned_points)
    # visualize_icp_results(pca_points_1[:, :3], pca_points_2[:, :3], np.array([[0,0,0]]))
    # visualize_icp_results(enriched_t1[:,:3], enriched_t2[:,:3], np.array([[0,0,0]]))

    # Compare trajectories
    print("Visualizing trajectories...")
    imu_trajectory = [np.zeros(3)]
    icp_trajectory = [np.zeros(3)]

    imu_position = (ground_truth_transform @ np.hstack((imu_trajectory[-1], 1)).T)[:3]
    icp_position = (predicted_transform @ np.hstack((icp_trajectory[-1], 1)).T)[:3]
    imu_trajectory.append(imu_position)
    icp_trajectory.append(icp_position)

    imu_trajectory = np.array(imu_trajectory)
    icp_trajectory = np.array(icp_trajectory)

    plt.figure(figsize=(10, 7))
    plt.plot(imu_trajectory[:, 0], imu_trajectory[:, 1], label="IMU Trajectory (GT)", color="red", alpha= 0.7)
    plt.plot(icp_trajectory[:, 0], icp_trajectory[:, 1], label="ICP Trajectory", color="blue", alpha=0.7)
    plt.title("Trajectories: IMU (Ground Truth) vs ICP")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

    R_pred, t_pred = predicted_transform[:3, :3], predicted_transform[:3, 3]
    R_gt, t_gt = ground_truth_transform[:3, :3], ground_truth_transform[:3, 3]

    rotation_error = np.linalg.norm(R_pred - R_gt)
    translation_error = np.linalg.norm(t_pred - t_gt)
    print(f"Rotation Error: {rotation_error}")
    print(f"Translation Error: {translation_error}")


if __name__ == "__main__":
    test_pipeline()
