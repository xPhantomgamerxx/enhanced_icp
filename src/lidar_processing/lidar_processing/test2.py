import numpy as np
from calib_parse import *
from icp import *
from load_imu import *
from load_lidar import *
from main import *
from process_image import *
from process_lidar import *
from process_imu import *
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2


def icp_pipeline(start_frame, end_frame, base_path, max_iterations=100, tolerance=0.001, verbose=False):
    """
    Perform ICP over a sequence of frames.

    :param start_frame: int
        Starting frame index (inclusive).
    :param end_frame: int
        Ending frame index (exclusive).
    :param base_path: str
        Base path for the dataset.
    :param max_iterations: int
        Maximum number of iterations for ICP.
    :param tolerance: float
        Tolerance for ICP convergence.
    :param verbose: bool
        If True, print debug information.
    """
    # Calibration data
    print("Loading calibration data...")
    velo_to_cam = parse_velo_to_cam()
    cam_to_cam = parse_cam_to_cam()
    rectification = cam_to_cam["R_rect_02"]
    projection = cam_to_cam["P_rect_02"]
    cnn_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    global_trajectory = [np.eye(4)]
    imu_trajectory = [np.eye(4)]
    cumulative_transform = np.eye(4)

    oxts_file_1 = f"{base_path}/oxts/data/{start_frame}.txt"
    initial_transform = oxts_to_global_transformation(oxts_file_1)
    

    for frame in range(int(start_frame), int(end_frame)):
        timestep_1 = f"{frame:010d}"
        timestep_2 = f"{frame + 1:010d}"

        # File paths
        lidar_file_1 = f"{base_path}/velodyne_points/data/{timestep_1}.bin"
        lidar_file_2 = f"{base_path}/velodyne_points/data/{timestep_2}.bin"
        image_file_1 = f"{base_path}/image_02/data/{timestep_1}.png"
        image_file_2 = f"{base_path}/image_02/data/{timestep_2}.png"
        oxts_file_1 = f"{base_path}/oxts/data/{timestep_1}.txt"
        oxts_file_2 = f"{base_path}/oxts/data/{timestep_2}.txt"

        # Load data
        points_t1 = load_lidar_data(lidar_file_1)
        points_t2 = load_lidar_data(lidar_file_2)


        # Load ground truth transforms
        transform_t1 = oxts_to_global_transformation(oxts_file_1)
        transform_t2 = oxts_to_global_transformation(oxts_file_2)
        ground_truth_transform = np.linalg.inv(transform_t1) @ transform_t2

        # Enrich points with semantic features
        _, _, enriched_t1 = enrich_lidar_points(lidar_file_1, image_file_1,{"velo_to_cam": velo_to_cam, "rectification": rectification, "projection": projection},cnn_model )
        _, _, enriched_t2 = enrich_lidar_points(lidar_file_2, image_file_2,{"velo_to_cam": velo_to_cam, "rectification": rectification, "projection": projection},cnn_model)

        # Run ICP
        print(f"Running ICP for frames {frame} -> {frame + 1}...")
        predicted_transform, _, _ = icp(points_t1, points_t2, init_pose=np.eye(4),max_iterations=max_iterations, tolerance=tolerance,use_semantic_features=False, simple=True)
        cumulative_transform = cumulative_transform @ predicted_transform

        # Accumulate transform for trajectory
        global_trajectory.append(global_trajectory[-1] @ predicted_transform)
        imu_trajectory.append(imu_trajectory[-1] @ ground_truth_transform)

        # Debug output
        if verbose:
            print(f"Ground Truth Transform:\n{ground_truth_transform}")
            print(f"Predicted Transform:\n{predicted_transform}")

    # Visualize trajectories
    visualize_trajectories(global_trajectory, imu_trajectory)
    oxts_file_2 = f"{base_path}/oxts/data/{end_frame}.txt"
    end_transform = oxts_to_global_transformation(oxts_file_2)
    print("Final Transform: ", end_transform)

    final_ground_truth_transform = np.linalg.inv(initial_transform) @ end_transform
    points_t1 = load_lidar_data(f"{base_path}/velodyne_points/data/{start_frame}.bin")
    points_t2 = load_lidar_data(f"{base_path}/velodyne_points/data/{end_frame}.bin")
    points_t1_homogeneous = np.hstack((points_t1[:, :3], np.ones((points_t1.shape[0], 1))))


    # Apply the predicted transform
    aligned_points = (cumulative_transform @ points_t1_homogeneous.T).T
    gt_translation_points = (ground_truth_transform @ points_t1_homogeneous.T).T
    # aligned_points= (np.linalg.inv(predicted_transform) @ points_t1_homogeneous.T).T

    # Drop the homogeneous coordinate
    aligned_points = aligned_points[:, :3]

    # Visualize ICP results
    visualize_icp_results(points_t1[:, :3], points_t2[:, :3], aligned_points)


def visualize_trajectories(global_trajectory, imu_trajectory):
    """
    Visualize global and IMU trajectories.

    :param global_trajectory: list of 4x4 np.array
        Global trajectory estimated by ICP.
    :param imu_trajectory: list of 4x4 np.array
        Ground truth trajectory from IMU data.
    """
    global_positions = [np.array([0, 0, 0])]
    imu_positions = [np.array([0, 0, 0])]

    for transform in global_trajectory[1:]:
        global_positions.append(transform[:3, 3])
    for transform in imu_trajectory[1:]:
        imu_positions.append(transform[:3, 3])

    global_positions = np.array(global_positions)
    imu_positions = np.array(imu_positions)

    plt.figure(figsize=(10, 7))
    plt.plot(global_positions[:, 0], global_positions[:, 1], label="ICP Trajectory", color="blue", alpha=0.7)
    plt.plot(imu_positions[:, 0], imu_positions[:, 1], label="IMU Trajectory (GT)", color="red", alpha=0.7)
    plt.title("Trajectories: ICP vs IMU")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage
    base_path = "/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI"
    timestep_1 = "0000000000"  # First timestep
    timestep_2 = "0000000020"  # Second timestep
    icp_pipeline(timestep_1, timestep_2, base_path, verbose=False)
