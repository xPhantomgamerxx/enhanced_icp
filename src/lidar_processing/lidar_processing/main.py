from load_lidar import *
from calib_parse import *
from process_lidar import *
from process_image import *
from load_imu import *
from icp import *
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


def load_calibration_data(debug = False):
    """
    Load camera and LiDAR calibration data.

    Input:
        debug: Bool to enable debug print lines
    """
    cam_to_cam_data = parse_cam_to_cam()
    velo_to_cam_matrix = parse_velo_to_cam()
    imu_to_velo_matrix = parse_imu_to_velo()
    if debug: print("Calibration data Loaded")

    return {
        "cam_to_cam_data": cam_to_cam_data,
        "velo_to_cam_matrix": velo_to_cam_matrix,
        "imu_to_velo_matrix": imu_to_velo_matrix,
    }


def load_timestep_data(timestep, debug = False):
    """
    Load LiDAR, image, and OXTS data for a specific timestep.

    Input:
        timestep: The current timestep of image that is being processed 
        debug: Bool to enable debug print lines
    """
    base_path = "/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI"
    lidar_file = f"{base_path}/velodyne_points/data/0000000{timestep}.bin"
    image_file = f"{base_path}/image_02/data/0000000{timestep}.png"
    oxts_file = f"{base_path}/oxts/data/0000000{timestep}.txt"
    oxts_data = load_oxts(oxts_file)
    if debug: print("Data for Timestep", timestep, "loaded Successfully")
    return lidar_file, image_file, oxts_data


def perform_pca(enriched_points, n_components=128, debug=False):
    """
    Perform PCA on enriched points to reduce semantic feature dimensions.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(enriched_points[:, 3:])
    points_for_icp = np.hstack((enriched_points[:, :3], reduced_features))
    if debug:
        print(f"Reduced points for ICP shape: {points_for_icp.shape}")
        print("PCA Successfull")
    return points_for_icp


def process_timestep(timestep, cnn_model, calibration_data, n_components=128, debug=False):
    """
    Process a single timestep: enrich points with semantics and perform PCA.
    """

    # Extract calibration data
    cam_to_cam_data = calibration_data["cam_to_cam_data"]
    velo_to_cam_matrix = calibration_data["velo_to_cam_matrix"]
    rectification_matrix = cam_to_cam_data["R_rect_02"]
    projection_matrix = cam_to_cam_data["P_rect_02"]

    # Load timestep-specific data
    lidar_file, image_file, oxts_data = load_timestep_data(timestep, debug)    

    # Compute the initial pose from OXTS data
    init_pose = pose_to_transform(oxts_data)

    # Enrich LiDAR points with semantic features
    pixel_coords, valid_mask, enriched_points = enrich_lidar_points_with_semantics_dynamic(
        lidar_file,
        image_file,
        calib_data={
            "velo_to_cam": velo_to_cam_matrix,
            "rectification": rectification_matrix,
            "projection": projection_matrix,
        },
        cnn_model=cnn_model,
        verbose=debug,
    )

    if debug: visualize_semantic_features(enriched_points[:, 3:], timestep)

    scaler = StandardScaler()
    enriched_points[:, 3:] = scaler.fit_transform(enriched_points[:, 3:])

    # Reduce dimensions using PCA
    # points_for_icp = perform_pca(enriched_points, n_components)

    # Optionally visualize projected points
    if debug:
        visualize_projected_points(image_file, pixel_coords, valid_mask)
        print(f"Enriched points shape: {enriched_points.shape}")
        print(f"Initial Pose Matrix:\n{init_pose}")
        print("Timestep Processed Successfully")
    return enriched_points, init_pose
    return points_for_icp, init_pose


def icp_with_error_analysis(start_timestep, end_timestep, cnn_model, calibration_data, n_components=128, use_semantic_features = True, spatial_weight = 1.0, semantic_weight = 0.1, debug=False):
    """
    Perform ICP over multiple timesteps and compute errors compared to OXTS ground truth.
    """
    results = []
    accumulated_transform = np.eye(4)  # Start with identity matrix for global pose

    for t in range(int(start_timestep), int(end_timestep)):
        start_time = time.time()
        print("Start timestep ", t)

        # Load and process data for current and next timestep
        points_t1, transform_t1 = process_timestep(str(t).zfill(3), cnn_model, calibration_data, n_components, debug)
        points_t2, transform_t2 = process_timestep(str(t + 1).zfill(3), cnn_model,calibration_data, n_components, debug)

        # Compute ground truth relative transform from OXTS
        ground_truth_transform = np.linalg.inv((transform_t1)) @ (transform_t2)

        # print(f"Ground Truth Transform for timestep {t} -> {t+1}:\n{ground_truth_transform}")

        # visualize_icp_results(points_t1[:, :3], points_t2[:, :3], points_t1[:, :3])  # The third argument is just a placeholder for unaligned source

        # Perform ICP to get predicted relative transform
        predicted_transform, distances, iterations = icp(
            points_t1, points_t2, 
            init_pose=np.eye(4), max_iterations=20, tolerance=0.001, 
            use_semantic_features=use_semantic_features, spatial_weight=spatial_weight, semantic_weight=semantic_weight, simple=False
        )

        # Accumulate predicted transform to compute global pose
        accumulated_transform = np.dot(predicted_transform, accumulated_transform)

        # Compute error metrics
        transform_error = np.linalg.norm(predicted_transform - ground_truth_transform)
        mean_error = np.mean(distances)

        # Store results
        results.append({
            'timestep': t,
            'predicted_transform': predicted_transform,
            'ground_truth_transform': ground_truth_transform,
            'transform_error': transform_error,
            'mean_distance_error': mean_error,
            'iterations': iterations
        })

        end_time=time.time()
        elapsed_time = end_time - start_time
        print(f"Timestep {t} -> {t + 1} took {elapsed_time:.4f} seconds")

        if debug:
            print(f"Timestep {t} -> {t + 1}:")
            print(f"Predicted Transform:\n{predicted_transform}")
            print(f"Ground Truth Transform:\n{ground_truth_transform}")
            print(f"Transform Error: {transform_error}")
            print(f"Mean Distance Error: {mean_error}")
            print(f"ICP Iterations: {iterations}")
        print("End Timestep ", t)
        
    return results


def vizualize_results(results):
    # Extract data for plotting
    timesteps = [result['timestep'] for result in results]
    transform_errors = [result['transform_error'] for result in results]
    mean_distance_errors = [result['mean_distance_error'] for result in results]

    # Plot transform errors
    plt.figure(figsize=(10, 5))
    plt.bar(timesteps, transform_errors, color='blue', alpha=0.7, label='Transform Error')
    plt.plot(timesteps, transform_errors, color='blue', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Error')
    plt.title('Transform Error Across Timesteps')
    plt.xticks(timesteps)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # Plot mean distance errors
    plt.figure(figsize=(10, 5))
    plt.bar(timesteps, mean_distance_errors, color='orange', alpha=0.7, label='Mean Distance Error')
    plt.plot(timesteps, mean_distance_errors, color='orange', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Error')
    plt.title('Mean Distance Error Across Timesteps')
    plt.xticks(timesteps)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def main():
    # Configuration
    start_timestep = "000"
    end_timestep = "010"  # Adjust as needed
    n_components = 32
    use_semantic_features = False
    spatial_weight = 1.0
    semantic_weight = 0.1
    debug = False
    full_start = time.time()

    # Load the pretrained CNN model
    cnn_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    calibration_data = load_calibration_data(debug)

    # Perform ICP and analyze error
    results = icp_with_error_analysis(start_timestep, end_timestep, cnn_model, calibration_data, n_components, use_semantic_features, spatial_weight, semantic_weight, debug)

    # Visualize errors as bar charts
    # vizualize_results(results)

    # Visualize point clouds and trajectories
    imu_trajectory = [np.array([0, 0, 0])]
    icp_trajectory = [np.array([0, 0, 0])]

    for result in results:
        ground_truth_transform = result['ground_truth_transform']
        predicted_transform = result['predicted_transform']

        # Extract trajectories by applying transforms
        imu_position = (ground_truth_transform @ np.hstack((imu_trajectory[-1], 1)).T)[:3]
        icp_position = (predicted_transform @ np.hstack((icp_trajectory[-1], 1)).T)[:3]
        imu_trajectory.append(imu_position)
        icp_trajectory.append(icp_position)

    imu_trajectory = np.array(imu_trajectory)
    icp_trajectory = np.array(icp_trajectory)
    
    full_end = time.time()
    elapsed_time = full_end - full_start
    print(f"Full run took {elapsed_time:.4f} seconds")

    # Visualize trajectories
    visualize_trajectories(imu_trajectory, icp_trajectory)

    # Optionally visualize the last transform for point clouds
    # points_t1, _ = process_timestep(start_timestep, cnn_model, n_components, debug)
    # points_t2, _ = process_timestep(end_timestep, cnn_model, n_components, debug)

    # visualize_transforms(points_t1[:, :3], points_t2[:, :3], 
    #                      results[0]['ground_truth_transform'], 
    #                      results[-1]['predicted_transform'])


if __name__ == "__main__":
    main()