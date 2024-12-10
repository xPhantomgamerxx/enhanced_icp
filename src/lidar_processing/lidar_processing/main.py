from load_lidar import *
from calib_parse import *
from process_lidar import *
from process_image import *
from load_imu import *
import numpy as np
import cv2
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.applications import MobileNetV2
from sklearn.decomposition import PCA


def calib():
    cam_to_cam_data = parse_cam_to_cam("/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/calib/calib_cam_to_cam.txt")
    velo_to_cam_matrix = parse_velo_to_cam("/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/calib/calib_velo_to_cam.txt")

    return cam_to_cam_data, velo_to_cam_matrix
    



def main():
    # File paths to KITTI calibration files
    timestep = "000"
    debug = False
    N = 128
    cnn_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Parse the calibration files and Extract matrices
    cam_to_cam_data , velo_to_cam_matrix = calib()
    rectification_matrix = cam_to_cam_data["R_rect_02"]  # Adjust based on camera index
    projection_matrix = cam_to_cam_data["P_rect_02"]     # Camera 2 projection matrix

    
    pixel_coords, valid_mask, enriched_points = enrich_lidar_points_with_semantics_dynamic(
        lidar_file="/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/velodyne_points/data/0000000"+timestep+".bin",
        image_file="/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/image_02/data/0000000"+timestep+".png",
        calib_data={
            "velo_to_cam": velo_to_cam_matrix,
            "rectification": rectification_matrix,
            "projection": projection_matrix
        },
        cnn_model=cnn_model,
        verbose=debug
    )

    if debug: 
        visualize_projected_points(
            image_path="/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/image_02/data/0000000"+timestep+".png",
            pixel_coords=pixel_coords,
            valid_mask=valid_mask
        )
        print(enriched_points.shape)

    # Keep the first N semantic features
    pca = PCA(n_components=N)  # Reduce to N dimensions
    reduced_features = pca.fit_transform(enriched_points[:, 3:])
    points_for_icp = np.hstack((enriched_points[:, :3], reduced_features))
    if debug: print("Reduced points for ICP shape:", points_for_icp.shape)

    oxts_data = load_oxts("/Users/toby/My Stuf/Sweden Uni Stuf/Exchange Semester/Autonomous Vehicles/Project/enhanced_icp/datasets/KITTI/oxts/data/0000000"+timestep+".txt")
    init_pose = pose_to_transform(oxts_data)

    

if __name__ == "__main__":
    main()