import numpy as np

def project_lidar_to_image(lidar_points, velo_to_cam, rectification, projection):
    """
    Projects LiDAR points into the image plane.
    :param lidar_points: (N, 4) LiDAR points in homogeneous coordinates (x, y, z, 1).
    :param velo_to_cam: (4, 4) Transformation from Velodyne to camera.
    :param rectification: (3, 3) Rectification matrix.
    :param projection: (3, 4) Projection matrix.
    :return: (N, 2) Projected 2D pixel coordinates, mask of valid points.
    """
    # Transform LiDAR points to the camera frame
    lidar_camera_frame = (velo_to_cam @ lidar_points.T).T

    # Create a mask for points in front of the camera
    valid_mask = lidar_camera_frame[:, 2] > 0

    # Apply the mask to filter points
    lidar_camera_frame = lidar_camera_frame[valid_mask]

    # Apply rectification
    rectified_points = (rectification @ lidar_camera_frame[:, :3].T).T

    # Project points into the image plane
    projected_points = (projection @ np.hstack((rectified_points, np.ones((rectified_points.shape[0], 1)))).T).T

    # Normalize homogeneous coordinates
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    # Extract pixel coordinates
    pixel_coords = projected_points[:, :2]

    # Return pixel coordinates and a mask matching the original input points
    final_mask = np.zeros(lidar_points.shape[0], dtype=bool)
    final_mask[np.where(valid_mask)] = True
    return pixel_coords, final_mask
