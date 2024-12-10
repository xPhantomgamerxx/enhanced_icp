import numpy as np
import cv2

def project_lidar_to_image(lidar_points, velo_to_cam, rectification, projection, verbose = False):
    """
    Projects LiDAR points into the image plane and filters points within the camera's FOV.

    :param lidar_points: (N, 4) LiDAR points in homogeneous coordinates (x, y, z, 1).
    :param velo_to_cam: (4, 4) Transformation matrix from LiDAR to camera.
    :param rectification: (3, 3) Rectification matrix.
    :param projection: (3, 4) Projection matrix.
    :return: (N, 2) Projected 2D pixel coordinates, mask of valid points.
    """

    # Transform LiDAR points to the camera frame
    lidar_camera_frame = (velo_to_cam @ lidar_points.T).T

    # Filter points with positive depth (in front of the camera)
    valid_mask = lidar_camera_frame[:, 2] > 0

    if verbose:     
        print("Original LiDAR points:", lidar_points.shape[0])
        print("Number of valid LiDAR points:", np.sum(valid_mask))
        print("lidar_camera_frame shape:", lidar_camera_frame.shape)
        print("valid_mask shape:", valid_mask.shape)

    # Apply rectification and projection
    rectified_points = (rectification @ lidar_camera_frame[:, :3].T).T
    projected_points = (projection @ np.hstack((rectified_points, np.ones((rectified_points.shape[0], 1)))).T).T

    # Normalize homogeneous coordinates to pixel coordinates
    projected_points[:, 0] /= projected_points[:, 2]  # u = x / z
    projected_points[:, 1] /= projected_points[:, 2]  # v = y / z

    # Extract pixel coordinates
    pixel_coords = projected_points[:, :2]

    # Camera image dimensions (update with actual dimensions)
    image_width, image_height = 1242, 375

    # FOV filter: Ensure points are within the image plane
    fov_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_width) & \
               (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_height)

    # Combine depth and FOV masks
    valid_mask &= fov_mask

    # Return filtered pixel coordinates and valid mask
    return pixel_coords, valid_mask

def visualize_projected_points(image_path, pixel_coords, valid_mask):
    """
    Visualize valid LiDAR points projected onto the image.

    :param image_path: str
        Path to the image file.
    :param pixel_coords: np.ndarray
        2D pixel coordinates of all LiDAR points.
    :param valid_mask: np.ndarray
        Boolean mask indicating valid points.

    :output: displays image with projected lidar points on it
    """
    import matplotlib.pyplot as plt

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Filter valid pixel coordinates
    valid_pixel_coords = pixel_coords[valid_mask]

    # Plot the image with overlaid points
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(valid_pixel_coords[:, 0], valid_pixel_coords[:, 1], s=0.5, c='red', label="LiDAR Points")
    plt.title("LiDAR Points Projected Onto Image")
    plt.legend()
    plt.show()
