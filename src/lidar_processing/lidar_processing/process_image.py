import numpy as np
from load_lidar import *
from process_lidar import *
from load_lidar import *
from calib_parse import *
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.applications import MobileNetV2

def generate_feature_map_dynamic(image, cnn_model):
    """
    Processes the whole image through a CNN to generate a dense feature map, 
    handling non-square input images by resizing while preserving aspect ratio.
    """
    # CNN expected input size
    input_height, input_width = cnn_model.input.shape[1:3]

    # Resize the image to the CNN input size while preserving aspect ratio
    h, w, _ = image.shape
    scale_h = input_height / h
    scale_w = input_width / w
    scale = min(scale_h, scale_w)  # Preserve aspect ratio

    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Pad to match the CNN input size
    padded_image = np.zeros((input_height, input_width, 3), dtype=np.float32)
    pad_h, pad_w = (input_height - new_h) // 2, (input_width - new_w) // 2
    padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image

    # Preprocess the image for the CNN
    image_normalized = preprocess_input(np.expand_dims(padded_image, axis=0))  # Add batch dimension and normalize

    # Run the CNN to generate the feature map
    feature_map = cnn_model.predict(image_normalized, batch_size=1)[0]  # Remove batch dimension
    return feature_map, scale, pad_h, pad_w, new_h, new_w

def map_features_to_lidar_dynamic(lidar_pixel_coords, feature_map, scale, pad_h, pad_w, original_image_shape):
    """
    Maps features from the CNN feature map to the LiDAR points using the scaled pixel coordinates.

    :param lidar_pixel_coords: np.ndarray
        The 2D coordinates of LiDAR points projected onto the original image (N x 2).
    :param feature_map: np.ndarray
        The dense feature map (H_f x W_f x F).
    :param scale: float
        The scale used to resize the image to the CNN's input size.
    :param pad_h: int
        Vertical padding applied to the resized image.
    :param pad_w: int
        Horizontal padding applied to the resized image.
    :param original_image_shape: tuple
        The shape of the original image (H, W, C).
    :return: np.ndarray
        Enriched LiDAR features (N x F).
    """
    # Original and feature map dimensions
    h_orig, w_orig = original_image_shape[:2]
    h_map, w_map, f_dim = feature_map.shape

    # Placeholder for enriched LiDAR features
    num_points = lidar_pixel_coords.shape[0]
    lidar_features = np.zeros((num_points, f_dim), dtype=np.float32)

    for i, (u, v) in enumerate(lidar_pixel_coords):
        # Scale the coordinates to match the resized image
        u_resized = u * scale + pad_w
        v_resized = v * scale + pad_h

        # Scale further to the feature map resolution
        u_scaled = int(u_resized * (w_map / w_orig))
        v_scaled = int(v_resized * (h_map / h_orig))

        # Skip invalid points
        if not (0 <= u_scaled < w_map and 0 <= v_scaled < h_map):
            continue

        # Assign the feature vector from the feature map
        lidar_features[i] = feature_map[v_scaled, u_scaled, :]

    return lidar_features

def enrich_lidar_points_with_semantics_dynamic(lidar_file, image_file, calib_data, cnn_model, verbose = False):
    """
    Combines LiDAR points with semantic features from the image using whole image processing,
    dynamically handling non-square input images.

    :param lidar_file: str
        Path to the LiDAR data file.
    :param image_file: str
        Path to the RGB image file.
    :param calib_data: dict
        Calibration data for LiDAR-to-image projection.
    :param cnn_model: keras.Model
        Pretrained CNN model for feature extraction.
    :return: np.ndarray
        Enriched LiDAR points (N x (3 + F)).
    """
    # Load LiDAR points
    lidar_points = load_lidar_data(lidar_file)
    lidar_points_homogeneous = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))

    # Project LiDAR points into the image plane
    pixel_coords, valid_mask = project_lidar_to_image(
        lidar_points_homogeneous,
        calib_data['velo_to_cam'],
        calib_data['rectification'],
        calib_data['projection']
    )

    if verbose:
        print("Pixel coordinates shape:", pixel_coords.shape)
        print("Valid mask shape:", valid_mask.shape)
        print("Number of valid points:", np.sum(valid_mask))

    # Generate the CNN feature map
    image = cv2.imread(image_file)
    feature_map, scale, pad_h, pad_w, new_h, new_w = generate_feature_map_dynamic(image, cnn_model)

    # Map features to valid LiDAR points
    lidar_features = map_features_to_lidar_dynamic(
        pixel_coords[valid_mask], feature_map, scale, pad_h, pad_w, image.shape
    )

    # Enrich valid LiDAR points
    enriched_lidar_points = np.hstack((lidar_points[valid_mask, :3], lidar_features))

    return pixel_coords, valid_mask, enriched_lidar_points
