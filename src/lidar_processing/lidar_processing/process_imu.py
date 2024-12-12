import numpy as np
from pyproj import Proj, transform, Transformer


def oxts_to_transformation(file_path):
    """
    Compute the transformation matrix from an OXTS data file.
    
    Args:
        file_path (str): Path to the OXTS data file containing one row of IMU data.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    # Define WGS84 ellipsoid
    wgs84 = Proj(proj="latlong", datum="WGS84")
    enu_proj = Proj(proj="utm", zone=32, datum="WGS84")  # Modify UTM zone based on your location
    
    def euler_to_rot_matrix(roll, pitch, yaw):
        """Compute rotation matrix from Euler angles (roll, pitch, yaw)."""
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x

    # Read the OXTS data
    with open(file_path, "r") as file:
        data = file.readline().strip().split()
    
    # Extract relevant data fields
    lat, lon, alt = float(data[0]), float(data[1]), float(data[2])
    roll, pitch, yaw = float(data[3]), float(data[4]), float(data[5])
    
    # Reference point for ENU conversion (set this as needed)
    ref_lat, ref_lon, ref_alt = lat, lon, alt  # Replace with a global reference if needed
    
    # Convert latitude, longitude, altitude to ENU coordinates
    ecef_x, ecef_y, ecef_z = transform(wgs84, enu_proj, lon, lat, alt)
    ref_ecef_x, ref_ecef_y, ref_ecef_z = transform(wgs84, enu_proj, ref_lon, ref_lat, ref_alt)
    t = np.array([ecef_x - ref_ecef_x, ecef_y - ref_ecef_y, ecef_z - ref_ecef_z])
    
    # Compute rotation matrix
    R = euler_to_rot_matrix(roll, pitch, yaw)
    
    # Construct the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def oxts_to_global_transformation(file_path):
    """
    Compute the global transformation matrix from an OXTS data file.
    
    Args:
        file_path (str): Path to the OXTS data file containing one row of IMU data.

    Returns:
        np.ndarray: 4x4 transformation matrix in global (ECEF) frame.
    """
    def euler_to_rot_matrix(roll, pitch, yaw):
        """Compute rotation matrix from Euler angles (roll, pitch, yaw)."""
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x

    # Read the OXTS data
    with open(file_path, "r") as file:
        data = file.readline().strip().split()
    
    # Extract relevant data fields
    lat, lon, alt = float(data[0]), float(data[1]), float(data[2])
    roll, pitch, yaw = float(data[3]), float(data[4]), float(data[5])
    
    # Convert latitude, longitude, altitude to ECEF coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978")  # WGS84 to ECEF
    ecef_x, ecef_y, ecef_z = transformer.transform(lat, lon, alt)
    t = np.array([ecef_x, ecef_y, ecef_z])
    
    # Compute rotation matrix
    R = euler_to_rot_matrix(roll, pitch, yaw)
    
    # Construct the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T
