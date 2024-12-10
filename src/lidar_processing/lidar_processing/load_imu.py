import numpy as np
from scipy.spatial.transform import Rotation as R

def load_oxts(oxts_file):
    """
    Parse KITTI oxts file to extract pose information.
    
    :param oxts_file: str
        Path to the oxts file.
    :return: list of dict
        A list of dictionaries containing parsed pose information.
    """
    oxts_data = []
    with open(oxts_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            oxts_data.append({
                'lat': values[0],  # Latitude
                'lon': values[1],  # Longitude
                'alt': values[2],  # Altitude
                'roll': values[3],  # Roll (radians)
                'pitch': values[4],  # Pitch (radians)
                'yaw': values[5],  # Yaw (radians)
                'vx': values[6],  # Velocity x
                'vy': values[7],  # Velocity y
                'vz': values[8],  # Velocity z
                'ax': values[9],  # Acceleration x
                'ay': values[10],  # Acceleration y
                'az': values[11],  # Acceleration z
                # Include other fields as needed
            })
    return oxts_data

def pose_to_transform(pose):
    """
    Convert pose data to a 4x4 homogeneous transformation matrix.
    
    :param pose: dict
        A dictionary containing 'roll', 'pitch', 'yaw', 'lat', 'lon', 'alt'.
    :return: np.ndarray
        4x4 homogeneous transformation matrix.
    """
    # Extract orientation (roll, pitch, yaw)
    roll, pitch, yaw = pose['roll'], pose['pitch'], pose['yaw']
    
    # Compute rotation matrix from roll, pitch, and yaw
    rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    # Extract position (latitude, longitude, altitude)
    # NOTE: You may need to convert lat/lon to meters if necessary.
    # For now, use them directly as x, y, z coordinates.
    position = np.array([pose['lat'], pose['lon'], pose['alt']])
    
    # Construct the transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    
    return T
