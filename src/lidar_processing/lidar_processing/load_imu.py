import numpy as np
from scipy.spatial.transform import Rotation as R

def load_oxts(oxts_file):
    """
    Load OXTS data from a single-line file.
    
    :param oxts_file: str
        Path to the oxts file.
    :return: numpy array
        Array containing the parsed OXTS data.
    """
    with open(oxts_file, 'r') as f:
        line = f.readline().strip()
        values = np.array([float(x) for x in line.split()])
    return values


def pose_to_transform(oxts_data):
    """
    Convert OXTS data to a 4x4 homogeneous transformation matrix.
    
    :param oxts_data: numpy array
        Array containing the OXTS data.
        Expected order: [lat, lon, alt, roll, pitch, yaw, ...].
    :return: np.ndarray
        4x4 homogeneous transformation matrix.
    """
    # Extract orientation (roll, pitch, yaw)

    roll, pitch, yaw = oxts_data[3], oxts_data[4], oxts_data[5]
    
    # Compute rotation matrix from roll, pitch, and yaw
    rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    # Extract position (latitude, longitude, altitude)
    # You might want to convert lat/lon to Cartesian coordinates if needed
    position = geodetic_to_ecef(oxts_data[0], oxts_data[1], oxts_data[2])
    
    # Construct the transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    
    return T


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert geodetic coordinates to Earth-Centered Earth-Fixed (ECEF) coordinates.
    Assumes WGS-84 ellipsoid.
    """
    # WGS-84 ellipsoid constants
    a = 6378137.0  # Semi-major axis (meters)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # Square of eccentricity

    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate N, the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Convert to ECEF
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - e2) * N + alt) * np.sin(lat_rad)

    return np.array([x, y, z])