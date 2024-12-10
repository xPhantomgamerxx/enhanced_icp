import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d



def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001, use_semantic_features=True):
    """
    Enhanced ICP to handle enriched points with semantic features.

    Input:
        A: Nxm numpy array of source points (3D spatial + semantic features).
        B: Nxm numpy array of destination points (3D spatial + semantic features).
        init_pose: (4x4) homogeneous transformation matrix.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        use_semantic_features: If True, include semantic features in nearest neighbor matching.
    Output:
        T: Final homogeneous transformation matrix.
        distances: Euclidean distances of the nearest neighbor.
        i: Number of iterations to converge.
    """
    assert A.shape == B.shape
    m = A.shape[1]

    # Split spatial and semantic features
    A_spatial, A_features = A[:, :3], A[:, 3:]
    B_spatial, B_features = B[:, :3], B[:, 3:]

    # Initialize homogeneous coordinates for spatial alignment
    
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[:3, :] = A_spatial.T
    dst[:3, :] = B_spatial.T

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # Nearest neighbor search
        if use_semantic_features:
            # Combine spatial and semantic features
            combined_src = np.hstack((src[:3, :].T, A_features))
            combined_dst = np.hstack((dst[:3, :].T, B_features))
            distances, indices = nearest_neighbor(combined_src, combined_dst)
        else:
            distances, indices = nearest_neighbor(src[:3, :].T, dst[:3, :].T)

        # Compute best-fit transformation for spatial alignment
        T, _, _ = best_fit_transform(src[:3, :].T, dst[:3, indices].T)

        # Apply transformation to source points
        src = np.dot(T, src)

        # Check convergence
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Calculate final transformation
    T, _, _ = best_fit_transform(A_spatial, src[:3, :].T)

    return T, distances, i

def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions

    Input:
    A: Nxm numpy array of corresponding points
    B: Nxm numpy array of corresponding points
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    """

    assert A.shape == B.shape

    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src

    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel() 
        
def visualize_icp_results(source_points, target_points, transformed_source):
    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()
    transformed_cloud = o3d.geometry.PointCloud()

    source_cloud.points = o3d.utility.Vector3dVector(source_points[:, :3])
    target_cloud.points = o3d.utility.Vector3dVector(target_points[:, :3])
    transformed_cloud.points = o3d.utility.Vector3dVector(transformed_source[:, :3])

    source_cloud.paint_uniform_color([1, 0, 0])  # Red for original source
    target_cloud.paint_uniform_color([0, 1, 0])  # Green for target
    transformed_cloud.paint_uniform_color([0, 0, 1])  # Blue for transformed source

    o3d.visualization.draw_geometries([source_cloud, target_cloud, transformed_cloud])
