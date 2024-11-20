from calib_parse import *# parse_cam_to_cam, parse_velo_to_cam

cam_to_cam_data = parse_cam_to_cam()#cam_to_cam_path)
imu_to_velo_matrix = parse_imu_to_velo()#imu_to_velo_path)
velo_to_cam_matrix = parse_velo_to_cam()#velo_to_cam_path)

# Access parsed data
K_02 = cam_to_cam_data["K_02"]
P_rect_02 = cam_to_cam_data["P_rect_02"]
print(cam_to_cam_data['R_rect_00'])
# print("K_02:\n", K_02)
# print("P_rect_02:\n", P_rect_02)
# print("IMU to Velodyne:\n", imu_to_velo_matrix)
# print("Velodyne to Camera:\n", velo_to_cam_matrix)