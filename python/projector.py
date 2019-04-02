import plyfile
import pptk
import numpy as np
from pprint import pprint
import os
import sys
import cv2

data_dir = os.path.join("..", "data")
data_subdir = "20180810_110712_merged_cds1_vs"
lidar_file  = "163_1533892329329955072.ply"

view = "front"
camera_file = "163_1533892329329955072.png"
# view = "rear"
# camera_file = "163_1533892329329914112.png"
# view = "left"
# camera_file = "163_1533892329329912064.png"
# view = "right"
# camera_file = "163_1533892329329910016.png"

lidar_path = os.path.join(data_dir, data_subdir, "lidar", lidar_file)
camera_path = os.path.join(data_dir, data_subdir, "images", "front", camera_file)

# read point cloud and image files
cloud = plyfile.PlyData.read(lidar_path)
image = cv2.imread(camera_path)
rows, cols, _ = image.shape
data = cloud["vertex"]
xyz = np.c_[data['x'], data['y'], data['z']]
rgb = np.c_[data['red'], data['green'], data['blue']]

# extrinsics/intrinsics of cameras
if view == "front":
		camera_matrix = np.array([[1179.490325, 0, 643.2975554],
															[0, 1180.029922, 405.3232479],
															[0, 0, 1]], np.float)
		distortion = np.array([0.3860145467, 8.039363313, 0.002375426059, 0.008772298988], np.float)
		xi = np.array([2.472719715], np.float)
		rotation_matrix = np.array([[-0.0303299131870576, 0.2627572075231432, 0.9643851649079964],
																[-0.9989171973768571, -0.0420208983349173, -0.0199668956574236],
																[0.0352778852079964, -0.9639465203174237, 0.2637471835061358]], np.float)
		translation_vector = np.array([3.4359928140000013, -0.0299222211300000, 0.5531512269999997], np.float)
elif view == "rear":
		camera_matrix = np.array([[1192.952515, 0, 637.723947],
															[0, 1192.935194, 405.8451243],
															[0, 0, 1]], np.float)
		distortion = np.array([0.2969756276, 9.647238147, -0.00331740976, -0.001876569656], np.float)
		xi = np.array([2.543406294], np.float)
		rotation_matrix = np.array([[0.0409540282131009, 0.0077676185071361, -0.9991308380768675],
																[0.9967952490547533, -0.0690900523652394, 0.0403211611593155],
																[-0.0687168025691570, -0.9975801865905771, -0.0105722411359310]], np.float)
		translation_vector = np.array([-0.7274199422000002, -0.0125287283800000, 0.8264463890000003], np.float)
elif view == "left":
		camera_matrix = np.array([[1198.106024, 0, 639.2537473],
															[0, 1201.230742, 406.2035438],
															[0, 0, 1]], np.float)
		distortion = np.array([0.3141793111, 9.239584037, 0.001900200444, 0.01152732215], np.float)
		xi = np.array([2.521813709], np.float)
		rotation_matrix = np.array([[0.9775362421532896, -0.2037356101757669, 0.0539879284924119],
																[-0.1880587071242332, -0.7274556703119985, 0.6598804212794160],
																[-0.0951673155624121, -0.6552099273205845, -0.7494285377347125]], np.float)
		translation_vector = np.array([1.8727794350000009, 0.9891099504999999, 0.8964326523000002], np.float)
else: # view = "right"
		camera_matrix = np.array([[1186.967946, 0, 634.94413],
															[0, 1188.934661, 399.6917314],
															[0, 0, 1]], np.float)
		distortion = np.array([0.3040144946, 8.82342109, 0.001172507258, 0.01047374008], np.float)
		xi = np.array([2.524133284], np.float)
		rotation_matrix = np.array([[-0.9547663921401841, -0.2908932279241032, 0.0616625200022027],
																[-0.2725045054774874, 0.7729582786437673, -0.5729544431879827],
																[0.1190060121153396, -0.5638409611071875, -0.8172643021964534]], np.float)
		translation_vector = np.array([1.8431070280000008, -0.9961844253000001, 0.9175194592000003], np.float)

# crop the point cloud
if view == "front":
		rgb = rgb[xyz[:, 0] > 0]
		xyz = xyz[xyz[:, 0] > 0]
elif view == "rear":
		rgb = rgb[xyz[:, 0] < 0]
		xyz = xyz[xyz[:, 0] < 0]
elif view == "left":
		rgb = rgb[xyz[:, 1] > 0]
		xyz = xyz[xyz[:, 1] > 0]
else:  # view = "right"
		rgb = rgb[xyz[:, 1] < 0]
		xyz = xyz[xyz[:, 1] < 0]
# ground removal
rgb = rgb[xyz[:, 2] > .1]
xyz = xyz[xyz[:, 2] > .1]
# distance cropping
rgb = rgb[np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2) < 100]
xyz = xyz[np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2) < 100]

# change the transformation to: world -> camera
rotation_matrix = np.linalg.inv(rotation_matrix)
translation_vector *= -1

# transform the points
xyz += translation_vector.T
xyz = (rotation_matrix.dot(xyz.T)).T

# ===========================
# This is the important part
# ===========================
# the reshapes fix a bug in the projectPoints() function
xyz = xyz.reshape((1,xyz.shape[0],3))
image_points, _ = cv2.omnidir.projectPoints(xyz, np.zeros((3,1), np.float), np.zeros((3,1), np.float), camera_matrix, xi, distortion)
image_points = image_points.reshape(image_points.shape[1], 2)
xyz = xyz.reshape((xyz.shape[1], 3))

# compute Euclidean distance for each point without taking translation into account
distance = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2).reshape((xyz.shape[0], 1))

blank_image = np.zeros((rows, cols, 1), np.uint8)
for i in range(image_points.shape[0]):
		u = int(np.round(image_points[i, 0]))
		v = int(np.round(image_points[i, 1]))

		try:
				cv2.circle(blank_image, (u, v), 2, int(np.round(100/distance[i])), -1)
				cv2.circle(image, (u, v), 1, (0, 0, 255), -1)
		except:
				pass
blank_image = np.round(blank_image * 255.0 / blank_image.max()).astype(np.uint8)

# undistort the raw image including the project point cloud
# scaling_parameter = 2
# K_scaled = camera_matrix.copy()
# K_scaled[0][0] = K_scaled[0][0] / scaling_parameter
# K_scaled[1][1] = K_scaled[1][1] / scaling_parameter
# image = cv2.omnidir.undistortImage(image, camera_matrix, distortion, xi, cv2.omnidir.RECTIFY_PERSPECTIVE, Knew=K_scaled)

# plot the colored point cloud
# v = pptk.viewer(xyz)
# v.attributes(rgb / 255.)

# show visual results
scaling = .5
size = (int(scaling * cols), int(scaling * rows))
blank_image = cv2.resize(blank_image, size)
image = cv2.resize(image, size)
# cv2.imshow("blank_image", blank_image)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
