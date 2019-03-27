import plyfile
import pptk
import numpy as np
from pprint import pprint
import os
import sys
import cv2

data_dir = "../data/"
data_subdir = "20180810_110712_merged_cds1_vs"
lidar_file  = "27_1533892329325454000.ply"
camera_file = "27_1533892329329955072.png"

camera_path = os.path.join(data_dir, data_subdir, "images", "front", camera_file)

if True:
    if True:
        filename = lidar_file
# for filename in os.listdir(os.path.join(data_dir, data_subdir, "lidar")):
#     if filename.endswith(".ply"):
#         lidar_file = filename
        lidar_front_file = os.path.splitext(filename)[0] + ".png"
        lidar_path = os.path.join(data_dir, data_subdir, "lidar", lidar_file)
        lidar_front_path = os.path.join(data_dir, data_subdir, "lidar_front", lidar_front_file)

        print(lidar_path)
        print(lidar_front_path)

        if not os.path.exists(os.path.dirname(lidar_front_path)):
            os.makedirs(os.path.dirname(lidar_front_path))

        cloud = plyfile.PlyData.read(lidar_path)
        image = cv2.imread(camera_path)
        rows, cols, _ = image.shape

        data = cloud["vertex"]
        xyz = np.c_[data['x'], data['y'], data['z']]

        camera_matrix = np.array([[1179.490325, 0, 643.2975554], [0, 1180.029922, 405.3232479], [0, 0, 1]], np.float)
        distortion = np.array([0.3860145467, 8.039363313, 0.002375426059, 0.008772298988], np.float)
        xi = np.array([2.472719715], np.float)
        rotation_matrix = np.array([[-0.0303299131870576, 0.2627572075231432, 0.9643851649079964],
                                    [-0.9989171973768571, -0.0420208983349173, -0.0199668956574236],
                                    [0.0352778852079964, -0.9639465203174237, 0.2637471835061358]], np.float)
        translation_vector = np.array([3.4359928140000013, -0.0299222211300000, 0.5531512269999997], np.float)

        # crop the point cloud
        fov_angle = 90 * np.pi / 180
        # xyz = xyz[xyz[:, 0] < 150]
        xyz = xyz[xyz[:, 0] * np.tan(fov_angle) > np.abs(xyz[:, 1])]
        # xyz = xyz[xyz[:, 2] < 20]
        xyz = xyz[xyz[:, 2] > 0]

        # change the transformation to: world -> camera
        rotation_matrix = np.linalg.inv(rotation_matrix)
        translation_vector *= -1

        # ===========================
        # This is the important part
        # ===========================
        # the reshapes fix a bug in the projectPoints() function
        xyz = xyz.reshape((1,xyz.shape[0],3))
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        image_points, _ = cv2.omnidir.projectPoints(xyz, rotation_vector, translation_vector, camera_matrix, xi, distortion)
        image_points = image_points.reshape(image_points.shape[1], 2)
        xyz = xyz.reshape((xyz.shape[1], 3))

        scaling_parameter = 1
        K_scaled = camera_matrix
        K_scaled[0][0] = K_scaled[0][0] / scaling_parameter
        K_scaled[1][1] = K_scaled[1][1] / scaling_parameter
        # image = cv2.omnidir.undistortImage(image, camera_matrix, distortion, xi, cv2.omnidir.RECTIFY_PERSPECTIVE, Knew=K_scaled)

        distance = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2).reshape((xyz.shape[0], 1))
        distance *= 255.0 / distance.max()

        blank_image = np.zeros((rows, cols, 1), np.uint8)
        for i in range(image_points.shape[0]):
            u = int(np.round(image_points[i, 0]))
            v = int(np.round(image_points[i, 1]))

            try:
                cv2.circle(blank_image, (u, v), 1, int(np.round(distance[i])), -1)
                cv2.circle(image, (u, v), 1, (0, 0, 255), -1)
            except:
                pass

        blank_image = np.round(blank_image * 255.0 / blank_image.max()).astype(np.uint8)

        # cv2.imwrite(lidar_front_path, blank_image)

        v = pptk.viewer(xyz)

        scaling = .8
        blank_image = cv2.resize(blank_image, (int(scaling*cols), int(scaling*rows)))
        # cv2.imshow("blank_image", blank_image)

        image = cv2.resize(image, (int(scaling*cols), int(scaling*rows)))
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
