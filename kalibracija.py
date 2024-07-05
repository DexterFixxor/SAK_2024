import argparse
import numpy as np
import cv2
import os
from utils import ucitaj_fajlove
from utils import create_dir

def calibrate(images_path, file_name, width, height, cell_size):
    all_files = list()
    all_files = ucitaj_fajlove(images_path, ekstenzije={".png", ".jpg"})
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    single_object_points = np.zeros((width*height, 3), np.float32)
    for i in range(height):
        for j in range(width):
            single_object_points[i*width + j, :] = np.array(
                [j*cell_size, i*cell_size, 0], dtype=np.float32)
    object_points = []  # 3d point in real world space
    image_points = []
    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    for image in all_files:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(single_object_points)
            image_points.append(corners2)
            img = cv2.drawChessboardCorners(
                img, (width, height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    mean_error = 0
    print("Duzina objekata")
    print(len(object_points))
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(
            single_object_points, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(image_points[i],
                         imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error/len(object_points)))

    create_dir(file_name)
    file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    file.write("distortion", dist)
    file.write("intrinsic", mtx)
    file.release()


if __name__ == "__main__":
    ids = ['950122060411', '950122061707', '950122061749']
    #ids = ['950122061707']
    for id in ids:
        images_path = f"./data/calib/images/{id}"
        w, h = 8, 5
        cell_size = 30
        
        output_calib_file = f"./output/calib/{id}/calib.yaml"
        calibrate(images_path, output_calib_file, int(w), int(h), cell_size)
        
