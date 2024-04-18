import os
import cv2
import numpy as np


def ucitaj_fajlove(folder, sort=True, ekstenzije=None):
    all_files = list()
    for f in os.listdir(folder):
        if ekstenzije == None or (os.path.splitext(f)[-1] in ekstenzije):
            all_files.append(os.path.join(folder, f))
    if sort:
        all_files.sort()
    return all_files


def ucitaj_unutrasnje_parametre(calib_file):
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    cam_mat = fs.getNode('intrinsic').mat()
    dist_coeffs = fs.getNode('distortion').mat()
    fs.release()
    return (cam_mat, dist_coeffs)

def projektuj_tacke(tacke, camera_matrix, rot_mat, translation):
    image_points = np.matmul(
        camera_matrix, np.matmul(rot_mat, tacke)+translation)
    return (image_points/image_points[2, :])[0:2, :]
