import argparse
import numpy as np
import cv2
import os
from utils import ucitaj_sve_parametre, ucitaj_fajlove, ucitaj_markere, sacuvaj_pozu
import json

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def rekonstruisi_pozu(i, all_points, all_intrinsic, all_rot, all_cam_pos, threshold):
    num_cameras = len(all_points)
    points = list()
    for pt in all_points:
        points.append(pt[:, :, i])
    rays = list()
    confidence = list()
    for pt, intrinsic, rot, cp in zip(points, all_intrinsic, all_rot, all_cam_pos):
        q = pt.T.copy()
        q[2, :] = 1
        ray = np.matmul(np.linalg.inv(np.matmul(intrinsic,rot.T)) ,q)
        # izracunaj zrakove 
        rays.append(ray)
        confidence.append(pt[:,2])
    num_pts = ray.shape[1]        
    res = np.zeros((num_pts,4), dtype=float)        
    lhs = np.zeros((3*num_cameras, num_cameras+3), dtype=float)
    for idx in range(num_cameras):
        lhs[idx*3:idx*3+3, num_cameras:num_cameras+3] = np.identity(3, dtype=float)
    rhs = np.vstack(all_cam_pos)
    for j in range(num_pts):
        for idx, ray in enumerate(rays):
            lhs[3*idx:3*idx+3, idx] = ray[:,j]
        row_mask = np.array([True]*3*num_cameras, dtype=bool)
        col_mask = np.array([True]*(num_cameras + 3), dtype=bool)
        num_above = 0
        mean_confidence = 0
        for idx, conf in enumerate(confidence):
            if conf[j] > threshold:
                num_above =  num_above + 1
                mean_confidence = mean_confidence + conf[j]
            else:                
                col_mask[idx] = False
                row_mask[3*idx:3*idx+3]= False
        if num_above>1:
            # izbaci one za koje je confidence mali
            L = lhs[row_mask,:][:,col_mask]
            R = rhs[row_mask,:]
            sol = np.matmul(np.linalg.pinv(L),R)   
            res[j,0:3] = sol[-3:].ravel()
            res[j,3] = mean_confidence/num_above
        else:
            res[j,:] = 0            
    return res      

def rekonsruisi(json_paths, calib_files, rotations, output, threshold):
    num_cameras = len(json_paths)
    all_files = list()
    all_intrinsic = list()
    all_rot = list()
    all_trans = list()
    all_cam_pos = list()
    num_frames = 100000
    for path, calib_file in zip(json_paths, calib_files):
        all_files.append(ucitaj_fajlove(path, ekstenzije={".json"}))
        num_frames = min(num_frames, len(all_files[-1]))
        intrinsic, _ , rot, trans = ucitaj_sve_parametre(calib_file)
        all_intrinsic.append(intrinsic)
        # pozicija kamere u odnosu na koordinatni pocetak
        all_rot.append(rot.T)
        all_cam_pos.append(-np.matmul(rot.T,trans))
    all_points = list()
    for files, fr in zip(all_files, rotations):
        points = list()
        for file in files:
            f = open(file)
            data = json.load(f)
            for frame in data["pose2d"]:
                points.append(np.asarray(frame["poses"]))
        all_points.append(np.stack(points, axis=2))

    num_frames = all_points[0].shape[-1]
    for i in range(num_frames):
        pose = rekonstruisi_pozu(i, all_points, all_intrinsic, all_rot, all_cam_pos, threshold)
        sacuvaj_pozu(pose, i, output)        

                


if __name__ == "__main__":
    
    # putanje do JSON fajlova
    json_paths = [
        "./output/position/video/950122060411/",
        "./output/position/video/950122061749/"
    ]

    # putanja do kalibracionih fajlova
    calib_files = [
        "./output/calib/950122060411/calib.yaml",
        "./output/calib/950122061749/calib.yaml"
    ]

    frame_rotate = [0.0, 0.0, 0.0]

    # ime rezultujucih JSON fajlova
    output = "./output/poses3d/poze_{num:05d}.json"
    
    # vrednost za koju smatramo da je detektovana tacka validna
    threshold = 0.5
    rekonsruisi(json_paths, calib_files, frame_rotate, output, threshold)