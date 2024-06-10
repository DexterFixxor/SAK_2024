import argparse
import numpy as np
import cv2
import os
from itertools import zip_longest
from utils import ucitaj_fajlove, iscrtaj_markere, ucitaj_markere, \
    ucitaj_projektuj_3d_markere, ucitaj_sve_parametre

FPS = 15

def generisi_video(output, marker_3d_folder, focus, size, pos, rot):
    video_out = cv2.VideoWriter(
        output, cv2.VideoWriter_fourcc(*'MP4V'), FPS, size)
    intrinsic = np.matrix(
        [[focus, 0, size[0]/2], [0, focus, size[1]/2], [0, 0, 1]])
    trans = np.array(pos).reshape((3,1))
    Rz, _ = cv2.Rodrigues((0, 0, rot[2]*np.pi/180.0))
    Ry, _ = cv2.Rodrigues((0, rot[1]*np.pi/180.0, 0))
    Rx, _ = cv2.Rodrigues((rot[0]*np.pi/180.0, 0, 0))
    rot_mat = np.matmul(Rz, np.matmul(Ry,Rx)).T
    trans = - np.matmul(rot_mat,np.array(pos).reshape((3,1)))
    marker_3d_fajlovi = ucitaj_fajlove(marker_3d_folder, ekstenzije={".json"})
    frame = np.zeros([size[1],size[0], 3], dtype=np.uint8)
    cv2.namedWindow('Marker', cv2.WINDOW_KEEPRATIO)
    for m3d_file in marker_3d_fajlovi:
        frame.fill(255)
        keypoints = ucitaj_projektuj_3d_markere(m3d_file, intrinsic, rot_mat, trans)
        frame = iscrtaj_markere(
                frame, keypoints, threshold=0.6, color=(255, 255, 0))
        video_out.write(frame)                            
        cv2.imshow('Marker', frame)
        cv2.waitKey(int(1000/FPS))
    video_out.release()


if __name__ == "__main__":
    
    output = "video_x.mp4"
    marker_3d = "./output/poses3d/"

    focus = 800
    width = 1280
    height = 720

    x = 3500
    y = 0
    z = -1500
    roll = 90
    pitch = 0
    yaw = -90

    generisi_video(output, 
                   marker_3d, 
                   focus, 
                   (width,  height),
                   pos=(x, y, z), 
                   rot=(roll, pitch, yaw)
                   )