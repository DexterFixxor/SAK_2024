import argparse
import numpy as np
import cv2
import os
from utils import ucitaj_unutrasnje_parametre

def undistort_video(input_video, calib_file, output_video):
    cam_mat, dist_coeffs = ucitaj_unutrasnje_parametre(calib_file)
    try:
        video_input = cv2.VideoCapture(input_video)
    except:
        print("Nisam mogao da otvorim video")
        return
    if not video_input.isOpened():
        print("Nisam mogao da otvorim video")
        return      
    width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))        
    fps = float(video_input.get(cv2.CAP_PROP_FPS))
    fourcc = int(video_input.get(cv2.CAP_PROP_FOURCC))
    video_out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(
        cam_mat, dist_coeffs, None, cam_mat, (width, height),cv2.CV_16SC2)
    cv2.namedWindow('Undistorted', cv2.WINDOW_KEEPRATIO)
    while True:
        ret, frame = video_input.read()
        if not ret:
            break
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        cv2.imshow("Undistorted", np.hstack((frame, undistorted)))
        cv2.waitKey(int(1000.0/fps))
        video_out.write(undistorted)
    video_out.release()        

if __name__ == "__main__":
    
    video = "./output/position/video/950122061749/vid_00000.mp4"
    calib_file = "./output/calib/950122061749/calib.yaml"
    output = "./output/position/video/950122061749/vid_00000_undistorted.mp4"
    undistort_video(video, calib_file, output)

    video = "./output/position/video/950122060411/vid_00000.mp4"
    calib_file = "./output/calib/950122060411/calib.yaml"
    output = "./output/position/video/950122060411/vid_00000_undistorted.mp4"
    undistort_video(video, calib_file, output)