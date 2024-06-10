import cv2
import numpy as np

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from PoseDetection import PoseDetector
import time
import json

def pose_detection(video_path, json_output_path):
    try:
        video_input = cv2.VideoCapture(video_path)
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
    
    detector = PoseDetector()
    pTime = 0
    cTime = 0
    target_fps = 30
    wait_key_delay = int(1000 / target_fps)

    frame_count = 0
    output_data = {
        "pose2d" : []
    }

    visualize = True
    if not visualize:
        wait_key_delay = 1

    while True:

        ret, img = video_input.read()

        if ret:
            img = detector.findPose(img, draw=visualize)
            imList = detector.findPosition(img)
            output_data["pose2d"].append(
                {
                    "frame": frame_count,
                    "poses" : imList
                }
            )

            frame_count += 1

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            if visualize:
                cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.imshow("Image", img)

        else:
            break
        
        if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'):
            break

    video_input.release()
    cv2.destroyAllWindows()

    with open(json_output_path, 'w') as fp:
        json.dump(output_data, fp)

if __name__ == "__main__":

    video_path = "./output/position/video/950122060411/vid_00000_undistorted.mp4"
    json_path = "./output/position/video/950122060411/vid_00000_pose2d.json"
    pose_detection(video_path, json_path)

    video_path = "./output/position/video/950122061749/vid_00000_undistorted.mp4"
    json_path = "./output/position/video/950122061749/vid_00000_pose2d.json"
    pose_detection(video_path, json_path)