from tkinter import font
import cv2
import time 
import os
import numpy as np 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_hands = 1
)

detector = HandLandmarker.create_from_options(options)

#----------Variable-------------
p_time = 0
image_dir = "PoseImages"
pose_img = "PoseImages/1.jpg"


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    if not success:
        break
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
    
    result = detector.detect(mp_image)
    
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            
            h, w, _ = img.shape
            lm_list = []
            x_list = []
            y_list = []
            
            for lm in hand:
                lm_list.append((int(lm.x * w), int(lm.y * h)))
                x_list.append(lm.x * w)
                y_list.append(lm.y * h)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            cv2.rectangle(img,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        3
                    )

            
            for x, y in lm_list:
                cv2.circle(img, (x, y), 8, (0, 255, 0), cv2.FILLED)
    


    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    
    cv2.putText(img,
                f"FPS: {int(fps)}",
                (10, 40),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 0, 0),
                3
                )


    cv2.imshow("Play and Pause", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows