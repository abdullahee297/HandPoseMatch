import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------- MediaPipe Setup ----------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisualRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisualRunningMode.IMAGE,
    num_hands=2
)

detector = HandLandmarker.create_from_options(options)


# -------- Angle Functions ----------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return np.degrees(angle)


def extract_angle_features(landmarks):
    angles = []

    # Thumb
    angles.append(calculate_angle(landmarks[1], landmarks[2], landmarks[3]))
    angles.append(calculate_angle(landmarks[2], landmarks[3], landmarks[4]))

    # Index
    angles.append(calculate_angle(landmarks[5], landmarks[6], landmarks[7]))
    angles.append(calculate_angle(landmarks[6], landmarks[7], landmarks[8]))

    # Middle
    angles.append(calculate_angle(landmarks[9], landmarks[10], landmarks[11]))
    angles.append(calculate_angle(landmarks[10], landmarks[11], landmarks[12]))

    # Ring
    angles.append(calculate_angle(landmarks[13], landmarks[14], landmarks[15]))
    angles.append(calculate_angle(landmarks[14], landmarks[15], landmarks[16]))

    # Pinky
    angles.append(calculate_angle(landmarks[17], landmarks[18], landmarks[19]))
    angles.append(calculate_angle(landmarks[18], landmarks[19], landmarks[20]))

    return np.array(angles)


# -------- Load Pose Dataset ----------
folderPath = "PoseImages"
img_list = os.listdir(folderPath)

pose_features = []
pose_names = []

print("Loading pose dataset...")

for img_name in img_list:
    img = cv2.imread(f"{folderPath}/{img_name}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_img = mp.Image(mp.ImageFormat.SRGB, rgb)
    result = detector.detect(mp_img)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        lm = [(lm.x, lm.y) for lm in hand]

        features = extract_angle_features(lm)

        pose_features.append(features)
        pose_names.append(img_name.split('.')[0])

print("Dataset loaded successfully!")


# -------- Webcam ----------
cap = cv2.VideoCapture(0)
ptime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(mp.ImageFormat.SRGB, rgb)

    result = detector.detect(mp_img)

    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):

            h, w, _ = img.shape
            lm_list = []
            x_list = []
            y_list = []

            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                lm_list.append((x, y))
                x_list.append(x)
                y_list.append(y)

            if len(x_list) > 0:
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

                # -------- Pose Matching --------
                lm = [(lm.x, lm.y) for lm in hand]
                live_features = extract_angle_features(lm)

                min_dist = float("inf")
                matched_pose = "Unknown"

                for j, pose in enumerate(pose_features):
                    dist = np.linalg.norm(live_features - pose)

                    if dist < min_dist:
                        min_dist = dist
                        matched_pose = pose_names[j]

                # -------- Drawing --------
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

                cv2.putText(img,
                            f'Pose: {matched_pose}',
                            (x_min, y_max + 40),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1,
                            (0, 255, 255),
                            3)

                if result.handedness:
                    xlabel = result.handedness[i][0].category_name
                    cv2.putText(img,
                                xlabel,
                                (x_min, y_min - 40),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                1,
                                (255, 0, 0),
                                3)

            # -------- Draw landmarks --------
            for x, y in lm_list:
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    # -------- FPS --------
    ctime = time.time()
    fps = 1 / (ctime - ptime + 1e-6)
    ptime = ctime

    cv2.putText(img,
                f'FPS: {int(fps)}',
                (20, 50),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (255, 0, 0),
                3)

    cv2.imshow("High Accuracy Hand Pose", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
