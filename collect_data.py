import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Prepare folders
CSV_PATH = 'data/gesture_landmarks_both_hands.csv'
os.makedirs('data', exist_ok=True)

# Get label and sample count
gesture_label = input("Enter the gesture label (e.g., hello, thanks): ").lower()
samples_per_class = int(input("How many samples to capture for this gesture? "))

data = []
cap = cv2.VideoCapture(0)
print(f"📷 Capturing both hands for: '{gesture_label}'")

count = 0
while cap.isOpened() and count < samples_per_class:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Initialize empty lists for left/right hand landmarks
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_info in enumerate(result.multi_handedness):
            hand_type = hand_info.classification[0].label
            landmarks = result.multi_hand_landmarks[idx]

            # Get flattened [x,y,z,...] list
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            if hand_type == 'Left':
                left_hand = coords
            elif hand_type == 'Right':
                right_hand = coords

            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Combine features: left + right + label
        features = left_hand + right_hand + [gesture_label]
        data.append(features)
        count += 1
        print(f"Sample {count}/{samples_per_class} captured")

    cv2.imshow("Collecting Gesture Data (Both Hands)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to stop
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

# Column names
columns = [f'l_{a}{i}' for a in 'xyz' for i in range(21)] + \
          [f'r_{a}{i}' for a in 'xyz' for i in range(21)] + ['label']

# Save to CSV
df = pd.DataFrame(data, columns=columns)
if os.path.exists(CSV_PATH):
    df.to_csv(CSV_PATH, mode='a', index=False, header=False)
else:
    df.to_csv(CSV_PATH, index=False)

print(f"\nSaved to: {CSV_PATH}")
