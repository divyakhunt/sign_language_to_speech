import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import time
from collections import Counter

# Load model and label encoder
model = tf.keras.models.load_model("model/gesture_dl_model.keras")
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Sign Language Translator started. Press ESC to exit.")

while cap.isOpened():
    prediction_buffer = []

    print("Waiting for hand movement to start capture...")

    # --- Phase 0: Wait for hand to appear ---
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            print("Hand detected! Starting 3-second capture...")
            break

        cv2.putText(frame, "Waiting for hand...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Sign Language Translator", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

    # --- Phase 1: Record for 3 seconds ---
    start_time = time.time()
    while time.time() - start_time < 2:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        left_hand = [0.0] * 63
        right_hand = [0.0] * 63

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, hand_info in enumerate(result.multi_handedness):
                hand_type = hand_info.classification[0].label
                landmarks = result.multi_hand_landmarks[idx]

                coords = []
                for lm in landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                if hand_type == 'Left':
                    left_hand = coords
                elif hand_type == 'Right':
                    right_hand = coords

                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            features = np.array(left_hand + right_hand).reshape(1, -1)
            prediction = model.predict(features)
            predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]
            prediction_buffer.append(predicted_label)

            cv2.putText(frame, "Capturing...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Sign Language Translator", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

    # --- Phase 2: Process Result ---
    if prediction_buffer:
        final_prediction = Counter(prediction_buffer).most_common(1)[0][0]
        print(f"Final Prediction: {final_prediction}")
        engine.say(final_prediction)
        engine.runAndWait()

        # Display result for 2 seconds
        end_time = time.time() + 2
        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f'Predicted Sign: {final_prediction}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Sign Language Translator", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                exit()
    else:
        print("No valid gesture detected.")

cap.release()
cv2.destroyAllWindows()
hands.close()
