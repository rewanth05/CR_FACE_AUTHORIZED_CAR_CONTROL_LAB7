#!/usr/bin/env python3
import cv2, os, numpy as np, serial, time, mediapipe as mp

URL = "http://10.14.25.228:8080/video"  # replace with your phone IP cam
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

# Haarcascade
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    import urllib.request
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
face_cascade = cv2.CascadeClassifier(cascade_path)

# Arduino serial
ser = serial.Serial('/dev/ttyAMA10', 9600, timeout=1)
time.sleep(2)

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_face.yml")

def get_gesture(hand_landmarks):
    tips = [8, 12]
    pips = [6, 10]
    finger_states = []
    for tip, pip in zip(tips, pips):
        finger_states.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)
    return sum(finger_states)

print("🔹 Starting Face + Hand Control with Recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    authorized = False
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face)

        if conf < 70:
            authorized = True
            cv2.putText(frame, "Authorized", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            cv2.putText(frame, "Unauthorized", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            ser.write(b'S')

    if authorized:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = get_gesture(hand_landmarks)

                if gesture == 1:
                    ser.write(b'F')
                    cv2.putText(frame, "Gesture: 1 (Forward)", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif gesture == 2:
                    ser.write(b'B')
                    cv2.putText(frame, "Gesture: 2 (Backward)", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    ser.write(b'S')
                    cv2.putText(frame, "Gesture: None (Stop)", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        else:
            ser.write(b'S')
            cv2.putText(frame, "No hand detected (Stop)", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    else:
        ser.write(b'S')
        cv2.putText(frame, "Waiting for authorized face...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Face + Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
