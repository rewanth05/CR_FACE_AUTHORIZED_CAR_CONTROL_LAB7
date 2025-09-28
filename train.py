import cv2, os, numpy as np

data_path = "faces/rewanth"
faces, ids = [], []

for img_name in os.listdir(data_path):
    img_path = os.path.join(data_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    faces.append(img)
    ids.append(1)   # Only 1 person (you)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))
recognizer.save("trained_face.yml")

print("✅ Training complete. Model saved as trained_face.yml")
