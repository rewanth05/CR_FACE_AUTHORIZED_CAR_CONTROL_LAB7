# CR_FACE_AUTHORIZED_CAR_CONTROL_LAB7

Smart Car Kit Intelligence (Raspberry Pi Based)
Face Recognition Mode:
Detects a registered face.
If face is recognized → Move Forward.
If face not recognized → Stop.
Hand Gesture Recognition Mode:
Gesture 1 (Palm Up) → Move Forward.
Gesture 2 (Palm Down) → Move Backward.
Gesture 3 (Closed Fist) → Stop.
Basic Control Intelligence:
1 → Forward
2 → Backward
3 → Stop
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Step 1: Collect dataset
python3 collect_faces.py

# Step 2: Train model
python3 train.py

# Step 3: Run the system
python3 main.py
