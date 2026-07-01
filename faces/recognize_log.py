"""
Face Recognition + Attendance Logger
---------------------------------------
Same as recognize_faces.py, but logs each recognized person to a CSV
with a timestamp. Uses a cooldown so the same person isn't logged
every single frame (e.g. once every 30 seconds per person).

Usage:
    python recognize_and_log.py
"""

import cv2
import json
import csv
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------- CONFIG ----------
MODEL_PATH = "face_cnn_model.pth"
LABELS_PATH = "class_labels.json"
ATTENDANCE_CSV = "attendance_log.csv"
IMG_SIZE = 128
CAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.70
LOG_COOLDOWN_SECONDS = 30   # min gap before logging the same person again
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model():
    with open(LABELS_PATH, "r") as f:
        class_names = json.load(f)
    model = FaceCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def init_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "time", "confidence"])


def log_attendance(name, confidence):
    now = datetime.now()
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), f"{confidence*100:.1f}"])


def main():
    model, class_names = load_model()
    init_csv()
    print(f"Loaded model. Classes: {class_names}")
    print(f"Logging attendance to '{ATTENDANCE_CSV}'\n")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    last_logged = {}  # name -> last logged timestamp (epoch seconds)

    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            pil_img = Image.fromarray(face_crop)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()

            if confidence >= CONFIDENCE_THRESHOLD:
                name = class_names[predicted_idx]
                color = (0, 255, 0)
                label = f"{name} ({confidence*100:.1f}%)"

                now_ts = time.time()
                if name not in last_logged or (now_ts - last_logged[name]) > LOG_COOLDOWN_SECONDS:
                    log_attendance(name, confidence)
                    last_logged[name] = now_ts
                    print(f"Logged: {name} at {datetime.now().strftime('%H:%M:%S')}")
            else:
                name = "Unknown"
                color = (0, 0, 255)
                label = f"Unknown ({confidence*100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition + Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()