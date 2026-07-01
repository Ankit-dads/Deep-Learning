"""
Face Dataset Collector
-----------------------
Opens your webcam, detects a face using OpenCV's Haar cascade,
and saves cropped face images into dataset/<person_name>/ folder.

Run this once per person (change the name each time).

Usage:
    python collect_faces.py
"""

import cv2
import os

# ---------- CONFIG ----------
DATASET_DIR = "dataset"
IMG_SIZE = 128          # size to resize each saved face crop to
NUM_IMAGES = 150        # how many images to capture per person
CAM_INDEX = 0           # change if you have multiple webcams
# -----------------------------

def collect_faces():
    person_name = input("Enter the person's name (label): ").strip().lower().replace(" ", "_")
    if not person_name:
        print("Name cannot be empty.")
        return

    save_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load OpenCV's built-in Haar cascade face detector (no download needed)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    print(f"\nCollecting {NUM_IMAGES} images for '{person_name}'.")
    print("Look at the camera and move your head slightly (angles/expressions).")
    print("Press 'q' at any time to stop early.\n")

    count = 0
    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

            file_path = os.path.join(save_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, face_resized)
            count += 1

            # draw box + counter on the live preview
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/{NUM_IMAGES}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # only take one face per frame to avoid duplicates/strangers in frame

        cv2.imshow("Collecting Faces - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Saved {count} images to '{save_dir}'")


if __name__ == "__main__":
    collect_faces()