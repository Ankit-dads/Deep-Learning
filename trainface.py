import cv2
import os
import numpy as np
import pickle

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Haar Cascade
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

dataset_path = "dataset"

faces = []
labels = []

label_ids = {}
current_id = 0

# Read all folders inside dataset
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    # Assign numeric ID
    if person_name not in label_ids:
        label_ids[person_name] = current_id
        current_id += 1

    label = label_ids[person_name]

    # Read every image
    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            faces.append(face)
            labels.append(label)

# Train the recognizer
recognizer.train(faces, np.array(labels))

# Save trained model
recognizer.save("trainer.yml")

# Save labels
with open("labels.pkl", "wb") as f:
    pickle.dump(label_ids, f)

print("Training completed successfully!")
print(label_ids)