import cv2
import os

# Create folder to save face samples
person_name = input("Enter person's name: ")
folder = f"dataset/{person_name}"

os.makedirs(folder, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Crop face
        face = frame[y:y+h, x:x+w]

        cv2.putText(frame,
                    f"Samples: {count}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        key = cv2.waitKey(1)

        # Press S to save image
        if key == ord('s'):
            count += 1
            cv2.imwrite(f"{folder}/{count}.jpg", face)
            print(f"Saved Image {count}")

    cv2.imshow("Face Sample Collection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()