import cv2
import pickle

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Reverse dictionary (ID -> Name)
labels = {v: k for k, v in labels.items()}

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100,100)
    )

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]

        id_, confidence = recognizer.predict(roi)

        # Convert LBPH distance to confidence %
        confidence_percent = max(0, min(100, 100 - confidence))

        if confidence < 70:

            name = labels[id_]

            text = f"{name} ({confidence_percent:.1f}%)"

            color = (0,255,0)

        else:

            text = "Unknown"

            color = (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.putText(frame,
                    text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    cv2.imshow("Face Recognition",frame)

    key = cv2.waitKey(1) & 0xFF

    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()