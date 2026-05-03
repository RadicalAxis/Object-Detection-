import os
import cv2
from ultralytics import YOLO

# -------- PATH SETUP --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")

# -------- LOAD MODEL --------
model = YOLO(model_path)

# -------- WEBCAM --------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam")
    exit()

# -------- MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Safe display
    try:
        cv2.imshow("YOLO Detection", annotated_frame)
    except:
        pass

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
