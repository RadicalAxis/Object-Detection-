import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLO Detection", annotated_frame)

    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()