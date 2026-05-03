import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 🔥 CHANGE THIS AFTER CALIBRATION
SCALE = 0.0625  # example value (you will adjust)

# Load YOLO model
model = YOLO("runs/detect/train2/weights/best.pt")

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Load transforms
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Store object distances
    objects = {}

    # YOLO detection
    results = model(frame)

    # Prepare image for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to("cpu")

    # Get depth
    with torch.no_grad():
        depth = midas(input_batch)

    depth = depth.squeeze().cpu().numpy()

    # Resize depth to match frame
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # Loop through detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Safe region
            h, w = depth.shape
            x1_safe = max(0, cx - 3)
            x2_safe = min(w, cx + 3)
            y1_safe = max(0, cy - 3)
            y2_safe = min(h, cy + 3)

            region = depth[y1_safe:y2_safe, x1_safe:x2_safe]

            if region.size > 0:
                depth_value = np.mean(region)
                distance_cm = depth_value * SCALE
            else:
                distance_cm = 0

            # Get label
            label = model.names[int(box.cls[0])]

            # Save distance
            objects[label] = distance_cm

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show distance
            cv2.putText(frame, f"{label} {distance_cm:.1f} cm",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # 🔥 Compare objects
    if len(objects) >= 2:
        items = list(objects.items())

        obj1, dist1 = items[0]
        obj2, dist2 = items[1]

        if dist1 < dist2:
            closer_text = f"{obj1} is closer by {abs(dist1 - dist2):.1f} cm"
        else:
            closer_text = f"{obj2} is closer by {abs(dist1 - dist2):.1f} cm"

        cv2.putText(frame, closer_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Final System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()