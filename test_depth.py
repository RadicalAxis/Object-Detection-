import cv2
import torch
import numpy as np
from ultralytics import YOLO


SCALE = 0.0625  

model = YOLO("runs/detect/train2/weights/best.pt")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    objects = {}

    results = model(frame)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to("cpu")

    with torch.no_grad():
        depth = midas(input_batch)

    depth = depth.squeeze().cpu().numpy()

    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

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

            label = model.names[int(box.cls[0])]

            objects[label] = distance_cm

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"{label} {distance_cm:.1f} cm",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

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

    cv2.imshow("Final System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
