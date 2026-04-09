# Real-Time Object Detection and Distance Estimation

## Overview

This project implements a real-time computer vision system that combines object detection and depth estimation to determine the relative distance of objects from a person using a single camera.

The system detects objects (e.g., pen, earphone case, person) and estimates their distance, displaying which object is closer and by how much.

---

## Objective

* Detect specific objects using a custom-trained model
* Estimate relative depth from a monocular camera
* Compute approximate distances in real time
* Compare object distances with respect to a person

---

## System Pipeline

Camera → Object Detection (YOLOv8) → Depth Estimation (MiDaS) → Distance Calculation → Comparison

---

## Components

### Object Detection

* Model: YOLOv8 (Ultralytics)
* Trained on a custom dataset (~200 images)
* Classes: `pen`, `case` (person detected via pretrained weights)

### Depth Estimation

* Model: MiDaS (monocular depth estimation)
* Provides relative depth from a single image

### Distance Calculation

* Depth values converted to approximate distance using calibration:
  distance ≈ depth × scale
* Scale determined using a known reference distance

---

## Tech Stack

* Python
* YOLOv8 (Ultralytics)
* MiDaS (PyTorch)
* OpenCV
* NumPy

---

## Dataset

* Images collected manually with variation in lighting, angle, and background
* Annotated using CVAT in YOLO format

---

## Training

```bash id="v1dr6j"
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

---

## Real-Time Execution

```bash id="l8q5d7"
python main.py
```

---

## Output

* Bounding boxes with labels
* Estimated distance (in cm, approximate)
* Comparison output indicating which object is closer

---

## Challenges

* Depth values are relative and unstable
* Sensitivity to lighting and background
* No direct access to camera calibration parameters

---

## Solutions

* Depth normalization and spatial averaging
* Temporal smoothing using a moving window
* Manual calibration using a known distance

---

## Limitations

* Distance estimation is approximate, not absolute
* Performance depends on camera quality and environment
* Monocular depth estimation has inherent inaccuracies

---

## Learning Outcomes

* Understanding of object detection and localization
* Use of pretrained models for depth estimation
* Integration of multiple computer vision components
* Handling real-time data and noise reduction

---

## Author

Aryan Rao
