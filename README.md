# Object-Detection-

# Custom Object Detection using YOLOv8

## Overview

This project implements a real-time object detection system using YOLOv8.
The model is trained on a custom dataset to detect specific objects such as a pen and an earphone case through a webcam feed.

---

## Objective

To build a custom object detection model that:

* Detects specific objects in real time
* Demonstrates understanding of detection (classification + localization)

---

## Key Concepts

### Object Detection

Object detection involves:

* Classification (what the object is)
* Localization (where the object is)

The model outputs bounding boxes, class labels, and confidence scores.

### YOLOv8

YOLO (You Only Look Once) is a real-time detection algorithm that:

* Processes the entire image in one pass
* Predicts bounding boxes and classes simultaneously

---

## Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy

---

## Dataset

### Data Collection

* ~200 images captured manually
* Includes variation in lighting, angles, and background

### Classes

* pen
* case

### Annotation

* Tool: CVAT
* Format: YOLO

---

## Training

### Command

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

### Metrics

* Precision (P)
* Recall (R)
* mAP50
* mAP50-95

---

## Real-Time Detection

### Run

```bash
python test_yolo.py
```

### Output

* Webcam feed with bounding boxes
* Detected labels and confidence scores

---

## Challenges

* Limited dataset size
* Sensitivity to annotation quality

---

## Improvements

* Increase dataset size
* Use larger YOLO models
* Add more object classes

---

## Learning Outcomes

* Understanding object detection vs classification
* Dataset creation and annotation
* Training and evaluating detection models
* Real-time computer vision implementation

---

## Author

Aryan Rao
