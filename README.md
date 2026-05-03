# Object Detection using YOLOv8 + Depth Estimation

## 🚀 Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/RadicalAxis/Object-Detection-
cd Object-Detection-
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

If that fails:

```
pip install ultralytics opencv-python torch numpy
```

---

## ▶️ Run the Project

### 🔹 Webcam Object Detection

```
python test_yolo.py
```

Press **q** to exit.

---

### 🔹 Detect from Image

```
python test_yolo.py --source dataset/images/sample.jpg
```

---

### 🔹 Depth + Detection System

```
python test_depth.py
```

---

## ⚠️ Notes

* First run may download the MiDaS model (~100MB)
* Make sure your webcam is available
* Use Python 3.8 – 3.11

---


## 🎯 Classes

* pen
* case

---

## 👨‍💻 Author

Aryan Rao
