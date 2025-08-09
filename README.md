# YOLOv8 Pose vs YOLOv11 Pose — Real-time Pose Estimation

This project compares **YOLOv8n-pose** and **YOLOv11n-pose** models for **human pose estimation** using OpenCV and Python.  
The goal is to measure **performance, accuracy, and visual stability** between the two versions in real-time applications.

---

## Features
- Run YOLOv8 or YOLOv11 pose detection on videos or live webcam feed
- Draw human skeleton with keypoints (excluding face keypoints)
- Display **real-time speed metrics** (preprocess, inference, postprocess)
- Save processed output video for comparison
- Works on CPU and GPU

---


 **Observation:** YOLOv11 Pose gives more stable skeleton tracking and better handling of motion blur, especially for fast-moving subjects.

---

## Tech Stack
- Python 3.10+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo-pose-comparison.git
cd yolo-pose-comparison

# Install dependencies
pip install -r requirements.txt
