# YOLO Basics

## Introduction
The YOLO_Basics folder contains scripts to help understand the basics of working with YOLO models, including object detection and tracking. These examples demonstrate how to use YOLO's outputs effectively in real-world applications.

## Contents

### 1. Simple YOLO Object Detection (`yolo_detection.py`)
This script provides a minimal example of running YOLO on a webcam feed. It:
- Loads a pre-trained YOLO model (`yolov8n.pt`)
- Captures frames from the webcam
- Runs YOLO inference and extracts:
  - Bounding boxes
  - Confidence scores
  - Object class labels
- Draws detections on the frame and displays them in real-time

#### **Code Overview:**
```python
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load YOLO model
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    results = model(frame)  # Run YOLO inference
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = model.names[class_id]  # Class label
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. YOLO Face Recognition Benchmark (`yolo_facial_recognition.py`)
This script expands on basic object detection by implementing:
- **Custom YOLO model (`eu_11.pt`) trained for facial recognition**
- **Tracking with ByteTrack**
- **Custom class IDs:**
  - `ID 0` = Any face
  - `ID 1` = User's face (recognized person)
- **Behavioral logic:**
  - If `ID 1` is detected → Print "Welcome!"
  - If only `ID 0` is detected → Count number of people
  - Saves images of unrecognized people

#### **Key Code Features:**
- Uses `YOLO.track()` to track faces
- Extracts bounding box coordinates and class IDs
- Uses OpenCV (`cv2`) to display results
- Implements **frame processing optimizations** for real-time performance

#### **How to Run the Face Recognition Script:**
```bash
python yolo_facial_recognition.py
```

## Summary
This folder provides two fundamental scripts:
1. **Basic Object Detection** (`yolo_detection.py`) - For learning YOLO output handling
2. **Facial Recognition Benchmark** (`yolo_facial_recognition.py`) - A step towards a real-world use case

By understanding these, you’ll be well-prepared to work with YOLO on custom applications, including **low-budget hardware like Raspberry Pi**.

🚀 Happy Coding!

