# Training YOLOv8 on a Custom Dataset

## Introduction
This document explains how to train a YOLOv8 model using a custom dataset from Roboflow. The training process involves setting up dependencies, downloading the dataset, training the model, and running predictions.

## Setup
### 1. Install Required Libraries
```python
%pip install ultralytics
import ultralytics
ultralytics.checks()

!pip install roboflow
```

- `ultralytics`: Provides YOLOv8 implementation.
- `roboflow`: Used to fetch datasets from Roboflow.

### 2. Import Dependencies
```python
from ultralytics import YOLO
import os
from IPython.display import Image, display
from roboflow import Roboflow
```

## Dataset Download
### 3. Fetch Dataset from Roboflow
```python
rf = Roboflow(api_key="API_KEY")  # Replace with your API key
project = rf.workspace("caio-yn7ed").project("face-detection-f6kds")
version = project.version(5)
dataset = version.download("yolov8")
```

### 4. Organize Dataset
```python
os.makedirs('datasets', exist_ok=True)
!cp -r face-detection-5 datasets/face-detection-5
```

## Model Training
### 5. Initialize YOLO Model
```python
yolo_model = YOLO('yolov8n.pt')  # Load pre-trained YOLOv8 model
```

### 6. Train the Model
```python
yolo_model.train(
    data='face-detection-5/data.yaml',
    epochs=100,
    imgsz=448,
    batch=64,
    project='YOLO_weights',
    name='face-detection-5output',
    exist_ok=True
)
```
- `data='face-detection-5/data.yaml'`: Specifies the dataset configuration file.
- `epochs=100`: Number of training iterations.
- `imgsz=448`: Image size for training.
- `batch=64`: Batch size.
- `project='YOLO_weights'`: Directory to save model weights.
- `name='face-detection-5output'`: Output folder for this training session.
- `exist_ok=True`: Overwrites existing output folder if necessary.

## Running Predictions
### 7. Load Trained Model
```python
yolo_model = YOLO('YOLO_weights/face-detection-5output/weights/best.pt')
```

### 8. Run Inference on Test Images
```python
yolo_model.predict(source='face-detection-5/test/images', save=True)
```

## Visualizing Results
### 9. Display Predicted Images
```python
res_path = 'runs/detect/predict'
for image in os.listdir(res_path)[-5:]:
    display(Image(filename=os.path.join(res_path, image), height=500, width=500))
```

## Conclusion
This pipeline trains a YOLOv8 model on a face detection dataset from Roboflow and evaluates it on test images. The trained model can be used for real-time face recognition on various platforms.

