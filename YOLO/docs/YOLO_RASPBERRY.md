# Raspberry Pi YOLO Detection and Facial Recognition

## Overview
This folder contains the main scripts for object detection and facial recognition using a Raspberry Pi with a camera, a button (GPIO2), a servo motor (GPIO18), and an LED (GPIO8). The implementation utilizes the YOLO model for real-time inference.

## Main Scripts

### 1. `RASPI_Detection.py`
This script captures an image using the Raspberry Pi camera and performs object detection using the YOLO model (`yolov8n.pt`). The user interacts with the script by pressing 'o' to capture an image and run inference, or 'q' to quit.

#### Functionality:
- Captures an image from the camera.
- Runs YOLO inference on the captured image.
- Displays the annotated image with detected objects.

#### Usage:
Run the script and enter:
- `'o'` - Captures an image and detects objects.
- `'q'` - Exits the program.

---

### 2. `RASPI_facial_recognition.py`
This script builds on `RASPI_Detection.py`, adding facial recognition using a custom-trained YOLO model (`eu_novo.pt`).

#### Functionality:
- Captures an image when the button is pressed.
- Runs YOLO inference to detect faces.
- If a known face (Class 1) is detected, it unlocks the servo motor.
- If no known face is detected, it performs individual inferences on detected faces.
- Uses an LED to indicate access status.

#### GPIO Connections:
- **Button (GPIO2):** Captures an image on press.
- **Servo (GPIO18):** Acts as a lock, moving to unlock position if access is granted.
- **LED (GPIO8):** Indicates access status.

#### Usage:
- Press the button to capture an image.
- If the script detects a recognized face, the servo unlocks.
- If no recognized face is found, it analyzes each face individually.
- Press `'q'` to quit.

---

## Testing Sub-folder
This folder contains individual test scripts for hardware components:
- **`button.py`** - Tests button functionality.
- **`cam.py`** - Tests camera functionality.
- **`servo.py`** - Tests servo motor movement.

---

## Dependencies
Ensure the following dependencies are installed before running the scripts:
```sh
pip install opencv-python ultralytics picamera2 gpiozero RPi.GPIO
```

---

## Notes
- The `RASPI_facial_recognition.py` script is optimized to minimize processing time by running YOLO inference only when necessary.
- Ensure that the YOLO model weights (`eu_novo.pt`) are in the same directory as the script.
- The camera must be correctly configured on the Raspberry Pi before use.

---

## Future Improvements
- Optimize model inference speed for real-time performance.
- Implement logging for access attempts.
- Improve detection accuracy by fine-tuning the model with more training data.

