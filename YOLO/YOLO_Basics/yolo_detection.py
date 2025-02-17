import cv2
from ultralytics import YOLO

# Load the YOLO model (smallest version for speed)
model = YOLO("yolov8n.pt")  # Uses a pre-trained model for general object detection

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()  # Read frame from webcam
    if not success:
        break  # Exit loop if no frame is captured

    # Run YOLO inference
    results = model(frame)

    # Get detection details
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = model.names[class_id]  # Class label

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
