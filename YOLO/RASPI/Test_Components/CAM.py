from picamera2 import Picamera2
import cv2
#from ultralytics import YOLO
picam2 =Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
picam2.capture_file("test3.jpg")
#img =picam2.capture_array()
picam2.stop()
image_path="test3.jpg"
image =cv2.imread(image_path)
img_resized = cv2.resize(image,(640,480))
#img_bgr=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
print(image.shape)
cv2.imshow("teste", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


