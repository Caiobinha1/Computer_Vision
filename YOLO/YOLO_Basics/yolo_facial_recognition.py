#This code is the benchmark por the raspberry pi application
import cv2
import time
from ultralytics import YOLO

def main():
    model = YOLO('eu_11.pt')

    url = 0

    cap = cv2.VideoCapture(0)

    # Interval to take pictures in seconds
    interval = 0 
    last_capture_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            current_time = time.time()

            # Check if it's time to capture the next frame
            if current_time - last_capture_time >= interval:
                last_capture_time = current_time  # Update the last capture time

                results = model.track(frame, conf=0.5, save=False, tracker="bytetrack.yaml")  # Run YOLO inference on the frame

                annotated_frame = results[0].plot()

                boxes = results[0].boxes.cpu().numpy()  # Get the coordinates of each box
                boxes_classes = boxes.cls  # Get the class IDs for each box

                track_ids = results[0].boxes.id.int().cpu().tolist() #lista alinhada com as classes (boxes_classes), lembrar que sao organizados de acordo com as probabilidades

                print(boxes_classes, track_ids)
                
                count = 0
                count_true =0
                for ids in boxes_classes:  # Iterate through the detected class IDs
                    if ids == 0:  # Check if the detected object is a person (class ID 0)
                        count += 1
                    elif (ids == 1):
                        count_true = 1
                if count == 0 and count_true ==0:
                    del frame
                elif (count_true ==1):
                    print("Seja bem vindo!")
                else:
                    print(f'{count} Pessoas estao na porta')
                    filename = f'png_seguranca//frame_{int(current_time)}.png'
                    # cv2.imwrite(filename, annotated_frame)    #uncomment if you want to save the images

                # Efficiently show the frame
                cv2.imshow("YOLOv8 inference", annotated_frame)

                # Release frames to free memory
                # del frame
                del results
                del annotated_frame

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
