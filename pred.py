from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO('path/to/your/model.pt')

# Function to process image
def process_image(image):
    results = model(image)
    
    # Process results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

# For image
image = cv2.imread('path/to/your/image.jpg')
processed_image = process_image(image)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# For video
video = cv2.VideoCapture('path/to/your/video.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    processed_frame = process_image(frame)
    cv2.imshow('Processed Video', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
