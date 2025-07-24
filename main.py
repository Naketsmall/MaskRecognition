import cv2
import numpy as np
from ultralytics import YOLO

from src.classification.mask_classifier import classify_mask
from src.detection.detector import detect_faces

model = YOLO('./models/yolov8n-face.pt')


def process_image(image: np.ndarray):
    faces = detect_faces(model, image, 0.5)
    for box in faces.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'face: {box.conf[0]:.2f}, mask: {classify_mask(image[y1:y2, x1:x2])}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

test_name = 'test_tolpa.jpg'
image = cv2.imread(f'./data/input/{test_name}')

process_image(image)
cv2.imwrite(f'./data/output/{test_name}', image)
print('Детекция завершена!')