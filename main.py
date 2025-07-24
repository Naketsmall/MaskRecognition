import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.classification.mask_classifier import load_model
from src.classification.utils import stretch_resize
from src.detection.detector import detect_faces



detector = YOLO('./models/yolov8n-face.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './models/best_model.pth'
classifier = load_model(model_path, device)


def process_image(image: np.ndarray):
    faces = detect_faces(detector, image, 0.5)
    for box in faces.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        status = {
            0: "with_mask",
            1: "without_mask",
            2: "mask_incorrect"
        }
        mask = classifier.predict_mask(stretch_resize(image[y1:y2, x1:x2]))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'face: {box.conf[0]:.2f}, mask: {status[mask]}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

test_name = 'test_hospital.png'
image = cv2.imread(f'./data/input/{test_name}')

image = process_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite(f'./data/output/{test_name}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print('Детекция завершена!')