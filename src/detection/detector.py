from ultralytics import YOLO
import numpy as np

def detect_faces(model: YOLO, image: np.ndarray, conf=0.5):
    faces = model.predict(image, conf=conf)[0]
    return faces