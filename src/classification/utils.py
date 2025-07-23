import cv2
import numpy as np

def stretch_resize(face_roi: np.ndarray, size=128):
    return cv2.resize(face_roi, (size, size))