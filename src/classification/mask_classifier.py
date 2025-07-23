import cv2
import numpy as np


def classify_mask(face_roi, brightness_threshold=200, saturation_threshold=30):
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    # Диапазон для белого:
    # - Hue (H): 0-180 (не важен для белого)
    # - Saturation (S): 0-30 (чем меньше, тем "белее")
    # - Value (V): > brightness_threshold (яркость)

    lower_white = np.array([0, 0, brightness_threshold])
    upper_white = np.array([180, saturation_threshold, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_pixels = cv2.countNonZero(white_mask)
    total_pixels = face_roi.shape[0] * face_roi.shape[1]
    return (white_pixels / total_pixels) > 0.25