import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: Размножить датасет зашумлением данных

class MaskClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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