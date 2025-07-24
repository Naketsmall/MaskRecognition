import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from tqdm import tqdm

from src.dataset_loader import basic_transformer


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

    @property
    def device(self):
        return next(self.parameters()).device

    def predict_mask(self, image: np.ndarray):
        """
        Предсказывает, есть ли маска на изображении

        Аргументы:
            image (np.ndarray): Изображение в формате RGB

        Возвращает:
            int: 0:"with_mask", 1:"without_mask" или 2:"mask_incorrect"
        """
        self.eval()

        transform = basic_transformer()

        image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(self.device)


        with torch.no_grad():
            output = self(image_tensor)
            _, predicted = torch.max(output, 1)


        #class_names = ['with_mask', 'without_mask', 'mask_incorrect']
        #return class_names[predicted.item()]
        return predicted.item()


def train_epoch(model, dataloader, optimizer, criterion, device='cuda'):
    """
    Обучение модели на одной эпохе

    Аргументы:
        model (nn.Module): Модель для обучения (MaskClassifier)
        dataloader (DataLoader): Загрузчик данных
        optimizer: Оптимизатор (Adam/SGD)
        criterion: Функция потерь (CrossEntropyLoss)
        device (str): Устройство для вычислений ('cuda' или 'cpu')

    Возвращает:
        float: Среднее значение функции потерь на эпохе
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for batch_idx, (inputs, labels) in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
    return epoch_loss

def load_model(path, device):
    model = MaskClassifier().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    return model

"""
# Простой способ через проверку, заполнена ли 1/4 площади белым

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
"""