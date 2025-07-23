import os

import torch
from torchvision import transforms

from PIL import Image

from src.classification.mask_classifier import MaskClassifier


def load_model(path, device):
    checkpoint = torch.load(path)
    model = MaskClassifier().to(device)
    model.load_state_dict(torch.load(path))
    return model


def predict_mask(image_path, model_path, device='cuda'):
    """
    Предсказывает, есть ли маска на изображении

    Аргументы:
        image_path (str): Путь к изображению
        model_path (str): Путь к сохраненной модели
        device (str): Устройство для вычислений ('cuda' или 'cpu')

    Возвращает:
        str: "with_mask", "without_mask" или "mask_incorrect"
    """
    model = load_model(model_path, device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)


    class_names = ['with_mask', 'without_mask', 'mask_incorrect']
    return class_names[predicted.item()]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '../models/epoch_1.pth'
classname = predict_mask('../data/Dataset/without_mask/2.png', model_path, device)
print(classname)

