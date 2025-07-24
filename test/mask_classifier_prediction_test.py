import cv2
import torch
from PIL import Image

from src.classification.mask_classifier import MaskClassifier, load_model
from src.dataset_loader import basic_transformer




"""
def predict_mask(image_path, classifier, device='cuda'):
    ""
    Предсказывает, есть ли маска на изображении

    Аргументы:
        image_path (str): Путь к изображению
        model_path (str): Путь к сохраненной модели
        device (str): Устройство для вычислений ('cuda' или 'cpu')

    Возвращает:
        str: "with_mask", "without_mask" или "mask_incorrect"
    ""
    classifier.eval()

    transform = basic_transformer()

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        output = classifier(image_tensor)
        _, predicted = torch.max(output, 1)


    class_names = ['with_mask', 'without_mask', 'mask_incorrect']
    return class_names[predicted.item()]
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '../models/best_model.pth'
model = load_model(model_path, device)
classname = model.predict_mask(cv2.cvtColor(cv2.imread('../data/val/with_mask/2.png'), cv2.COLOR_BGR2RGB))
print(classname)

