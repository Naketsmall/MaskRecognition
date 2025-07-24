import cv2
import torch

from config.constants import CLASSIFIER_PATH
from src.classification.mask_classifier import MaskClassifier, load_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '../' + CLASSIFIER_PATH
model = load_model(model_path, device)
classname = model.predict_mask(cv2.cvtColor(cv2.imread('../data/val/with_mask/2.png'), cv2.COLOR_BGR2RGB))
print(classname)

