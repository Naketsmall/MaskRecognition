import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config.constants import FACE_ROI_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD, NUM_STATUS


def basic_transformer():
    return transforms.Compose([
        transforms.Resize(FACE_ROI_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])


class MaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Путь к корневой директории с папками классов
            transform (callable, optional): Трансформации для изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = NUM_STATUS.values()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # Всегда работаем с RGB

        if self.transform:
            image = self.transform(image)

        return image, label
