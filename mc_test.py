from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
import torch

from src.classification.mask_classifier import MaskClassifier, train_epoch
from src.dataset_loader import MaskDataset

import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MaskClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MaskDataset('./data/Dataset', transform)
print(f"Всего изображений в датасете: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

save_dir = './models/'
num_epochs = 20
for epoch in range(num_epochs):
    train_epoch(model, dataloader, optimizer, criterion, device)
    torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))

