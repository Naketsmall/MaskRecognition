import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import os

from config.constants import TRAIN_DATASET_PATH, VAL_DATASET_PATH, MODELS_PATH
from src.classification.mask_classifier import MaskClassifier, train_epoch
from src.dataset_loader import MaskDataset, basic_transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaskClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

transform = basic_transformer()


train_dataset = MaskDataset('../../' + TRAIN_DATASET_PATH, transform=transform)
val_dataset = MaskDataset('../../' + VAL_DATASET_PATH, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


def validate(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(val_loader), 100. * correct / total



num_epochs = 20
best_val_acc = 0.0
save_dir = '../../' + MODELS_PATH
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    train_epoch(model, train_loader, optimizer, criterion, device)

    # Валидация
    val_loss, val_acc = validate(model, val_loader)
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    # Сохранение лучшей модели
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'acc': val_acc
        }, os.path.join(save_dir, 'best_model.pth'))
        print(f'New best model saved with accuracy {val_acc:.2f}%')

    torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))

