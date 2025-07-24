import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import os

from src.classification.mask_classifier import MaskClassifier
from src.dataset_loader import MaskDataset, basic_transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaskClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

transform = basic_transformer()


train_dataset = MaskDataset('../../data/train', transform=transform)
val_dataset = MaskDataset('../../data/val', transform=transform)

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
save_dir = '../../models'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})

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


# 5. Загрузка лучшей модели (пример)
def load_best_model():
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model = MaskClassifier().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


#best_model = load_best_model()