import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product
import json
from torchvision.models import efficientnet_b4

# Задание пути к данным и другие настройки
data_dir = 'Our_training_dataset'
num_classes = 10
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Преобразования для предобработки изображений
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Загрузка данных
dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms['train'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Гиперпараметры для перебора
hyperparams = {
    'lr': [0.001, 0.0005],
    'weight_decay': [1e-5, 1e-4],
    'dropout': [0.3, 0.5],
}


# Модифицированная функция для обучения модели с возвращением результатов
def train_and_evaluate(lr, weight_decay, dropout, num_epochs=2):
    print(f"Начало обучения модели с параметрами: lr={lr}, weight_decay={weight_decay}, dropout={dropout}")

    model = efficientnet_b4(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    training_loss, validation_loss = [], []
    train_accuracy, val_accuracy = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        print(f"  Эпоха {epoch + 1}/{num_epochs} началась.")

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train
        training_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)

        val_loss, val_acc = test_model_epoch(model, val_loader, criterion)
        validation_loss.append(val_loss)
        val_accuracy.append(val_acc)

        print(
            f"  Эпоха {epoch + 1} завершена: Train Loss={epoch_train_loss:.4f}, Train Accuracy={epoch_train_accuracy:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Accuracy={val_acc:.4f}")

    print(f"Обучение модели с параметрами lr={lr}, weight_decay={weight_decay}, dropout={dropout} завершено.")
    return training_loss, validation_loss, train_accuracy, val_accuracy, val_accuracy[-1], model


# Функция для оценки на каждой эпохе
def test_model_epoch(model, val_loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(val_loader), correct / total


# Обучение и вывод результатов для гиперпараметров
results = []
for lr, weight_decay, dropout in product(hyperparams['lr'], hyperparams['weight_decay'], hyperparams['dropout']):
    print(f"\nЗапуск комбинации гиперпараметров: lr={lr}, weight_decay={weight_decay}, dropout={dropout}")
    train_loss, val_loss, train_acc, val_acc, final_acc, model = train_and_evaluate(lr, weight_decay, dropout,
                                                                                    num_epochs=3)

    model_filename = f"model_lr{lr}_wd{weight_decay}_do{dropout}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"  Модель сохранена как '{model_filename}'")

    results.append((final_acc, (lr, weight_decay, dropout), model, train_loss, val_loss, train_acc, val_acc))

# Сортируем и выводим лучшие комбинации
results = sorted(results, key=lambda x: x[0], reverse=True)[:5]
print("\nТоп-5 лучших комбинаций гиперпараметров:")

# Построение графиков для лучших моделей
for rank, (accuracy, params, model, train_loss, val_loss, train_acc, val_acc) in enumerate(results, 1):
    lr, weight_decay, dropout = params
    print(
        f"\nМодель {rank} - lr={lr}, weight_decay={weight_decay}, dropout={dropout} - Accuracy: {accuracy * 100:.2f}%")

    # График потерь
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss per Epoch for lr={lr}, wd={weight_decay}, dropout={dropout}")
    plt.legend()
    plt.show()

    # График точности
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy")
    plt.plot(range(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per Epoch for lr={lr}, wd={weight_decay}, dropout={dropout}")
    plt.legend()
    plt.show()
