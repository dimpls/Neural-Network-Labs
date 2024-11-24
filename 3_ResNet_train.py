import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Путь к данным
data_dir = 'Our_training_dataset_new — копия'

# Преобразования для изображений
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

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")
if device.type == 'cuda':
    print(f"Доступное GPU: {torch.cuda.get_device_name(0)}")

# Модель
model = models.resnet152(weights=None)
model.fc = nn.Linear(model.fc.in_features, 14)
model = model.to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Функция для обучения или валидации
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    epoch_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for inputs, labels in tqdm(loader, desc="Обучение" if is_train else "Валидация"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = epoch_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# Функция для расчета метрик на данных
def calculate_metrics(model, loader):
    model.eval()  # Переводим модель в режим оценки
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Подсчет метрик"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Расчет метрик
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

# Обучение модели с расчетом точности
def train_model_with_accuracy(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    train_loss_history, val_loss_history = [], []
    train_accuracy_history, val_accuracy_history = [], []

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        start_time = time.time()

        # Тренировка
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        train_accuracy, _, _, _ = calculate_metrics(model, train_loader)

        # Валидация
        val_loss, _ = run_epoch(model, val_loader, criterion)
        val_accuracy, _, _, _ = calculate_metrics(model, val_loader)

        # Сохранение истории
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)

        print(f"Эпоха {epoch + 1} завершена за {time.time() - start_time:.2f} секунд")
        print(f"  Потери: Обучение={train_loss:.4f}, Валидация={val_loss:.4f}")
        print(f"  Точность: Обучение={train_accuracy * 100:.2f}%, Валидация={val_accuracy * 100:.2f}%")

    return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

# Запуск обучения
print("Запуск обучения модели...")
num_epochs = 15
train_loss, val_loss, train_accuracy, val_accuracy = train_model_with_accuracy(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs
)

# Построение графиков потерь
epochs = range(1, num_epochs + 1)

plt.figure()
plt.plot(epochs, train_loss, label="Потери на обучении")
plt.plot(epochs, val_loss, label="Потери на валидации")
plt.xlabel("Эпохи")
plt.ylabel("Потери")
plt.legend()
plt.title("График потерь")
plt.grid()
plt.show()

# Построение графика точности
plt.figure()
plt.plot(epochs, train_accuracy, label="Точность на обучении")
plt.plot(epochs, val_accuracy, label="Точность на валидации")
plt.xlabel("Эпохи")
plt.ylabel("Точность")
plt.legend()
plt.title("График точности")
plt.grid()
plt.show()

# Расчет и вывод итоговых метрик
print("\nПодсчет итоговых метрик...")

# Метрики на обучающей выборке
train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(model, train_loader)

# Метрики на валидационной выборке
val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(model, val_loader)

print("\nМетрики на обучающей выборке:")
print(f"  Accuracy: {train_accuracy * 100:.2f}%")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall: {train_recall:.4f}")
print(f"  F1 Score: {train_f1:.4f}")

print("\nМетрики на валидационной выборке:")
print(f"  Accuracy: {val_accuracy * 100:.2f}%")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall: {val_recall:.4f}")
print(f"  F1 Score: {val_f1:.4f}")
