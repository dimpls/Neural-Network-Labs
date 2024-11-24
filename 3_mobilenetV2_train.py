import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Создание директории для сохранения графиков
os.makedirs("images_training", exist_ok=True)

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Путь к вашему датасету
data_dir = "Our_training_dataset_new — копия"

# Преобразования для обучающей выборки с аугментацией
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Преобразования для валидационной выборки
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка датасета с использованием ImageFolder
print("Загрузка датасета...")
dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

# Разделение датасета на обучающую и валидационную выборки (80%/20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Применение преобразований для валидационного набора
val_dataset.dataset.transform = transform_val

# Вывод классов
class_names = dataset.classes
print(f"Найдены классы: {class_names}")
print(f"Размер обучающей выборки: {len(train_dataset)} изображений")
print(f"Размер валидационной выборки: {len(val_dataset)} изображений")


# Функции обучения и валидации
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return running_loss / len(loader), accuracy, precision, recall, f1


# Гиперпараметры
num_epochs = 15
batch_size = 64
learning_rate = 0.001

# Инициализация модели MobileNet V2 с нуля
model = models.mobilenet_v2(weights=None)

# Замена последнего слоя
num_classes = len(class_names)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # Количество классов из датасета

model = model.to(device)

# Функция потерь и оптимизатор с weight decay
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Планировщик обучения
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Загрузчики данных
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Тренировка модели
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy, _, _, _ = validate(model, val_loader, criterion)  # Только потери и точность для мониторинга

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Эпоха {epoch+1}/{num_epochs} - Потери на обучении: {train_loss:.4f}, Точность: {train_accuracy:.2f}% - "
          f"Потери на валидации: {val_loss:.4f}, Точность: {val_accuracy:.2f}%")

    # Обновление планировщика
    scheduler.step()

    # Сохранение наилучшей модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()


# Расчет метрик в конце обучения
_, final_accuracy, final_precision, final_recall, final_f1 = validate(model, val_loader, criterion)
print("\nИтоговые метрики на валидационной выборке:")
print(f"  Accuracy: {final_accuracy:.2f}%")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall: {final_recall:.4f}")
print(f"  F1 Score: {final_f1:.4f}")

# Построение графиков
plt.figure(figsize=(12, 5))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Потери на обучении")
plt.plot(val_losses, label="Потери на валидации")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.title("Потери на обучении и валидации")
plt.legend()

# График точности
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Точность на обучении")
plt.plot(val_accuracies, label="Точность на валидации")
plt.xlabel("Эпоха")
plt.ylabel("Точность (%)")
plt.title("Точность на обучении и валидации")
plt.legend()

# Сохранение графика
plt.savefig("images_training/training_results.png")
plt.close()

print("Графики обучения сохранены в 'images_training/training_results.png'")

# Функция валидации с вычислением итоговых метрик
def validate_with_metrics(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Вычисление метрик
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return running_loss / len(loader), accuracy, precision, recall, f1


# После завершения обучения вычисление итоговых метрик
final_loss, final_accuracy, final_precision, final_recall, final_f1 = validate_with_metrics(model, val_loader, criterion)

print("\nИтоговые метрики на валидационной выборке:")
print(f"  Loss: {final_loss:.4f}")
print(f"  Accuracy: {final_accuracy:.2f}%")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall: {final_recall:.4f}")
print(f"  F1 Score: {final_f1:.4f}")
