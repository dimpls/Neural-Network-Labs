import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Задание пути к данным
data_dir = 'Classes'

# Преобразования для предобработки изображений
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Приведение изображений к размеру 224x224
        transforms.RandomHorizontalFlip(),  # Аугментация данных — случайное горизонтальное отражение
        transforms.ToTensor(),  # Преобразование в тензор
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # Приведение изображений к размеру 224x224
        transforms.ToTensor(),  # Преобразование в тензор
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация
    ]),
}

# Загрузка данных
dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms['train'])

# Разделение данных на тренировочные и валидационные наборы (80% на 20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Применяем валидационные преобразования для валидационного набора
val_dataset.dataset.transform = data_transforms['val']

# DataLoader для загрузки данных
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Проверка на наличие GPU и перевод модели на устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
# Загрузка предобученной модели ResNet50
model = models.resnet50(pretrained=True)
# model = models.resnet101(pretrained=True)
# Адаптация модели для классификации (теперь 11 классов, с учетом нового класса Primates)
num_classes = 11  # Изменено с 10 на 11
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Перевод модели на устройство
model = model.to(device)

# Функция потерь
criterion = nn.CrossEntropyLoss()

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Количество эпох для обучения
num_epochs = 25


# Функция для обучения модели
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # Переводим модель в режим тренировки
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход (forward)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход (backward) и оптимизация
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Выводим информацию о шаге
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Выводим среднее значение потерь после каждой эпохи
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed with Average Loss: {running_loss / len(train_loader):.4f}")

    print("Обучение завершено!")

    # Сохраняем модель после завершения обучения
    torch.save(model.state_dict(), 'resnet50_trained_model_newdataset_25.pth')
    print("Модель сохранена как 'resnet50_trained_model.pth'")


# Функция для тестирования модели с расчетом метрик
def test_model(model, val_loader):
    model.eval()  # Переводим модель в режим оценки
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Отключаем градиенты для режима оценки
        for i, (inputs, labels) in enumerate(val_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Выводим информацию о шаге инференса
            print(f"Validation Step [{i}/{len(val_loader)}] completed")

    # Рассчитываем метрики
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Используем средневзвешенную метрику F1
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Выводим метрики
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")


# Обучаем модель
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

# Тестируем модель на валидационном наборе
test_model(model, val_loader)
