import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Задание пути к данным
data_dir = 'Our_training_dataset'

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

# Проверка на наличие GPU и перевод модели на устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка предобученной модели EfficientNet-B4
from torchvision.models import efficientnet_b4

model = efficientnet_b4(pretrained=True)
num_classes = 10

# Добавление Dropout и изменение классификатора
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),  # Увеличение Dropout
    nn.Linear(model.classifier[1].in_features, num_classes)
)

model = model.to(device)

# Функция потерь
criterion = nn.CrossEntropyLoss()

# Оптимизатор и планировщик уменьшения скорости обучения
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

num_epochs = 10


# Модифицированная функция для обучения модели с сохранением потерь и точности на каждой эпохе
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    model.train()
    training_loss = []
    validation_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Обучение
        for i, (inputs, labels) in enumerate(train_loader, 1):
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

            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Среднее значение потерь и точности на тренировочной выборке за эпоху
        epoch_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train
        training_loss.append(epoch_loss)
        train_accuracy.append(epoch_train_accuracy)

        # Расчет потерь и точности на валидационной выборке
        val_loss, val_acc = test_model_epoch(model, val_loader, criterion)
        validation_loss.append(val_loss)
        val_accuracy.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    print("Обучение завершено!")
    torch.save(model.state_dict(), 'efficientnet_b4_trained_model_10epoch.pth')
    print("Модель сохранена как 'efficientnet_b4_trained_model.pth'")

    return training_loss, validation_loss, train_accuracy, val_accuracy


# Вспомогательная функция для оценки потерь и точности на валидационной выборке за каждую эпоху
def test_model_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
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


# Функция для тестирования модели с расчетом метрик после обучения
def test_model_final(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            print(f"Validation Step [{i}/{len(val_loader)}] completed")

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", conf_matrix)

    return accuracy, precision, recall, f1, conf_matrix


# Функция для построения графиков потерь и точности
def plot_metrics(train_loss, val_loss, train_accuracy, val_accuracy):
    # График потерь
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.show()

    # График точности
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label="Train Accuracy")
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy per Epoch")
    plt.legend()
    plt.show()


# Обучаем модель и получаем данные о потерях и точности
training_loss, validation_loss, train_accuracy, val_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Строим графики потерь и точности на обучающей и валидационной выборках по эпохам
plot_metrics(training_loss, validation_loss, train_accuracy, val_accuracy)

# Тестируем модель на валидационном наборе после обучения и выводим метрики
accuracy, precision, recall, f1, conf_matrix = test_model_final(model, val_loader)
