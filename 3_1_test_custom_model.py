import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split

# Задание пути к данным
data_dir = 'Our_training_dataset'

# Преобразования для предобработки изображений с усиленной аугментацией
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Загрузка всех данных с аугментацией для тренировочного набора
dataset = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms['train'])

# Получаем индексы и метки всех данных
indices = list(range(len(dataset)))
labels = [label for _, label in dataset.samples]

# Стратифицированное разделение индексов на тренировочный и валидационный наборы (80% на 20%)
train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

# Создание поднаборов с использованием индексов
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Устанавливаем преобразования для валидационного набора
val_dataset.dataset.transform = data_transforms['val']

# DataLoader для загрузки данных
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Проверка на наличие GPU и перевод модели на устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Определение улучшенной модели CNN
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = self.pool(self.bn4(torch.relu(self.conv4(x))))

        x = self.avgpool(x)
        x = x.view(-1, 512)  # Flatten

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Создаем модель и перемещаем её на устройство
model = ImprovedCNN(num_classes=10).to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Создание директории для сохранения данных
output_dir = 'custom_model'
os.makedirs(output_dir, exist_ok=True)

# Генерация имени файла на основе количества эпох
num_epochs = 30
patience = 3  # Количество эпох для ранней остановки
log_filename = f"training_log_{num_epochs}_epochs_4.txt"
log_filepath = os.path.join(output_dir, log_filename)


# Реализация Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Learning Rate Scheduler (ReduceLROnPlateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)


# Функция для обучения модели с Early Stopping и записью в файл
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=5):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    early_stop = EarlyStopping(patience=patience)

    # Открытие файла для записи данных
    with open(log_filepath, 'w') as log_file:
        log_file.write("Epoch\tTraining Loss\tTraining Accuracy\tValidation Loss\tValidation Accuracy\n")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

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

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct_train / total_train
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            # Валидация
            model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_epoch_loss = val_running_loss / len(val_loader)
            val_epoch_acc = correct_val / total_val
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            # Запись данных об эпохе в файл
            log_file.write(
                f"{epoch + 1}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{val_epoch_loss:.4f}\t{val_epoch_acc:.4f}\n")
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Acc: {epoch_acc:.4f}, "
                  f"Validation Loss: {val_epoch_loss:.4f}, Validation Acc: {val_epoch_acc:.4f}")

            # Проверка на раннюю остановку
            early_stop(val_epoch_loss)
            if early_stop.early_stop:
                print("Ранняя остановка сработала.")
                break

            # Обновляем learning rate с помощью scheduler
            scheduler.step(val_epoch_loss)

    # Сохранение модели
    model_filename = f"model_epoch_{num_epochs}_patience_{patience}.pth"
    model_filepath = os.path.join(output_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)  # Сохраняем только веса модели
    print(f"Модель сохранена в {model_filepath}")

    return train_loss, val_loss, train_acc, val_acc


# Обучение модели
train_loss, val_loss, train_acc, val_acc = train_model_with_early_stopping(model, train_loader, val_loader, criterion,
                                                                           optimizer, num_epochs=num_epochs,
                                                                           patience=patience)


# Функция для оценки модели и расчета метрик с записью в файл
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Рассчитываем метрики
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    # Записываем метрики в файл
    with open(log_filepath, 'a') as log_file:
        log_file.write("\nValidation Metrics:\n")
        log_file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        log_file.write(f"Precision: {precision:.4f}\n")
        log_file.write(f"Recall: {recall:.4f}\n")
        log_file.write(f"F1 Score: {f1:.4f}\n")
        log_file.write("\nClassification Report:\n" + class_report + "\n")
        log_file.write("\nConfusion Matrix:\n" + str(conf_matrix) + "\n")

    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", class_report)
    print("\nConfusion Matrix:\n", conf_matrix)

    return accuracy, precision, recall, f1, conf_matrix


# Оценка модели
accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, val_loader)


# Построение графиков потерь и точности
def plot_metrics(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(14, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy per Epoch")
    plt.legend()

    plt.show()


# Построение графиков
plot_metrics(train_loss, val_loss, train_acc, val_acc)
