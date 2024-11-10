import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import shutil

# Убедитесь, что у вас есть доступ к устройству с GPU или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Определение улучшенной модели (та же, что и во время обучения)
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

# Создаем модель с теми же параметрами, что и во время обучения
model = ImprovedCNN(num_classes=10).to(device)

# Загружаем сохранённые веса модели
model.load_state_dict(torch.load('custom_model/model_epoch_30_patience_3.pth', map_location=device))  # Путь к файлу модели

# Переводим модель в режим оценки
model.eval()

# Преобразования для изображения, аналогичные тем, что использовались при обучении
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация
])

# Список классов в том же порядке, в каком они использовались при обучении
class_names = ['Birds', 'Cats', 'Dogs', 'Herbivores', 'Horses',
               'Other', 'Predators', 'Primates', 'Reptiles', 'Sea animals']

# Функция для загрузки и предобработки одного изображения
def process_image(image_path):
    image = Image.open(image_path)  # Открытие изображения
    image = data_transforms(image)  # Применение преобразований
    image = image.unsqueeze(0)  # Добавление дополнительной размерности для батча
    return image

# Функция для предсказания
def predict(image, model):
    image = image.to(device)  # Переносим изображение на устройство (CPU или GPU)

    with torch.no_grad():  # Отключаем градиенты для режима оценки
        outputs = model(image)  # Прямой проход модели
        _, preds = torch.max(outputs, 1)  # Получаем индекс класса с максимальной вероятностью

    return preds.item()  # Возвращаем индекс предсказанного класса

# Функция для создания папок и перемещения изображений
def create_folders_and_move_images(folder_path, model):
    image_files = os.listdir(folder_path)  # Список всех файлов в папке
    image_files = [f for f in image_files if f.endswith(('.png', '.jpg', '.JPEG'))]  # Фильтруем только изображения

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)  # Полный путь к изображению
        try:
            image = process_image(image_path)  # Преобразуем изображение

            predicted_class_index = predict(image, model)  # Предсказываем класс
            predicted_class_name = class_names[predicted_class_index]  # Получаем название класса

            # Создаем папку для класса, если она не существует
            class_folder_path = os.path.join(folder_path, predicted_class_name)
            if not os.path.exists(class_folder_path):
                os.makedirs(class_folder_path)

            # Перемещаем изображение в папку класса
            shutil.move(image_path, os.path.join(class_folder_path, image_file))

            print(f"Изображение {image_file} -> Предсказанный класс: {predicted_class_name}")
        except RuntimeError as e:
            print(f"Ошибка при обработке изображения {image_file}: {e}")
            continue

# Пример использования: путь к папке с изображениями для предсказания
folder_path = "downloaded_images_from_VK — копия"  # Замените на путь к вашей папке
create_folders_and_move_images(folder_path, model)
