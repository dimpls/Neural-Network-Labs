import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Убедитесь, что у вас есть доступ к устройству с GPU или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузите модель ResNet50 и настройте её для классификации на 11 классов
model = models.resnet50(pretrained=False)
num_classes = 11
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Загрузите сохранённые веса модели
# model.load_state_dict(torch.load('resnet50_trained_model_newdataset.pth', map_location=device))
model.load_state_dict(torch.load('resnet50_trained_model_newdataset_25.pth', map_location=device))

# Переведите модель в режим оценки
model = model.to(device)
model.eval()

# Преобразования для изображения, аналогичные обучению
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация
])

# Список классов в том же порядке, в каком они использовались при обучении
class_names = ['Birds', 'Cats', 'Dogs', 'Herbivores', 'Horses',
               'Pigs', 'Predators', 'Primates', 'Reptiles', 'Rodents', 'Sea animals']


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


# Функция для предсказания на нескольких изображениях из папки
def predict_images_in_folder(folder_path, model, num_images=10):
    image_files = os.listdir(folder_path)  # Список всех файлов в папке
    #image_files = [f for f in image_files if f.endswith(('.png', '.jpg', '.JPEG'))]  # Фильтруем только изображения

    for i, image_file in enumerate(image_files[:num_images]):  # Берём только первые num_images файлов
        image_path = os.path.join(folder_path, image_file)  # Полный путь к изображению
        image = process_image(image_path)  # Преобразуем изображение

        predicted_class_index = predict(image, model)  # Предсказываем класс
        predicted_class_name = class_names[predicted_class_index]  # Получаем название класса

        print(f"Изображение {i + 1}: {image_file} -> Предсказанный класс: {predicted_class_name}")


# Пример использования: путь к папке с изображениями для предсказания
folder_path = "test_image"
predict_images_in_folder(folder_path, model, num_images=10)


