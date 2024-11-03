import json
import random
import os
import requests
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import shutil
from datetime import datetime
import copy
import re
import pandas as pd
import matplotlib.pyplot as plt

# Убедитесь, что у вас есть доступ к устройству с GPU или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузите модель ResNet50 и настройте её для классификации на 11 классов
model = models.resnet50(pretrained=False)
num_classes = 11
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Загрузите сохранённые веса модели
#model.load_state_dict(torch.load('resnet50_trained_model_newdataset_25.pth', map_location=device))

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


# Load the users.json data:
script_dir = os.path.abspath('')
json_file_path = os.path.join(script_dir, 'users.json')

with open(json_file_path, 'r', encoding='utf-8') as f:
    users = json.load(f)

print(len(users))

# Удаляем пользователей без фото
users_with_photo = [user for user in users if user.get('crop_photo') and user['crop_photo'].get('photo') and user['crop_photo']['photo'].get('sizes')]
# Результат
print(len(users_with_photo))

# Удаляем пользователей без указанных пола и года рождения
filtered_users = [user for user in users_with_photo if user.get('sex') and user.get('bdate') and len(user.get('bdate').split('.')) == 3]
# Результат
print(len(filtered_users))

filtered_users = [user for user in filtered_users if user.get("universities")]

# Список полей, которые нужно оставить
fields_to_keep = ['id', 'sex', 'bdate', 'crop_photo', 'universities']

# Список пользователей с оставшимися полями
users_with_main_fields = [
    {key: user[key] for key in fields_to_keep if key in user}
    for user in filtered_users
]

# Создаем новый список, фильтруя пользователей
filtered_users = []
for user in users_with_main_fields:
    # Проверяем наличие 'universities' и 'faculty_name'
    if user.get("universities") and "faculty_name" in user["universities"][0]:
        # Добавляем faculty_name и удаляем universities
        user["faculty_name"] = user["universities"][0]["faculty_name"]
        user.pop("universities", None)
        filtered_users.append(user)  # Добавляем пользователя в новый список только если faculty_name найден


# Выводим обновленный список пользователей
print(len(filtered_users))

for user in filtered_users:
    user['crop_photo'] = user['crop_photo']['photo']['sizes'][0]['url']
# Выводим обновленный список пользователей
print(len(filtered_users))


# Функция для вычисления возраста
def calculate_age(bdate):
    birth_date = datetime.strptime(bdate, "%d.%m.%Y")
    today = datetime.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

# Обновляем список пользователей
users_with_age = copy.deepcopy(filtered_users)

for user in users_with_age:
    user["age"] = calculate_age(user["bdate"])  # Вычисляем и добавляем возраст
    del user["bdate"]  # Удаляем дату рождения

# Результат
print(len(users_with_age))


# Функция для определения возрастной группы
def determine_age_group(age):
    if 18 <= age <= 25:
        return "young"
    elif age > 25:
        return "adult"
    else:
        return "child"


for user in users_with_age:
    age = user['age']
    user["age_group"] = determine_age_group(age)  # Вычисляем и добавляем возрастную группу

# Названия папок
folders = ['Birds', 'Cats', 'Dogs', 'Herbivores', 'Horses',
           'Pigs', 'Predators', 'Primates', 'Reptiles', 'Rodents', 'Sea animals']

base_path = 'class'  # Замените на ваш путь

# Проходим по каждой папке
for folder in folders:
    folder_path = os.path.join(base_path, folder)

    # Получаем список файлов в папке
    for filename in os.listdir(folder_path):
        file_name_without_ext = os.path.splitext(filename)[0]  # Убираем расширение

        # Находим словарь, где id совпадает с именем файла
        for item in users_with_age:
            if item.get("id") == file_name_without_ext:
                item["class"] = folder  # Добавляем новое поле с названием папки

print(users_with_age[0])

# Извлечение всех названий факультетов
faculty_names = [user["faculty_name"] for user in filtered_users]

# Сохранение названий факультетов в файл .txt
file_path = 'faculty_names.txt'
with open(file_path, 'w', encoding='utf-8') as f:
    for name in faculty_names:
        f.write(name + '\n')
# Вывод списка факультетов
print(faculty_names)

# Регулярные выражения для классификации
technicheskie = r'(?i)(инженер|технолог|физик|математик|машино|радио|строитель|информат|геолог|авто|физико|атомно|педагог|агро|хим|физико|институт\s|кафедра\s|систем)'
gumanitarnye = r'(?i)(гуманитар|истор|психолог|социаль|педагог|юрид|прав|филолог|перевод|язык|истор|литератур|правов|туризм|менеджмент|бизнес|финанс|юриспруденц|психол|туризм|истори)'
estestvenno_nauchnye = r'(?i)(био|гео|химик|естественно|педиатр|медиц|биоинжен|географ|пищев|сельскохозяйств|эколог|геол|педагог)'

# Фильтр русских факультетов
russian_filter = r'^[А-Яа-яёЁ\s]+$'

# Классификация факультетов
tech, hum, sci = [], [], []
for faculty in faculty_names:
    if not re.match(russian_filter, faculty):
        continue
    if re.search(technicheskie, faculty):
        tech.append(faculty)
    elif re.search(gumanitarnye, faculty):
        hum.append(faculty)
    elif re.search(estestvenno_nauchnye, faculty):
        sci.append(faculty)

# Результат
print("Технические факультеты:", len(tech))
print("Гуманитарные факультеты:", len(hum))
print("Естественно-научные факультеты:", len(sci))


# Функция для классификации факультета
def classify_faculty(user):
    faculty = user.get('faculty_name', '')

    if not re.match(russian_filter, faculty):
        return None  # Исключаем нерусские названия

    if re.search(technicheskie, faculty):
        user['faculty_type'] = 'technical'
    elif re.search(gumanitarnye, faculty):
        user['faculty_type'] = 'humanitarian'
    elif re.search(estestvenno_nauchnye, faculty):
        user['faculty_type'] = 'natural_sciences'
    else:
        return None  # Удаляем, если не подходит ни одна категория

    return user


# Применяем классификацию ко всем элементам и удаляем неподходящие
filtered_users = [classify_faculty(user) for user in users_with_age]
filtered_users = [user for user in filtered_users if user is not None]

# Результат
print(filtered_users[0])
print(len(filtered_users))

df = pd.DataFrame(filtered_users)
file_path = 'filtered_users_dataframe.csv'
df.to_csv(file_path, index=False)
# Делаем графики более презентабельными
plt.style.use('ggplot')

# Построение диаграммы для 'class'
plt.figure(figsize=(12, 7))
df['class'].value_counts().plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title("Distribution by Class", fontsize=16, weight='bold')
plt.xlabel("Class", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Построение диаграммы для 'faculty_type'
plt.figure(figsize=(12, 7))
df['faculty_type'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title("Distribution by Faculty Type", fontsize=16, weight='bold')
plt.xlabel("Faculty Type", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Построение диаграммы для 'age_group'
plt.figure(figsize=(12, 7))
df['age_group'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Distribution by Age Group", fontsize=16, weight='bold')
plt.xlabel("Age Group", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Фильтрация молодых пользователей
young_users = df[df['age_group'] == 'young']

# Распределение по 'class' среди молодых пользователей
plt.figure(figsize=(12, 7))
young_users['class'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution by Class for Young Adult Age Group", fontsize=16, weight='bold')
plt.xlabel("Class", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Разделение данных по полу для молодых пользователей
young_men = young_users[young_users['sex'] == 1]
young_women = young_users[young_users['sex'] == 2]

# Построение диаграммы для молодых мужчин, если есть данные
if not young_men.empty:
    plt.figure(figsize=(12, 7))
    young_men['class'].value_counts().plot(kind='bar', color='dodgerblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution by Class for Young Adult Age Group - Men", fontsize=16, weight='bold')
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Нет данных для молодых мужчин в группе 'young_adult'.")

# Построение диаграммы для молодых женщин, если есть данные
if not young_women.empty:
    plt.figure(figsize=(12, 7))
    young_women['class'].value_counts().plot(kind='bar', color='lightpink', edgecolor='black', alpha=0.7)
    plt.title("Distribution by Class for Young Adult Age Group - Women", fontsize=16, weight='bold')
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Нет данных для молодых женщин в группе 'young_adult'.")
