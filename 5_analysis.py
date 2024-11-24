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
folders = ['Birds', 'Cats', 'Cow', 'Dogs', 'Elephant', 'Giraffe', 'Other', 'People', 'Predators',
           'Primates', 'Rabbit', 'Reptiles', 'Sheep', 'Zebra']

base_path = 'downloaded_images_from_VK — копия/sorted_images'  # Замените на ваш путь

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
# Делаем графики более понятными
plt.style.use('ggplot')

# Построение диаграммы распределения по 'class' (класс)
plt.figure(figsize=(12, 7))
df['class'].value_counts().plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title("Распределение пользователей по классам после фильтрации", fontsize=16, weight='bold')
plt.xlabel("Класс", fontsize=14)
plt.ylabel("Количество", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Построение диаграммы распределения по типу факультетов 'faculty_type'
plt.figure(figsize=(12, 7))
df['faculty_type'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title("Распределение по типу факультетов", fontsize=16, weight='bold')
plt.xlabel("Тип факультета", fontsize=14)
plt.ylabel("Количество", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Построение диаграммы распределения по возрастным группам 'age_group'
plt.figure(figsize=(12, 7))
df['age_group'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
plt.title("Распределение по возрастным группам", fontsize=16, weight='bold')
plt.xlabel("Возрастная группа", fontsize=14)
plt.ylabel("Количество", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Фильтрация молодых пользователей
young_users = df[df['age_group'] == 'young']

# Распределение по 'class' среди молодых пользователей
plt.figure(figsize=(12, 7))
young_users['class'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Распределение по классам (молодежь)", fontsize=16, weight='bold')
plt.xlabel("Класс", fontsize=14)
plt.ylabel("Количество", fontsize=14)
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
    plt.title("Распределение по классам (молодежь, мужчины)", fontsize=16, weight='bold')
    plt.xlabel("Класс", fontsize=14)
    plt.ylabel("Количество", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Нет данных для молодых мужчин в группе 'молодежь'.")

# Построение диаграммы для молодых женщин, если есть данные
if not young_women.empty:
    plt.figure(figsize=(12, 7))
    young_women['class'].value_counts().plot(kind='bar', color='lightpink', edgecolor='black', alpha=0.7)
    plt.title("Распределение по классам (молодежь, женщины)", fontsize=16, weight='bold')
    plt.xlabel("Класс", fontsize=14)
    plt.ylabel("Количество", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Нет данных для молодых женщин в группе 'молодежь'.")

# Путь к csv файлу и базовой папке с изображениями
csv_file_path = "filtered_users_dataframe.csv"
base_path = "downloaded_images_from_VK — копия/sorted_images"

# Загрузка данных из CSV
df = pd.read_csv(csv_file_path)


# Функция для поиска фотографий пользователя по ID
def find_user_photos(user_id, user_class):
    user_photos = []
    class_folder = os.path.join(base_path, user_class)
    if os.path.isdir(class_folder):
        # Ищем файлы с именем, начинающимся на ID пользователя
        files = [f for f in os.listdir(class_folder) if f.startswith(str(user_id))]
        for file in files:
            user_photos.append(os.path.join(class_folder, file))
    return user_photos


# Группировка пользователей по классам
grouped = df.groupby('class')


# Вывод до 2 фотографий для каждого класса
for user_class, group in grouped:
    if pd.isna(user_class):  # Если класс не указан, пропускаем
        continue

    print(f"Класс: {user_class}")  # Заголовок для класса

    # Список пользователей в классе
    user_ids = group['id'].unique()
    photos_shown = 0  # Счетчик фотографий

    # Перебираем пользователей
    for user_id in user_ids:
        # Ищем фотографии пользователя
        user_photos = find_user_photos(user_id, user_class)

        # Если есть фотографии, выводим их
        for photo_path in user_photos[:2]:  # Берем до 2 фотографий
            img = Image.open(photo_path)
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(f"Класс: {user_class}\nID: {user_id}", fontsize=12, weight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            photos_shown += 1

            if photos_shown >= 5:  # Ограничение на 2 фотографии на класс
                break

        if photos_shown >= 5:
            break


