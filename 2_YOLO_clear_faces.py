import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
print(torch.cuda.is_available())


def detect_non_animal_objects(input_folder, output_folder, confidence_threshold=0.5):
    # Создаем YOLO модель, принудительно используя CPU
    model = YOLO('yolo11x.pt')
    model.to('cuda')

    # Список классов животных, которые мы будем исключать
    animal_classes = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}
    # Убедитесь, что папка вывода существует
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Перебираем все файлы в папке
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Проверяем, является ли файл изображением
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Загружаем изображение и выполняем детекцию
            results = model(file_path)

            # Проверяем, есть ли в изображении объекты, не относящиеся к животным
            non_animal_detected = any(
                model.names[int(box.cls)] not in animal_classes and box.conf >= confidence_threshold
                for box in results[0].boxes
            )

            if non_animal_detected:
                # Копируем изображение в папку вывода
                shutil.copy(file_path, output_folder)
                print(
                    f"Объекты, не относящиеся к животным, обнаружены на {filename}. Файл скопирован в {output_folder}.")

                # Удаляем изображение из папки input_folder
                os.remove(file_path)
                print(f"Файл {filename} удален из {input_folder}.")
            else:
                print(f"На {filename} обнаружены только животные или ничего не обнаружено.")


# Использование функции:
input_folder = 'downloaded_images_from_VK — копия'  # Путь к папке с исходными изображениями
output_folder = 'downloaded_images_from_VK — копия/faces_2'  # Путь к папке, куда будут сохранены изображения без животных

detect_non_animal_objects(input_folder, output_folder)
