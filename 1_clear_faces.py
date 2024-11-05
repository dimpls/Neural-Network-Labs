import os
import shutil
import face_recognition


def move_images_with_faces(directory):
    # Путь к папке, куда будут перемещаться изображения с лицами
    faces_directory = os.path.join(directory, "faces")

    # Создание папки faces, если она еще не существует
    os.makedirs(faces_directory, exist_ok=True)

    # Проход по всем файлам в директории
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Проверка, что файл является изображением
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Загрузка изображения
        image = face_recognition.load_image_file(file_path)

        # Поиск лиц на изображении
        face_locations = face_recognition.face_locations(image)

        # Если лицо найдено, перемещаем файл в папку faces
        if face_locations:
            destination_path = os.path.join(faces_directory, filename)
            print(f"Перемещается файл: {filename} в папку faces (найдено лицо).")
            shutil.move(file_path, destination_path)
        else:
            print(f"Файл {filename} оставлен (лицо не найдено).")


# Пример использования
directory_path = "downloaded_images_from_VK — копия"
move_images_with_faces(directory_path)
