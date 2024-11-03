import os
import shutil


def move_files_to_parent(parent_dir):
    # Получаем список всех подкаталогов в родительской папке
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(parent_dir, subdir)
        # Получаем список всех файлов в подкаталоге
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]

        for file in files:
            src_file = os.path.join(subdir_path, file)
            dst_file = os.path.join(parent_dir, file)
            # Перемещаем файл в родительскую папку
            shutil.move(src_file, dst_file)

        # Удаляем пустой подкаталог
        os.rmdir(subdir_path)


parent_dir = 'downloaded_images'
#parent_dir = 'animal_100'
move_files_to_parent(parent_dir)

