import os


# Функция для чтения маппинга из текстового файла
def load_mapping_from_file(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Разделяем WNID и название класса по первому пробелу
            parts = line.split(' ', 1)
            wnid = parts[0]
            class_name = parts[1].strip().split(',')[0]  # Берем только первое название до запятой
            mapping[wnid] = class_name.replace(" ", "_")  # Заменяем пробелы на подчеркивания
    return mapping


# Функция для переименования папок
def rename_folders(base_dir, mapping):
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            # Если WNID есть в маппинге, переименовываем папку
            if folder in mapping:
                new_folder_name = mapping[folder]
                new_folder_path = os.path.join(base_dir, new_folder_name)
                try:
                    os.rename(folder_path, new_folder_path)
                except:
                    continue

                print(f"Переименовано: {folder} -> {new_folder_name}")
            else:
                print(f"WNID {folder} не найдено в маппинге")

# Основная логика
if __name__ == "__main__":
    # Путь к папке imagenet-mini
    base_dir = "./imagenet-mini"

    # Путь к твоему текстовому файлу с маппингом
    mapping_file = "./map.txt"

    # Загружаем маппинг из файла
    mapping = load_mapping_from_file(mapping_file)

    # Переименовываем папки в train
    train_dir = os.path.join(base_dir, "train")
    rename_folders(train_dir, mapping)

    # Переименовываем папки в val
    val_dir = os.path.join(base_dir, "val")
    rename_folders(val_dir, mapping)


