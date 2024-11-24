# Импорт необходимых библиотек
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from pathlib import Path
from PIL import Image
import os
import shutil
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.applications import efficientnet
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

# Задание параметров
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
CHECKPOINT_PATH = "animals_classification_model_checkpoint_30epoch_b4.weights.h5"

# Словарь классов
class_indices = {
    0: 'Birds',
    1: 'Cats',
    2: 'Cow',
    3: 'Dogs',
    4: 'Elephant',
    5: 'Giraffe',
    6: 'Other',
    7: 'People',
    8: 'Predators',
    9: 'Primates',
    10: 'Rabbit',
    11: 'Reptiles',
    12: 'Sheep',
    13: 'Zebra'
}

# Загрузка предобученной модели
pretrained_model = tf.keras.applications.efficientnet.EfficientNetB4(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)

pretrained_model.trainable = False

# Построение модели с той же архитектурой
inputs = pretrained_model.input
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.RandomFlip("horizontal")(x)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.RandomContrast(0.1)(x)

x = pretrained_model.output
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.45)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.45)(x)
outputs = layers.Dense(14, activation='softmax')(x)  # Замените '10' на количество классов в вашем наборе данных

model = Model(inputs=inputs, outputs=outputs)

# Загрузка сохранённых весов
model.load_weights(CHECKPOINT_PATH)

# Функция для предсказания и сортировки изображений
def predict_and_sort(dataset_path, output_dir):
    # Преобразование пути к изображениям в DataFrame
    image_dir = Path(dataset_path)
    filepaths = list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.jpeg'))
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)

    # Генератор для предобработки изображений
    test_generator = ImageDataGenerator(preprocessing_function=efficientnet.preprocess_input)
    test_images = test_generator.flow_from_dataframe(
        dataframe=filepaths.to_frame(name='Filepath'),
        x_col='Filepath',
        y_col=None,
        target_size=TARGET_SIZE,
        color_mode='rgb',
        class_mode=None,  # Здесь class_mode=None, так как метки отсутствуют
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Предсказание
    predictions = model.predict(test_images, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Перемещение файлов в папки
    for filepath, pred_class in zip(filepaths, predicted_classes):
        class_name = class_indices[pred_class]  # Использование заранее определённого словаря
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)  # Создать папку для класса, если её нет

        destination = os.path.join(class_dir, os.path.basename(filepath))
        shutil.move(filepath, destination)  # Переместить файл
        print(f"Файл {filepath} перемещён в {destination}")


# Путь к новому датасету
new_dataset_path = "downloaded_images_from_VK — копия"  # Укажите путь к папке с новыми данными
output_dir = "downloaded_images_from_VK — копия/sorted_images"  # Папка, куда будут перемещены изображения

# Выполнение предсказания и сортировки
predict_and_sort(new_dataset_path, output_dir)

print("Все изображения были распределены по папкам!")

# Функция для получения массива изображения
def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Функция для создания тепловой карты Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


from scipy.ndimage import gaussian_filter


def save_and_display_gradcam(img_path, heatmap, alpha=0.4, sigma=2):
    img = load_img(img_path)
    img = np.array(img)

    # Преобразуем тепловую карту в диапазон [0, 255]
    heatmap = np.uint8(255 * heatmap)

    # Сглаживаем тепловую карту с помощью гауссовского размытия
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Приводим тепловую карту к размерам изображения
    heatmap = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]))
    heatmap = np.array(heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3] * 255

    # Накладываем тепловую карту на изображение
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)

    # Возвращаем наложенное изображение
    return superimposed_img


# Настройка имени последнего сверточного слоя
last_conv_layer_name = "top_conv"  # Указан последний сверточный слой
img_size = TARGET_SIZE


def display_gradcam_per_class(dataset_path, model, last_conv_layer_name, cols=5):
    # Словарь для хранения по одному изображению на класс
    class_to_image = {}

    # Проходим по всем файлам в датасете
    image_dir = Path(dataset_path)
    filepaths = list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.jpeg'))

    for img_path in filepaths:
        class_name = Path(img_path).parent.name  # Имя папки, содержащей файл
        # Если класс еще не добавлен в словарь, добавляем текущее изображение
        if class_name not in class_to_image:
            class_to_image[class_name] = img_path

    # Вычисляем количество строк и столбцов
    num_classes = len(class_to_image)
    rows = (num_classes + cols - 1) // cols  # Округление вверх для строк

    # Создаем визуализацию для каждого класса
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4), subplot_kw={'xticks': [], 'yticks': []})
    axes = axes.flatten()  # Преобразуем массив осей в плоский список

    for ax, (class_name, img_path) in zip(axes, class_to_image.items()):
        # Предобработка изображения
        img_array = preprocess_input(get_img_array(str(img_path), size=img_size))

        # Генерация Grad-CAM тепловой карты
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        superimposed_img = save_and_display_gradcam(str(img_path), heatmap)  # Получаем наложенное изображение

        # Отображение изображения с заголовком класса
        ax.imshow(superimposed_img)
        ax.set_title(f"Class: {class_name}")

    # Скрыть лишние оси, если классов меньше, чем ячеек
    for ax in axes[num_classes:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Пример вызова функции
display_gradcam_per_class(new_dataset_path, model, last_conv_layer_name)