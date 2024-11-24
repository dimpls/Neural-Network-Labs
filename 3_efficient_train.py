# Импорт библиотек
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from PIL import UnidentifiedImageError, Image

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Using CPU.")

def seed_everything(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything()

# Параметры
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

dataset = "Our_training_dataset_new — копия"

# Функция для преобразования пути в DataFrame
def convert_path_to_df(dataset):
    image_dir = Path(dataset)
    filepaths = list(image_dir.glob(r'**/*.jpg')) + \
                list(image_dir.glob(r'**/*.jpeg')) + \
                list(image_dir.glob(r'**/*.png'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    return pd.concat([filepaths, labels], axis=1)

# Преобразование пути в DataFrame
image_df = convert_path_to_df(dataset)

# Проверка на поврежденные изображения
for img_p in Path(dataset).rglob("*.*"):
    try:
        img = Image.open(img_p)
    except UnidentifiedImageError:
        print(f"Corrupted image: {img_p}")

label_counts = image_df['Label'].value_counts()

fig, ax = plt.subplots(figsize=(20, 8))
sns.barplot(
    x=label_counts.index,
    y=label_counts.values,
    palette="Spectral",
    ax=ax
)

ax.set_title('Распределение меток в наборе изображений', fontsize=18, fontweight='bold')
ax.set_xlabel('Метка', fontsize=16, labelpad=10)
ax.set_ylabel('Количество', fontsize=16, labelpad=10)

# Добавление аннотаций
for i, value in enumerate(label_counts.values):
    ax.text(
        i,
        value + max(label_counts.values) * 0.01,
        str(value),
        ha='center',
        va='bottom',
        fontsize=12
    )

ax.set_xticklabels(label_counts.index, rotation=45, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

random_index = np.random.randint(0, len(image_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})

# Главный заголовок поднят выше
fig.suptitle('Примеры изображений из набора данных', fontsize=20, fontweight='bold', y=1)

for i, ax in enumerate(axes.flat):
    img = plt.imread(image_df.Filepath.iloc[random_index[i]])
    ax.imshow(img)

    label = image_df.Label.iloc[random_index[i]]
    ax.set_title(
        label,
        fontsize=12,
        color='darkblue',
        pad=8
    )

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    subset='validation'
)

# Увеличение данных
augment = tf.keras.Sequential([
    layers.Resizing(224, 224),
    layers.Rescaling(1. / 255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='max'
# )

# Загрузка предобученной модели
pretrained_model = tf.keras.applications.efficientnet.EfficientNetB7(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)

# pretrained_model = tf.keras.applications.efficientnet.EfficientNetB5(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='max'
# )

pretrained_model.trainable = False

# Callback-и
checkpoint_path = "animals_classification_model_checkpoint_30epoch_b7_delete_test.weights.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor="val_accuracy",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Построение модели
inputs = pretrained_model.input
x = augment(inputs)
x = Dense(128, activation='relu')(pretrained_model.output)
x = BatchNormalization()(x)
x = Dropout(0.45)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.45)(x)
outputs = Dense(len(label_counts), activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Компиляция модели
model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение модели
history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    validation_data=val_images,
    validation_steps=len(val_images),
    epochs=30,
    callbacks=[
        early_stopping,
        checkpoint_callback,
        reduce_lr
    ]
)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# График точности
ax1.plot(epochs, accuracy, 'b-o', label='Training accuracy')
ax1.plot(epochs, val_accuracy, 'r-s', label='Validation accuracy')
ax1.set_title('Точность (accuracy)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Эпохи', fontsize=12)
ax1.set_ylabel('Точность', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.5)

# Добавляем аннотации для последней эпохи
ax1.annotate(f'{accuracy[-1]:.2f}', (epochs[-1], accuracy[-1]), textcoords="offset points", xytext=(-10, 10), ha='center', fontsize=10, color='blue')
ax1.annotate(f'{val_accuracy[-1]:.2f}', (epochs[-1], val_accuracy[-1]), textcoords="offset points", xytext=(-10, -15), ha='center', fontsize=10, color='red')

# График потерь
ax2.plot(epochs, loss, 'b-o', label='Training loss')
ax2.plot(epochs, val_loss, 'r-s', label='Validation loss')
ax2.set_title('Потери (loss)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Эпохи', fontsize=12)
ax2.set_ylabel('Потери', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.5)

# Добавляем аннотации для последней эпохи
ax2.annotate(f'{loss[-1]:.2f}', (epochs[-1], loss[-1]), textcoords="offset points", xytext=(-10, 10), ha='center', fontsize=10, color='blue')
ax2.annotate(f'{val_loss[-1]:.2f}', (epochs[-1], val_loss[-1]), textcoords="offset points", xytext=(-10, -15), ha='center', fontsize=10, color='red')

# Главный заголовок
fig.suptitle('Динамика обучения: точность и потери', fontsize=18, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Получение предсказаний на обучающем наборе
train_predictions = model.predict(train_images, verbose=1)
train_labels = train_images.classes
train_pred_labels = np.argmax(train_predictions, axis=1)

# Метрики для обучающего набора
train_accuracy = accuracy_score(train_labels, train_pred_labels)
train_precision = precision_score(train_labels, train_pred_labels, average='weighted')
train_recall = recall_score(train_labels, train_pred_labels, average='weighted')
train_f1 = f1_score(train_labels, train_pred_labels, average='weighted')
train_report = classification_report(train_labels, train_pred_labels, target_names=train_images.class_indices.keys())

# Получение предсказаний на валидационном наборе
val_predictions = model.predict(val_images, verbose=1)
val_labels = val_images.classes
val_pred_labels = np.argmax(val_predictions, axis=1)

# Метрики для валидационного набора
val_accuracy = accuracy_score(val_labels, val_pred_labels)
val_precision = precision_score(val_labels, val_pred_labels, average='weighted')
val_recall = recall_score(val_labels, val_pred_labels, average='weighted')
val_f1 = f1_score(val_labels, val_pred_labels, average='weighted')
val_report = classification_report(val_labels, val_pred_labels, target_names=val_images.class_indices.keys())

# Вывод результатов
print("=== Метрики для обучающего набора ===")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")
print("\nClassification Report:\n", train_report)

print("\n=== Метрики для валидационного набора ===")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1 Score: {val_f1:.4f}")
print("\nClassification Report:\n", val_report)

# Построение матриц ошибок
train_cm = confusion_matrix(train_labels, train_pred_labels)
val_cm = confusion_matrix(val_labels, val_pred_labels)

# Визуализация матриц ошибок
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Training Set', fontsize=14)
axes[0].set_xlabel('Predicted Labels', fontsize=12)
axes[0].set_ylabel('True Labels', fontsize=12)

sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix - Validation Set', fontsize=14)
axes[1].set_xlabel('Predicted Labels', fontsize=12)
axes[1].set_ylabel('True Labels', fontsize=12)

plt.tight_layout()
plt.show()

