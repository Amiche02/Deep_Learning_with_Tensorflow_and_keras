# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:12:43 2023

@author: stephane kpoviessi
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers

# Check if the version is TensorFlow 2.x
if tf.__version__.startswith('2'):
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        
def get_file_names_in_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"{directory_path} n'est pas un dossier valide.")
        return []

    return os.listdir(directory_path)

def shuffle_data(images, labels):
    num_samples = len(images)
    shuffled_indices = np.random.permutation(num_samples)

    shuffled_images = images[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    return shuffled_images, shuffled_labels

def load_data(directory):
    images = []
    labels = []
    label_dict = {}  # To map class names to unique labels
    current_label = 0

    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue

        label_dict[class_name] = current_label
        current_label += 1

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            if os.path.isfile(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is not None:
                    images.append(image)
                    labels.append(label_dict[class_name])

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


def count_elements(array):
    unique_elements, counts = np.unique(array, return_counts=True)
    element_counts = dict(zip(unique_elements, counts))
    return element_counts

def plot_images(images, labels, nb_rows=6, nb_cols=6, categories=None):
    nb = nb_cols * nb_rows
    total_images = len(images)
    
    categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    for i in range(min(nb, total_images)):
        plt.subplot(nb_rows, nb_cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"{labels[i]} : {categories[labels[i]]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
#--------------------------------------------------------------------------------------------------------------
directory = os.path.join("images", "train")

categories = get_file_names_in_directory(directory)
images, labels = load_data(directory)

print("\n", categories)
print()
print((images.shape, labels.shape))

plt.imshow(images[0], cmap="gray")
plt.show()

np.random.seed(0)
shuffled_images, shuffled_labels = shuffle_data(images, labels)
sum(shuffled_labels == labels)

print(shuffled_images.shape)
print()
print(shuffled_labels[:10])

plt.figure()
plt.hist(shuffled_labels, bins=15)
plt.show()

labels, counts = np.unique(shuffled_labels, return_counts=True)

print("\n", count_elements(shuffled_labels))
print(labels[:20])

plt.figure(figsize=(6,6))
plot_images(shuffled_images, shuffled_labels)
plt.show()


x_train, x_val, y_train, y_val = train_test_split(shuffled_images, shuffled_labels, test_size=0.2, random_state=0)
print(f"\n{x_train.shape, x_val.shape} \n{y_train[:10]}")

class_counts = count_elements(y_train)
class_weights = {class_label: 1.0 / count for class_label, count in class_counts.items()}
print(f"\nCount : {class_counts}, \nWeight : {class_weights}")

x_train_rgb = np.repeat(x_train[..., np.newaxis], 3, -1)
x_val_rgb = np.repeat(x_val[..., np.newaxis], 3, -1)

print(f"\nRGB : {(x_train_rgb.shape, x_val_rgb.shape)}")

x_train_resized = tf.image.resize(x_train_rgb, (224, 224))
x_val_resized = tf.image.resize(x_val_rgb, (224, 224))

print(f"\nResized : {(x_train_resized.shape, x_val_resized.shape)}")

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,         
    rotation_range=20,         
    width_shift_range=0.1,     
    height_shift_range=0.1,   
    shear_range=0.2,           
    zoom_range=0.2,            
    horizontal_flip=True,      
    fill_mode='nearest',  
)

batch_size = 32
train_generator = datagen.flow(
    x_train_resized,
    y_train,
    batch_size=batch_size,
    shuffle=True,
)

num_classes = len(categories)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential()
model.add(base_model)
print()
print(model.summary())
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
print()
print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=(x_val_resized, y_val),
    class_weight=class_weights
)