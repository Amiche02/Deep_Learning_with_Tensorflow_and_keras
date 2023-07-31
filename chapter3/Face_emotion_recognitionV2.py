# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:12:43 2023

@author: stephane kpoviessi
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

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

def shuffle_data(images, labels):
    num_samples = len(images)
    shuffled_indices = np.random.permutation(num_samples)

    shuffled_images = images[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    return shuffled_images, shuffled_labels

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
    
def count_elements(array):
    unique_elements, counts = np.unique(array, return_counts=True)
    element_counts = dict(zip(unique_elements, counts))
    return element_counts



directory = "images"

categories = get_file_names_in_directory(os.path.join(directory, "train"))

train_images, train_labels = load_data(os.path.join(directory, "train"))
val_images, val_labels = load_data(os.path.join(directory, "validation"))

print(f"Categories : {categories}")

print(f"Train data shape : {(train_images.shape, train_labels.shape)},\
    \nValidation data shape : {(val_images.shape, val_labels.shape)}")

idx = np.random.randint(0, len(val_labels))
plt.subplot(1, 2, 1)
plt.imshow(train_images[idx], cmap="gray")
plt.title(f"{categories[train_labels[idx]]}")

plt.subplot(1, 2, 2)
plt.imshow(val_images[idx], cmap="gray")
plt.title(f"{categories[val_labels[idx]]}")

plt.tight_layout()
plt.show()


np.random.seed(0)
train_shuffled_images, train_shuffled_labels = shuffle_data(train_images, train_labels)
val_shuffled_images, val_shuffled_labels = shuffle_data(val_images, val_labels)

print(f"Equality : {(sum(train_shuffled_labels == train_labels), sum(val_shuffled_labels == val_labels))}")

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.hist(train_shuffled_labels, bins=15)
plt.title("Train label count plot")
plt.xlabel("Categories")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(val_shuffled_labels, bins=15)
plt.title("Validation label count plot")
plt.xlabel("Categories")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


train_counts = np.unique(train_shuffled_labels, return_counts=True)
val_counts = np.unique(val_shuffled_labels, return_counts=True)

fig, ax = plt.subplots(1, 2, sharey=True)

ax[0].pie(train_counts[1], labels=categories, autopct='%1.1f%%')
ax[0].set_title('Train Data')

ax[1].pie(val_counts[1], labels=categories, autopct='%1.1f%%')
ax[1].set_title('Validation Data')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plot_images(train_shuffled_images, train_shuffled_labels)

plt.figure(figsize=(6,6))
plot_images(val_shuffled_images, val_shuffled_labels)

class_counts = count_elements(train_shuffled_labels)
class_weights = {class_label: 1.0 / count for class_label, count in class_counts.items()}

v_class_counts = count_elements(val_shuffled_labels)
v_class_weights = {class_label: 1.0 / count for class_label, count in class_counts.items()}

print(f"Train class counts : {class_counts} \nValidation class counts : {v_class_counts}")

print(f"\n\nTrain class weights : {class_weights} \n\nValidation class weights : {v_class_weights}")

num_classes = len(categories)

x_train = np.expand_dims(train_shuffled_images, axis=-1)
x_val = np.expand_dims(val_shuffled_images, axis=-1)

x_train.shape, x_val.shape



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

batch_size = 128
train_generator = datagen.flow_from_directory(
    os.path.join(directory, "train"),
    target_size = (48,48),
    color_mode = "grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

val_generator = datagen.flow_from_directory(
    os.path.join(directory, "validation"),
    target_size = (48,48),
    color_mode = "grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


model = keras.models.Sequential()

#1st CNN layer
model.add(keras.layers.Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(keras.layers.Dropout(0.25))

#2nd CNN layer
model.add(keras.layers.Conv2D(128,(5,5),padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(keras.layers.Dropout (0.25))

#3rd CNN layer
model.add(keras.layers.Conv2D(512,(3,3),padding = 'same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(keras.layers.Dropout (0.25))

#4th CNN layer
model.add(keras.layers.Conv2D(512,(3,3), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())

#Fully connected 1st layer
model.add(keras.layers.Dense(256))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.25))


# Fully connected layer 2nd layer
model.add(keras.layers.Dense(512))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

history = model.fit(
    x_train,
    train_shuffled_labels,
    epochs=20,
    validation_data=(x_val, val_shuffled_labels),
    class_weight=class_weights
)