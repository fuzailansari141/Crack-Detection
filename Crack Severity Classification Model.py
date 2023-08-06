import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Define the path to the image folders
data_dir = r"Path"
categories = ["Minor", "Moderate", "Major"]

# Prepare the training data
images = []
labels = []

# Load images and assign labels
for category in categories:
    folder_path = os.path.join(data_dir, category)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))  # Resize the image if needed
        images.append(image)
        labels.append(categories.index(category))
        
# Convert images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Normalize the pixel values (optional)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(len(categories), activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Augment the training data (optional)
data_augmentation = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
data_augmentation.fit(train_images)

# Train the model
model.fit(data_augmentation.flow(train_images, train_labels, batch_size=32),
          epochs=10, validation_data=(test_images, test_labels))
