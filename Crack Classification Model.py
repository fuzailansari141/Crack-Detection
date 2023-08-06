# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:41:21 2023

@author: Fuzail Ansari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report

positive_dir = Path(r"C:\Users\Fuzail Ansari\Downloads\archive (15)\Positive")
negative_dir = Path(r"C:\Users\Fuzail Ansari\Downloads\archive (15)\Negative")

## image_widht
image_widht = 120
## image_height
image_height = 120
## image_color_channel_size
image_color_channel_size = 255
## image_size
image_size = (image_widht, image_height)
## batch_size
batch_size = 32
## epochs
epochs = 20
## learning_rate
learning_rate = 0.01
## class_names
class_names = ['crack','without_crack']

def create_df(image, label):
    filepaths = pd.Series(list(image.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df
positive_df = create_df(positive_dir, label="POSITIVE")
negative_df = create_df(negative_dir, label="NEGATIVE")

df = pd.concat(
    [positive_df, negative_df], axis=0).sample(
    frac=1.0, random_state= 42).reset_index(
    drop=True)

train_df, test_df = train_test_split(
    df.sample(6000, random_state=42),
    train_size=0.75,
    shuffle=True,
    random_state= 42
)

## rescale: Normalize the pixels
## validation_split : 
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./image_color_channel_size,
    validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./image_color_channel_size
)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=image_size,
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False,
    seed=42
)

inputs = tf.keras.Input(shape=(120,120, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)