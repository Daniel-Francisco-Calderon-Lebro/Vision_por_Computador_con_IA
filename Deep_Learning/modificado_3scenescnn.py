# -*- coding: utf-8 -*-
"""Modificado 3scenesCNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15HxAEqKmRvf-GHyHjmbpHsCzB1643mXA

# Imports and drive
"""

from google.colab import drive
drive.mount('/content/drive')

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import os
import random
from google.colab import drive
drive.mount('/content/drive')

"""# Read images and create dataset"""

base_path = '/content/drive/MyDrive/Carpeta Parcial 1 VISION IA/'
folder = 'Costas,Forest and Higway/'
subfolders = ['coast', 'forest', 'highway']

data = []
labels = []
for subfolder in subfolders:
  files = [f for f in os.listdir(base_path + folder + subfolder) if f.endswith('.jpg')]
  # print(subfolder)
  # print(files)
  # print(len(files))
  for file in files:
    img = cv2.imread(base_path + folder + subfolder + '/' + file, cv2.IMREAD_COLOR)
    img = img.astype('float32') / 255.0
    img = cv2.resize(img, (128, 128))
    data.append(img)
    labels.append(subfolder)

pos = random.randint(0, len(data))
cv2_imshow(data[pos]*255)
print(labels[pos])
print(data[pos].shape)

print(len(data))

# encode the labels, converting them from strings to integers
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(labels)

"""# Split dataset into train and test"""

# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

"""# Define model architecture"""

# define our Convolutional Neural Network architecture
model = Sequential()
model.add(Conv2D(128, (7, 7), padding="same", input_shape=(128, 128, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(16, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation("softmax"))

#Summary
model.summary()

"""# Compile model and train"""

opt = Adam(learning_rate=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),	epochs=15, batch_size=32)

"""# Evaluate model"""

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),	predictions.argmax(axis=1), target_names=lb.classes_))

print(lb.classes_)

"""# Test on random sample"""

rand_pos = random.randint(0, len(testX))
rand_img = testX[rand_pos]
rand_img_resized = 255 * cv2.resize(rand_img, (128, 128))
cv2_imshow(rand_img_resized)

print('Ground truth class: ', lb.classes_[np.argmax(testY[rand_pos])])
print('Predicted class: ', lb.classes_[np.argmax(predictions[rand_pos])])