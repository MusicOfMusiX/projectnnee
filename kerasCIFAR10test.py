from __future__ import absolute_import, division, print_function, unicode_literals

#Ignore FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

#Remove SSL
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
	getattr(ssl, '_create_unverified_context', None)): 
	ssl._create_default_https_context = ssl._create_unverified_context

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("TF version: " + tf.__version__)

#Macrodef
HIDDEN_LAYERS = 4 #Dummy for now.
NODES_PER_LAYER = 64 #Dummy for now.
LEARNING_RATE = 0.01
EPOCHS = 10

#Load raw data from TF servers
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Normalise
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Model init
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
for i in range(HIDDEN_LAYERS):
	model.add(keras.layers.Dense(NODES_PER_LAYER, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer=keras.optimizers.SGD(LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train
print("Initiating training")
model.fit(x_train, y_train, epochs=EPOCHS)

#Evaluate model on test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)