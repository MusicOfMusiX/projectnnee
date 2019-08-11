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
NUMBER_OF_HIDDEN_LAYERS = 2
NUMBER_OF_NODES_PER_LAYER = 32
LEARNING_RATE = 0.01
NUMBER_OF_EPOCHS = 100

#Load raw data from TF servers

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalise
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Model init
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
for i in range(NUMBER_OF_HIDDEN_LAYERS):
	model.add(keras.layers.Dense(NUMBER_OF_NODES_PER_LAYER, activation=tf.nn.relu, kernel_initializer='random_uniform',
			bias_initializer='zeros'))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='random_uniform',
			bias_initializer='zeros'))

model.compile(optimizer=keras.optimizers.SGD(LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train
print("Initiating training")
model.fit(x_train, y_train, batch_size=512, epochs=NUMBER_OF_EPOCHS)

#Evaluate model on test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)