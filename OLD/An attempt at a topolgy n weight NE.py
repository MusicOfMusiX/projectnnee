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

#Macrodef - Defaults.
IMAGE_SIZE = 32
NUMBER_OF_CLASSES = 10

NUMBER_OF_HIDDEN_LAYERS = 2
NUMBER_OF_NODES_PER_LAYER = 128

NUMBER_OF_INDIVIDUALS = 100
NUMBER_OF_GENERATIONS = 10
NUMBER_OF_TESTS = 1000

#Load raw data from TF servers
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Normalise
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""
Chromosome design [GATTACA]:
number of inputs e.g. 1024
number of outputs e.g. 10
number of hidden layers e.g. 3
number of nodes per hidden layer e.g. 32

[There will be n_hidden_layers + 1 number of weight matrices.]

CODE = [array of weight values. No separation of layers, all linear.]
"""

#Chromosome first, individual later. Egg & chicken.
class Chromosome:
	def __init__(self):
		self.n_inputs = IMAGE_SIZE**2
		self.n_outputs = NUMBER_OF_CLASSES
		self.n_hidden_layers = NUMBER_OF_HIDDEN_LAYERS
		self.n_nodes_per_layer = NUMBER_OF_NODES_PER_LAYER
		self.weight_code = []
		

class Individual: #Our battle royale contestant.
	def __init__(chromosome):
		self.chromosome = chromosome
		
		#Create keras model
		self.model = keras.Sequential()
		
		#Add layers
		self.model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
		for i in range(chromosome.n_hidden_layers):
			self.model.add(keras.layers.Dense(self.chromosome.n_nodes_per_layer, activation=tf.nn.relu, kernel_initializer='random_uniform',
					bias_initializer='zeros'))	
		self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='random_uniform',
					bias_initializer='zeros'))
					
		#Compile model. Note that the optimiser parameter is insignifcant as we don't use it for fitting.
		self.model.compile(optimizer=keras.optimizers.SGD(1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		
	def model():
		return self.model
		
	def encode():
		
		
	def decode(chromosome):
		
def create_population(population_size):
	population = []
	for i in range(population_size):
		population.append(Individual(NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NODES_PER_LAYER))
	return population
	
def encode(individual): #Encode/extract genetic info; i.e. encode weights (And architecture info; this is for the future.) into an array.
	chromosome = []
	return chromosome

def decode(chromosome): #Decode chromosome and apply to model
	pass	

def crossover(chromosome1, chromosome2):
	pass
	
def mutate_weight(chromosome):
	pass

def evolve(population, retain=0.2, random_select=0.05, mutate=0.01):
	pass
	
###Main func###
population = create_population(NUMBER_OF_INDIVIDUALS)

for generation in range(NUMBER_OF_GENERATIONS):
	print("Generation " + str(generation+1))
	for test in range(NUMBER_OF_TESTS):
		for individual in range(NUMBER_OF_INDIVIDUALS):
			pass
			
			
		