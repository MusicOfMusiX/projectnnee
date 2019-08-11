"""
This is a FIXED-TOPOLOGY NENN; NOT A TWEANN.
"""

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

#Additional libaries
import random

print("TF version: " + tf.__version__)

#Macrodef - Defaults.
IMAGE_SIZE = 28
NUMBER_OF_CLASSES = 10

NUMBER_OF_HIDDEN_LAYERS = 2
NUMBER_OF_NODES_PER_LAYER = 32

NUMBER_OF_INDIVIDUALS = 10 #i.e. Population size
NUMBER_OF_GENERATIONS = 1000
BATCH_SIZE = 256

RETAIN_PROPORTION = 0.3
UNDERDOG_SURVIVAL = 0.05
MUTATION_CHANCE = 0.1

#Load raw data from TF servers
"""
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
"""
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalise
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""
Chromosome design [GATTACA]: [array of weight values. No separation of layers, all linear.]
"""
class Individual: #Our battle royale contestant.
	def __init__(self, number_of_hidden_layers, number_of_nodes_per_layer):
		self.n_hidden_layers = number_of_hidden_layers
		self.n_nodes_per_layer = number_of_nodes_per_layer
		
		#Create keras model
		self.model = keras.Sequential()
		
		#Add layers
		#self.model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
		self.model.add(keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)))
		
		for i in range(self.n_hidden_layers):
			self.model.add(keras.layers.Dense(self.n_nodes_per_layer, activation=tf.nn.relu, kernel_initializer='random_uniform',
					bias_initializer='zeros'))	
		self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='random_uniform',
					bias_initializer='zeros'))
					
		#Compile model. Note that the optimiser parameter is insignifcant as we don't use it for fitting.
		self.model.compile(optimizer=keras.optimizers.SGD(1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		
	def model():
		return self.model

###All other NE-related functions are done outside of the Individual class.
def fitness_test(individual, inputs, labels):
	return individual.model.evaluate(x=inputs, y=labels, batch_size=BATCH_SIZE, verbose=0)[0] #Returns cross entropy loss
	
def test_population(population, inputs, labels):
	loss_sum = 0
	acc_sum = 0
	for individual in population:
		result = individual.model.evaluate(x=inputs, y=labels, batch_size=BATCH_SIZE, verbose=0)
		loss_sum += result[0]
		acc_sum += result[1]
	
	return loss_sum / len(population), acc_sum / len(population)
	
def create_population(population_size):
	population = []
	for i in range(population_size):
		temp = Individual(NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NODES_PER_LAYER)
		population.append(temp)
	return population
	
#Encode function working as expected.
def encode(individual): #Encode/extract genetic info; i.e. encode weights into a 1D linear array.
	#This chromosome is exclusive to the individual's exact config, does NOT work for other architectures!
	chromosome = [] #This is a Python array, not numpy.
	for i in range(individual.n_hidden_layers+1): #Number of weight matrices
		layer = individual.model.get_layer(index=i+1).get_weights() #Layer index starts from 1!
		for j in layer: #layer[0] = weight matirx, layer[1] = bias vector
			#print(np.shape(j))
			for k in j.flatten(): #Flatten all matrices
				chromosome.append(k)
	return chromosome

def decode(individual, chromosome): #Decode chromosome and apply to model
	for i in range(individual.n_hidden_layers+1): #Number of weight matrices
		layer = individual.model.get_layer(index=i+1).get_weights() #Layer index starts from 1!
		weights = layer[0]
		biases = layer[1]
		
		for j in range(np.shape(weights)[0]):
			for k in range(np.shape(weights)[1]):
				weights[j, k] = chromosome.pop(0)
		for j in range(np.shape(biases)[0]):
			biases[j] = chromosome.pop(0)
		
		#Overwrite with decoded values
		layer[0] = weights
		layer[1] = biases
		
		individual.model.get_layer(index=i+1).set_weights(layer)
	return individual

def crossover(chromosome1, chromosome2):
	"""
	We only swap single weights.
	Half from the father, half from the mother. The half aspect will be done with RNG w/ 50% chances.
	Alternatives would be to swap weights of a whole neuron, or an entire layer.
	"""
	offspring_chromosome = []
	for i in range(len(chromosome1)):	
		if(random.randrange(100) < 50):
			offspring_chromosome.append(chromosome1[i]) #Father
		else:
			offspring_chromosome.append(chromosome2[i]) #Mother	
			
	return offspring_chromosome
	
def mutate(chromosome, chance): #Mutate weight or bias. The probability of doing so is handled by the parent.
	for i in range(len(chromosome)): #Iterate thru all weights/biases, mutate them as we go.
		if(random.random() < chance):
			random_scale = random.random() + 0.5 #A multiplication factor between 0.5 and 1.5.
			#See this: https://stackoverflow.com/questions/31708478/how-to-evolve-weights-of-a-neural-network-in-neuroevolution
			chromosome[i] *= random_scale
	return chromosome
	
def evolve(population, test_inputs, test_labels, retain_proportion, underdog_survival, mutation_chance):
	next_population = []
	leaderboard = sorted(population, key=lambda individual: fitness_test(individual, test_inputs, test_labels))
	expected_number_of_survivors = int(len(population) * retain_proportion)
	
	#Best-performing parents survive
	for i in range(expected_number_of_survivors):
		next_population.append(leaderboard[i])
		
	#Underdogs survive
	for individual in leaderboard:
		if(random.random() < underdog_survival):
			next_population.append(individual)
	
	#Fill in the dead
	number_of_required_offspring = len(population) - len(next_population)
	
	#Survivors breed
	for i in range(number_of_required_offspring):
		#Select parents
		father = random.choice(next_population)
		next_population.remove(father) #Temporarily, to avoid self-breeding..
		mother = random.choice(next_population)
		next_population.append(father)

		#Extract chromosomes
		father_chromosome = encode(father)
		mother_chromosome = encode(mother)
		
		#Crossover
		offspring_chromosome = crossover(father_chromosome, mother_chromosome)
		
		#We only mutate offspring, not parents.
		offspring_chromosome = mutate(offspring_chromosome, mutation_chance)
		
		#Apply new genetric material/chromosome to newborn offspring
		offspring = Individual(NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NODES_PER_LAYER) #Create 'blank' offspring individual
		
		offspring = decode(offspring, offspring_chromosome)
		#Add offspring to next generation population
		next_population.append(offspring)
	return next_population
	
###Main func###
population = create_population(NUMBER_OF_INDIVIDUALS)
for generation in range(NUMBER_OF_GENERATIONS):
	population = evolve(population, x_train, y_train, RETAIN_PROPORTION, UNDERDOG_SURVIVAL, MUTATION_CHANCE) #Update population
	print("Generation " + str(generation+1) + " average loss & accuracy: " + str(test_population(population, x_test, y_test)))