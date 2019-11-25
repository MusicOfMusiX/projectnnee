"""
MODEL C
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
import timeit
tic=timeit.default_timer()

print("TF version: " + tf.__version__)

#Macrodef - Defaults.
IMAGE_SIZE = 28
NUMBER_OF_CLASSES = 10

MAX_NUMBER_OF_HIDDEN_LAYERS = 5
MAX_NUMBER_OF_NODES_PER_LAYER = 64

NUMBER_OF_INDIVIDUALS = 10 #i.e. Population size
NUMBER_OF_GENERATIONS = 5000
BATCH_SIZE = 512
NUMBER_OF_EPOCHS = 10

RETAIN_PROPORTION = 0.3
UNDERDOG_SURVIVAL = 0.05
MUTATION_CHANCE = 0.1

#Load raw data from TF servers

cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
"""
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
#Normalise
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

ACTIVATIONS = ["elu", "relu", "sigmoid", "tanh"]
OPTIMISERS = ["sgd", "adam", "adamax"]

class Individual: #Our battle royale contestant.
	def __init__(self, number_of_hidden_layers, number_of_nodes_per_layer, activation, optimiser):
		#Create keras model
		self.number_of_hidden_layers = number_of_hidden_layers
		self.number_of_nodes_per_layer = number_of_nodes_per_layer
		#These are INDEX VALUES, not strings.
		self.activation = activation
		self.optimiser = optimiser
		
		self.model = keras.Sequential()
		
		#Add layers
		self.model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
		#self.model.add(keras.layers.Flatten(input_shape=(28, 28)))
		
		for i in range(self.number_of_hidden_layers):
			self.model.add(keras.layers.Dense(self.number_of_nodes_per_layer, activation=ACTIVATIONS[self.activation], kernel_initializer='random_uniform',
					bias_initializer='zeros'))	
		self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='random_uniform',
					bias_initializer='zeros'))
					
		#Compile model. Note that the optimiser parameter is insignifcant as we don't use it for fitting.
		self.model.compile(optimizer=OPTIMISERS[self.optimiser], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		
	def model():
		return self.model

###All other NE-related functions are done outside of the Individual class.
def fitness_test(individual, inputs, labels):
	individual.model.fit(x=inputs, y=labels, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS, verbose=1)
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
		one =random.randrange(0,6,1)
		two =random.randrange(10,65,1)
		three =random.randrange(0,4,1)
		four =random.randrange(0,3,1)
		temp = Individual(one, two, three, four) 
		population.append(temp)
		print("specs: " + str(temp.number_of_hidden_layers) + ", " + str(temp.number_of_nodes_per_layer) + ", " + str(temp.activation) + ", " + str(temp.optimiser))
	return population

"""
Chromosome design [GATTACA]: [array of index values for different hyperparameters & structural info.]
# of hidden layers (5 MAX): 0-5
# of nodes per layer (64 MAX): 1-64
Activation for hidden layer: 0 - elu, 1 - relu, 2 - tanh, 3 - sigmoid
Optimiser: 0 - SGD, 1 - Adam, 2 - Adamax
Learning rate - 0.001-0.1

 e.g. [3, 45, 2, 0.07]
"""

#Encode function working as expected.
def encode(individual): #Encode/extract genetic info; i.e. encode weights into a 1D linear array.
	#This chromosome is exclusive to the individual's exact config, does NOT work for other architectures!
	chromosome = [] #This is a Python array, not numpy.
	chromosome.extend([individual.number_of_hidden_layers, individual.number_of_nodes_per_layer, individual.activation, individual.optimiser])
	return chromosome

def decode_and_create(chromosome): #Decode chromosome.
	"""
	We HAVE to create a new instance, so no individuals as arguments.
	"""
	#We create an entirely new instance of an Individual, and simply give the name of the past generation's corpse.
	#Cannot use that [HACK] from Model B though.
	individual = Individual(chromosome[0], chromosome[1], chromosome[2], chromosome[3])
	return individual

def crossover(chromosome1, chromosome2):
	offspring_chromosome = []
	for i in range(4):	
		if(random.randrange(100) < 50):
			offspring_chromosome.append(chromosome1[i]) #Father
		else:
			offspring_chromosome.append(chromosome2[i]) #Mother	
			
	return offspring_chromosome
	
def mutate(chromosome, chance): #Mutate weight or bias. The probability of doing so is handled by the parent.
	#Hidden layers 0-5
	if(random.random() < chance):
		random_scale = random.choice([-2,-1,1,2]) #Choices to small to use 0.5-1.5 scales.
		chromosome[0] = max(min(chromosome[0] + random_scale,5),0)
	#Number of nodes 10-64
	if(random.random() < chance):
		random_scale = random.random() + 0.5
		chromosome[1] = max(min(chromosome[1] * random_scale,64),10)
	#Activation 0-3
	if(random.random() < chance):
		chromosome[2] = random.randrange(0,4,1)
	#Optimiser 0-2
	if(random.random() < chance):
		chromosome[3] = random.randrange(0,3,1)
		
	return chromosome
	
def evolve(generation, population, test_inputs, test_labels, retain_proportion, underdog_survival, mutation_chance):
	#We need to test the population now, as many are going to die off and be replaced with untrained offspring.
	next_population = []
	leaderboard = sorted(population, key=lambda individual: fitness_test(individual, test_inputs, test_labels))
	print("Generation " + str(generation+1) + " - average loss & accuracy: " + str(test_population(leaderboard, x_test, y_test)) + " Time: " + str(toc-tic))
	print("Lead specs: " + str(leaderboard[0].number_of_hidden_layers) + ", " + str(leaderboard[0].number_of_nodes_per_layer) + ", " + str(leaderboard[0].activation) + ", " + str(leaderboard[0].optimiser))
	expected_number_of_survivors = int(len(population) * retain_proportion)
	
	#Best-performing parents survive
	for i in range(expected_number_of_survivors):
		next_population.append(leaderboard[i])
		
	#Underdogs survive
	for individual in leaderboard:
		if(random.random() < underdog_survival):
			next_population.append(individual)
	
	#Fill in the dead
	survivors = len(next_population)
	number_of_required_offspring = len(population) - survivors
	
	#[HACK] Prepare parent chromosomes, to save encode time:
	parent_chromosomes = []
	for parent in next_population:
		parent_chromosomes.append(encode(parent))
		
	#RESET!!
	next_population = []
	
	for i in range(survivors):
		next_population.append(decode_and_create(parent_chromosomes[i]))
	
	#Survivors breed
	for i in range(number_of_required_offspring):
		#Select parent chromosome (Not the intuitive way, but hopefully faster.)
		father_chromosome = random.choice(parent_chromosomes)
		parent_chromosomes.remove(father_chromosome) #Temporarily, to avoid self-breeding..
		mother_chromosome = random.choice(parent_chromosomes)
		parent_chromosomes.append(father_chromosome)
		
		#Crossover
		offspring_chromosome = crossover(father_chromosome, mother_chromosome)
		
		#We only mutate offspring, not parents.
		offspring_chromosome = mutate(offspring_chromosome, mutation_chance)
		
		#No use of [HACK] this time..
		offspring = decode_and_create(offspring_chromosome)
		
		#Add offspring to next generation population
		next_population.append(offspring)
	return next_population
	
###Main func###
population = create_population(NUMBER_OF_INDIVIDUALS)
for generation in range(NUMBER_OF_GENERATIONS):
	toc=timeit.default_timer()
	population = evolve(generation, population, x_train, y_train, RETAIN_PROPORTION, UNDERDOG_SURVIVAL, MUTATION_CHANCE) #Update population
	