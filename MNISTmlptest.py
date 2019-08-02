#Python 2.7.16 w/ the latest version of numpy and matplotlib (as of 20 JULY 19)

import numpy as np
import matplotlib.pyplot as plt

import sys
np.set_printoptions(threshold=sys.maxsize)

###macrodef###

IMAGE_SIZE = 28 # width and length
NUMBER_OF_CLASSES = 10 #  i.e. 0, 1, 2, 3, ..., 9

NUMBER_OF_HIDDEN_LAYERS = 2 #This can be 0
NUMBER_OF_NODES_PER_LAYER = 32

NUMBER_OF_EXAMPLES = 500 #We perform stochasitc gradient descent; this is equivalent to the # of epochs.
NUMBER_OF_TESTING_EXAMPLES = 400

LEARINING_RATE = 0.05

LOG_VERBOSITY = 100

###funcdef###

def read_file(name, n):
	#These data files include labels in index 0 of each line.
	#784 0-255 integer values follow.
	return np.loadtxt("MNIST/" + name, delimiter=",", max_rows=n)

def load_examples(name, n):
	block = read_file(name, n)
	examples = np.zeros((n,IMAGE_SIZE**2))
	for i in range(n):
		for j in range(IMAGE_SIZE**2):
			examples[i,j] = block[i,j+1]
	return examples

def load_labels(name, n):
	#Why separate the two file reading? For CIFAR purposes. This MNIST data file has both the examples and labels.
	block = read_file(name ,n)
	labels = np.zeros((n))
	for i in range(n):
		labels[i] = block[i,0]
	return labels

def copy_single_example(examples, n): #Copy single example a.k.a. features vector
	#NOTE THAT WE ARE PRODUCING HORIZONTAL VECTORS/MATRICES.
	#This input/feature vector must be HORIZONTAL! Well, it's not a must but it is easier for forward propagation. Will convert to vertical during backprop.
	example = np.zeros((1,IMAGE_SIZE**2))
	for i in range(IMAGE_SIZE**2):
		if(examples[n,i] == 0):
			example[0,i] = 0
		elif(examples[n,i] != 0):
			example[0,i] = examples[n,i]/255
	return example
			
def copy_single_label(labels, n): #Convert single label to one-hot vector
	#TODO: Convert vector to horizontal (check whether labels should be horizontal first.)
	#TODO UPDATE: Not really. Doesn't really matter as we index through the elements with a for loop AND even that is done only once. 
	label = np.zeros((NUMBER_OF_CLASSES))
	for i in range(NUMBER_OF_CLASSES):
		if(i == labels[n]):
			label[i] = 1
		else:
			label[i] = 0
	return label

def softmax(logits):
	#Logits are horizontal vectors
	number_of_logits = logits.shape[1]
	probs = np.zeros((1, number_of_logits)) #Horizontal vector outputs
	sum = 0.0
	for i in range(number_of_logits):
		sum += np.exp(logits[0,i])
	for i in range(number_of_logits):
		probs[0,i] = np.exp(logits[0,i]) / sum
	return probs

def forward_propagate(weight_network, input):
	#Note that the network matrix is a list of matrices.
	probs = softmax(np.dot(input, weight_network[0]))
	softmax_network = []
	softmax_network.append(probs)
	for i in range(NUMBER_OF_HIDDEN_LAYERS):
		probs = softmax(np.dot(probs, weight_network[i+1]))
		softmax_network.append(probs)
	return probs, softmax_network
	
def test(weight_network, examples, labels):
	accsum = 0.0
	for i in range(NUMBER_OF_TESTING_EXAMPLES):
		features = copy_single_example(examples, i)
		dummy = []
		probabilities, dummy = forward_propagate(weight_network, features)	
		result = np.argmax(probabilities)
		if(result == labels[i]):
			accsum += 1
	return accsum/NUMBER_OF_TESTING_EXAMPLES*100

###backprop funcdef###

def softmax_derivative(output_softmax_layer):
	#WARNING: Softmax is a vector -> vector function! There is no single 'derivative', there are NxN partial derivatives instead.
	#Input softmax_layer vector is horizontal: np.array((1,n))
	#Read this: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
	#This... should work without transposing.
	n = output_softmax_layer.shape[1]
	jacobian = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			kronecker_delta = 1 if i == j else 0
			
			DjSi = output_softmax_layer[0,i] * (kronecker_delta - output_softmax_layer[0,j])
			jacobian[i,j] = DjSi
	return jacobian

def cross_entropy_derivative(input_softmax_layer, label_one_hot_vector): #input_softmax
	#Cross entropy is vector2scalar, the derivative is... vector2vector... because of the label vector...
	n = input_softmax_layer.shape[1]
	derivatives = np.zeros((1,n)) #Size: NUMBER_OF_CLASSES.
	for i in range(n):
		#ASSUMING LABEL IS VERTICAL FOR THE MOMENT.
		derivatives[0,i] = -1 * label_one_hot_vector[i] / input_softmax_layer[0,i] #dL/dO_i
	return derivatives

###__init__###

EXAMPLES = load_examples("mnist_train.csv", NUMBER_OF_EXAMPLES)
LABELS = load_labels("mnist_train.csv", NUMBER_OF_EXAMPLES)

TESTING_EXAMPLES = load_examples("mnist_test.csv", NUMBER_OF_TESTING_EXAMPLES)
TESTING_LABELS = load_labels("mnist_test.csv", NUMBER_OF_TESTING_EXAMPLES)

SOFTMAX_NETWORK = []
WEIGHT_NETWORK = []

if(NUMBER_OF_HIDDEN_LAYERS == 0): #SLP
	weights = np.zeros((IMAGE_SIZE**2, NUMBER_OF_CLASSES))
	WEIGHT_NETWORK.append(weights)

elif(NUMBER_OF_HIDDEN_LAYERS > 0): #MLP
	#InputL -> 1st HiddenL weights
	initial_weights = np.zeros((IMAGE_SIZE**2, NUMBER_OF_NODES_PER_LAYER))
	initial_weights.fill(0.5)
	WEIGHT_NETWORK.append(initial_weights)
	#HiddenL -> HiddenL weights
	hidden_weights = np.zeros((NUMBER_OF_NODES_PER_LAYER, NUMBER_OF_NODES_PER_LAYER))
	hidden_weights.fill(0.5)
	for i in range(NUMBER_OF_HIDDEN_LAYERS-1):
		WEIGHT_NETWORK.append(hidden_weights)
	#Last HiddenL -> OutputL weights
	output_weights = np.zeros((NUMBER_OF_NODES_PER_LAYER, NUMBER_OF_CLASSES))
	output_weights.fill(0.5)
	WEIGHT_NETWORK.append(output_weights)

### MAIN LOOP ###
print("Starting main loop")

for i in range(NUMBER_OF_EXAMPLES): #Iterate thru each epoch
	INPUTS = copy_single_example(EXAMPLES, i)
	LABEL = copy_single_label(LABELS, i)

	#Keep a log of softmax output values for each layer. Need to use them in softmax differentiation and hidden layer delta weight calculations.
	final_probs, SOFTMAX_NETWORK = forward_propagate(WEIGHT_NETWORK, INPUTS)
	
	#Test every 50 iterations
	if(i % LOG_VERBOSITY == 0):
		accuracy = test(WEIGHT_NETWORK, TESTING_EXAMPLES, TESTING_LABELS)
		print("Iteration #" + str(i) + ": " + "Accuracy: " + str(accuracy) + "%")
	
	#Ugh, difficult. See these notes: https://imgur.com/a/B0d9pIP
	
	#The pattern is like this [2xHiddenLayer setup, see above image.]:
	#CROSS-SOFTMAX-INPUT (Relative, Softmax output #2)
	#CROSS-SOFTMAX-[WEIGHTMATRIX-SOFTMAX]-INPUT (Relative, Softmax output #1)
	#CROSS-SOFTMAX-[WEIGHTMATRIX-SOFTMAX]-[WEIGHTMATRIX-SOFTMAX]-INPUT (Actual input vector)
	#And so on.
	
	#CROSS
	first_cross_entropy_derivative = cross_entropy_derivative(SOFTMAX_NETWORK[NUMBER_OF_HIDDEN_LAYERS], LABEL)
	#SOFTMAX
	first_softamx_derivative = softmax_derivative(SOFTMAX_NETWORK[NUMBER_OF_HIDDEN_LAYERS])
	#FIRST ERROR
	error = np.dot(first_cross_entropy_derivative, first_softamx_derivative)
	
	DELTA_WEIGHTS = []
	
	#We add the LAST delta weights in FRONT of DELTA_WEIGHTS.
	if(NUMBER_OF_HIDDEN_LAYERS == 0):
		DELTA_WEIGHTS.append(np.dot(INPUTS.T, error)) #Inputs is horizontal.  Need to transpose.
	else:
		DELTA_WEIGHTS.append(np.dot(SOFTMAX_NETWORK[NUMBER_OF_HIDDEN_LAYERS-1].T, error)) #SOFTMAX_NETWORK[i] is horizontal as well.
	
	#The looping part.
	for j in range(NUMBER_OF_HIDDEN_LAYERS): #Remember, j starts from 0.
		tempderiv1 = WEIGHT_NETWORK[NUMBER_OF_HIDDEN_LAYERS-j].T #Needs transposing.
		tempderiv2 = softmax_derivative(SOFTMAX_NETWORK[NUMBER_OF_HIDDEN_LAYERS-1-j])
		
		#Second, third, fourth... error
		error = np.dot(np.dot(error, tempderiv1), tempderiv2)
		if(j == NUMBER_OF_HIDDEN_LAYERS-1): #i.e. If j is at its last iteration
			DELTA_WEIGHTS.append(np.dot(INPUTS.T, error)) #Again, INPUTS is horizontal, which we do not want now. We need vertical.
		else:
			DELTA_WEIGHTS.append(np.dot(SOFTMAX_NETWORK[NUMBER_OF_HIDDEN_LAYERS-2-j].T, error)) #Convert hori to verti. Also, -2 @ the index. See diagram to understand.
			
	#Apply the deltas to weights
	for j in range(NUMBER_OF_HIDDEN_LAYERS+1): #There are NUMBER_OF_HIDDEN_LAYERS+1 amount of weight matrices in WEIGHT_NETWORK
		WEIGHT_NETWORK[NUMBER_OF_HIDDEN_LAYERS-j] = WEIGHT_NETWORK[NUMBER_OF_HIDDEN_LAYERS-j] - DELTA_WEIGHTS[j] * LEARINING_RATE
	
	#print DELTA_WEIGHTS[1]
print("FINISHED")