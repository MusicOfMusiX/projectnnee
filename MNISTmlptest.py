import numpy as np
import matplotlib.pyplot as plt

###macrodef###

IMAGE_SIZE = 28 # width and length
NUMBER_OF_CLASSES = 10 #  i.e. 0, 1, 2, 3, ..., 9

NUMBER_OF_HIDDEN_LAYERS = 0 #This can be 0
NUMBER_OF_NODES_PER_LAYER = 8

NUMBER_OF_EXAMPLES = 1000; #We perform stochasitc gradient descent; this is equivalent to the # of epochs.
NUMBER_OF_TESTING_EXAMPLES = 100;
LEARINING_RATE = 0.005;

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
	example = np.zeros((1, IMAGE_SIZE**2))
	for i in range(IMAGE_SIZE**2):
		if(examples[n,i] == 0):
			example[0,i] = 0
		elif(examples[n,i] != 0):
			example[0,i] = examples[n,i]/255
	return example
			
def copy_single_label(labels, n): #Convert single label to one-hot vector
	#TODO: Convert vector to horizontal (check whether labels should be horizontal first.)
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

def forward_propagate(network, input):
	#Note that the network matrix is a list of matrices.
	probs = softmax(np.dot(input, network[0]))
	for i in range(NUMBER_OF_HIDDEN_LAYERS):
		probs = softmax(np.dot(probs, network[i+1]))
	return probs
	
def test(network, examples, labels):
	accsum = 0.0
	for i in range(NUMBER_OF_TESTING_EXAMPLES):
		features = copy_single_example(examples, i)
		probabilities = forward_propagate(network, features)	
		result = np.argmax(probabilities)
		if(result == labels[i]):
			accsum += 1
	return accsum/NUMBER_OF_TESTING_EXAMPLES*100

###backprop funcdef###

def softmax_derivative(output_softmax_layer):
	#WARNING: Softmax is a vector -> vector function! There is no single 'derivative', there are NxN partial derivatives instead.
	#Input softmax_layer vector is horizontal: np.array((1,n))
	#Read this: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
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

#def backward_propagate(input_layer, output_layer):
	

###__init__###

EXAMPLES = load_examples("mnist_train.csv", NUMBER_OF_EXAMPLES)
LABELS = load_labels("mnist_train.csv", NUMBER_OF_EXAMPLES)

TESTING_EXAMPLES = load_examples("mnist_test.csv", NUMBER_OF_TESTING_EXAMPLES)
TESTING_LABELS = load_labels("mnist_test.csv", NUMBER_OF_TESTING_EXAMPLES)

NETWORK = []

if(NUMBER_OF_HIDDEN_LAYERS == 0): #SLP
	weights = np.zeros((IMAGE_SIZE**2, NUMBER_OF_CLASSES))
	NETWORK.append(weights)

elif(NUMBER_OF_HIDDEN_LAYERS > 0): #MLP
	#InputL -> 1st HiddenL weights
	initial_weights = np.zeros((IMAGE_SIZE**2, NUMBER_OF_NODES_PER_LAYER))
	NETWORK.append(initial_weights)
	#HiddenL -> HiddenL weights
	hidden_weights = np.zeros((NUMBER_OF_NODES_PER_LAYER, NUMBER_OF_NODES_PER_LAYER))
	for i in range(NUMBER_OF_HIDDEN_LAYERS-1):
		NETWORK.append(hidden_weights)
	#Last HiddenL -> OutputL weights
	output_weights = np.zeros((NUMBER_OF_NODES_PER_LAYER, NUMBER_OF_CLASSES))
	NETWORK.append(output_weights)

### MAIN LOOP ###
print("Starting main loop")

for i in range(NUMBER_OF_EXAMPLES): #Iterate thru each epoch
	INPUTS = copy_single_example(EXAMPLES, i)
	LABEL = copy_single_label(LABELS, i)

	probs = forward_propagate(NETWORK, INPUTS)
	
	#Test every 50 iterations
	if(i % 50 == 0):
		accuracy = test(NETWORK, TESTING_EXAMPLES, TESTING_LABELS)
		print("Iteration #" + str(i) + ": " + "Accuracy: " + str(accuracy) + "%")
	
	#Let's test out a no-hidden-layer setup:
	#INPUT * Cross_deriv * Softmax_deriv
	#X * C' * S'
	
	deriv = cross_entropy_derivative(probs, LABEL)
	deriv2 = softmax_derivative(probs)
	first_error = np.dot(deriv, deriv2)
	
	delta_weights = np.dot(INPUTS.T, first_error)
	
	NETWORK[0] = NETWORK[0] - delta_weights * LEARINING_RATE
	
print("FINISHED")