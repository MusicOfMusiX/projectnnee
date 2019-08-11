import numpy as np
import matplotlib.pyplot as plt

#For debugging
#import sys
#np.set_printoptions(threshold=sys.maxsize)

IMAGE_SIZE = 28 # width and length
NUMBER_OF_CLASSES = 10 #  i.e. 0, 1, 2, 3, ..., 9

NUMBER_OF_EXAMPLES = 4000;
NUMBER_OF_TESTING_EXAMPLES = 100;
LEARINING_RATE = 0.005;

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

def copy_features(examples, n): #a.k.a. load_features
	features = np.zeros((IMAGE_SIZE**2))
	for i in range(IMAGE_SIZE**2):
		if(examples[n,i] == 0):
			features[i] = 0
		elif(examples[n,i] != 0):
			features[i] = examples[n,i]/255
	return features
			
def copy_label(labels, n): #a.k.a.load_label
	label = np.zeros((NUMBER_OF_CLASSES))
	for i in range(NUMBER_OF_CLASSES):
		if(i == labels[n]):
			label[i] = 1
		else:
			label[i] = 0
	return label

def softmax(logits):
	probabilities = np.zeros((NUMBER_OF_CLASSES))
	sum = 0.0
	for i in range(NUMBER_OF_CLASSES):
		sum += np.exp(logits[i])
	for i in range(NUMBER_OF_CLASSES):
		probabilities[i] = np.exp(logits[i]) / sum
	return probabilities

def get_probabilities(weights, biases, features):
	logits = np.zeros((10))
	logits = np.dot(features, weights)
	logits = logits + BIASES
	
	probabilities = softmax(logits)
	return probabilities

def test(weights, biases, examples, labels):
	accsum = 0.0
	for i in range(NUMBER_OF_TESTING_EXAMPLES):
		features = copy_features(examples, i)
		probabilities = get_probabilities(weights, biases, features)
		result = np.argmax(probabilities)
		if(result == labels[i]):
			accsum += 1
	return accsum/NUMBER_OF_TESTING_EXAMPLES*100

EXAMPLES = load_examples("mnist_train.csv", NUMBER_OF_EXAMPLES)
LABELS = load_labels("mnist_train.csv", NUMBER_OF_EXAMPLES)

TESTING_EXAMPLES = load_examples("mnist_test.csv", NUMBER_OF_TESTING_EXAMPLES)
TESTING_LABELS = load_labels("mnist_test.csv", NUMBER_OF_TESTING_EXAMPLES)

WEIGHTS = np.zeros((IMAGE_SIZE**2, NUMBER_OF_CLASSES))
BIASES = np.zeros((NUMBER_OF_CLASSES))

### MAIN LOOP ###
print("Starting main loop")

for i in range(NUMBER_OF_EXAMPLES):
	FEATURES = copy_features(EXAMPLES, i)
	LABEL = copy_label(LABELS, i)
	
	PROBABILITIES = get_probabilities(WEIGHTS, BIASES, FEATURES)
	
	#Test every 50 iterations
	if(i % 50 == 0):
		accuracy = test(WEIGHTS, BIASES, TESTING_EXAMPLES, TESTING_LABELS)
		print("Iteration #" + str(i) + ": " + "Accuracy: " + str(accuracy) + "%")
	
	#The 'learning' happens here. Split the process into steps for easy understanding.
	diff = np.subtract(PROBABILITIES, LABEL)
	#We convert 1D-vectors to 2D matrices
	mat = FEATURES.reshape(IMAGE_SIZE**2,1)
	mat2 = diff.reshape(1,NUMBER_OF_CLASSES)
	dotprod = np.dot(mat,mat2) #result: IMAGE_SIZE^2 x NUMBER_OF_CLASSES matrix
	deltaweights = LEARINING_RATE * dotprod
	
	WEIGHTS = np.subtract(WEIGHTS, deltaweights)
	BIASES = np.subtract(BIASES, LEARINING_RATE * (np.subtract(PROBABILITIES, LABEL)))
	
print("FINISHED")