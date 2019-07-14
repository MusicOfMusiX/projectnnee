import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 28 # width and length
NUMBER_OF_CLASSES = 10 #  i.e. 0, 1, 2, 3, ..., 9

NUMBER_OF_EXAMPLES = 10;
NUMBER_OF_TESTING_EXAMPLES = 10;
LEARINING_RATE = 0.005;

def read_file(name, n):
	#These data files include labels in index 0 of each line.
	#784 0-255 integer values follow.
	return np.loadtxt("MNIST/" + name, delimiter=",", max_rows=n)

def load_examples(n):
	block = read_file("mnist_train.csv",n)
	examples = np.zeros((n,IMAGE_SIZE**2))
	for i in range(n):
		for j in range(IMAGE_SIZE**2):
			examples[i,j] = block[i,j+1]
	return examples

def load_labels(n):
	#Why separate the two file reading? For CIFAR purposes. This MNIST data file has both the examples and labels.
	block = read_file("mnist_train.csv",n)
	labels = np.zeros((n))
	for i in range(n):
		labels[i] = block[i,0]
	return labels

def copy_features(examples, n): #a.k.a. load_features
	features = np.zeros((IMAGE_SIZE**2))
	for i in range(IMAGE_SIZE**2):
		if(examples[n,i] == 0):
			features[i] = 0
		elif(examples[n,i]):
			features[i] = 1
	return features
			
def copy_label(labels, n): #a.k.a.load_label
	label = np.zeros((NUMBER_OF_CLASSES))
	for i in range(NUMBER_OF_CLASSES):
		if(i == labels[n]):
			label[i] = 1
		else:
			label[i] = 0
	return label

EXAMPLES = load_examples(NUMBER_OF_EXAMPLES)
LABELS = load_labels(NUMBER_OF_EXAMPLES)
