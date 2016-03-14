#!/usr/bin/python

#Coding an artificial neural network for binary classification

import numpy as np

a = np.load('input_data.npy')

def main():
	a = np.load('input_data.npy')
	x = np.array([6,3,8])
	y = np.array([25,14,28])
	
	x = a[:,0]
	y = a[:,1]
	dims = [1,1,1]
	ih_weights, ho_weights = init_weights(dims)	
	ih_weights = ih_weights[0]
	eta = 0.00001
	hidden_layer = np.zeros(2)
	output_layer = np.zeros(1)
	error = 1000
	new_error = 5
	iter_count = 0
	
	print 'Training...'
	
	while abs(error - new_error) > 0.01:
		iter_count += 1
		error = new_error
		new_error = 0
		for a in xrange(len(x)):
			#print forward(x[a], ih_weights, ho_weights)
			hidden_layer, output_layer = forward(x[a], ih_weights, ho_weights)
			new_error += (y[a] - output_layer) ** 2
			ih_weights, ho_weights = backprop(output_layer, y[a], eta, ho_weights, ih_weights, hidden_layer, dims, x) 
		print 'Error: {}'.format(new_error)	

	print 'Final number of iterations: {}'.format(iter_count)
	print 'Weights for input to hidden layer: {}'.format(ih_weights)
	print 'Weights for hidden layer to output: {}'.format(ho_weights)
	
	print 'Testing...'
	x_tests = [1,6,100]
	for test in x_tests:
		hidden_layer, y_pred = forward(test, ih_weights, ho_weights)
		print 'predict y when x = {0} ...... y = {1}'.format(test, y_pred)
	print np.mean(y)
	
		
def init_weights(dims):
	weights = []
	ih_weights = np.random.choice([0.0001,-0.0001], size=(dims[1],dims[0]+1))
	ho_weights = np.random.choice([0.0001,-0.0001], size=(dims[1]+1))
	return (ih_weights, ho_weights)

def sig(theta): 
	return 1/(1+np.exp(-theta))

def forward(X, ih_weights, ho_weights):
	X = np.insert(np.array(X), 0, 1)
	hidden_layer_z = np.dot(ih_weights, X)	
	hidden_layer_o = sig(hidden_layer_z)
	hl = np.array([1, hidden_layer_o])
	output_layer_y = np.dot(ho_weights, hl)
	return (hidden_layer_o, output_layer_y)

def backprop(out, y, eta, old_ho_weights, old_ih_weights, hidden_layer, dims, x):
	hidden_layer = np.array([1, hidden_layer])
	y = np.array(y)
	new_ho_weights = old_ho_weights + eta * (y - out) * hidden_layer 
	new_ih_weights = old_ih_weights
	for j in xrange(dims[0]+1):
		for i in xrange(dims[1]+1):
# 			print eta
# 			print (y - out)
# 			print hidden_layer[i]
# 			print old_ho_weights[i]
# 			print x[j]
# 			print eta * (y - out) * old_ho_weights[i] * hidden_layer[i] * (1 - hidden_layer[i]) * x[j]
			new_ih_weights[j] += eta * (y - out) * old_ho_weights[i] * hidden_layer[i] * (1 - hidden_layer[i]) * x[j]
	return (new_ih_weights, new_ho_weights)

if __name__ == "__main__":
	main()



#def cost(
