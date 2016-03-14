#!/Users/kennywesterman/anaconda/bin/python

import numpy as np

def main():
	alphas = [0.001,0.01,0.1,1,10,100,1000]
	hiddensize = 32
	X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
	Y = np.array([[0,1,1,0]]).T
	for alpha in alphas:
		print '\nTraining with alpha =', alpha
		np.random.seed(1)
		syn0 = 2*np.random.random((3,hiddensize)) - 1
		syn1 = 2*np.random.random((hiddensize,1)) - 1
		for i in xrange(60000):
			l1 = sigmoid(np.dot(X,syn0))
			l2 = sigmoid(np.dot(l1,syn1))
			Ez2 = (Y-l2) * sigmoid(l2,diff=True) #Scalar in the case of single output node
			if i%10000 == 0:
				print 'Error after %i iterations:' % i, np.mean(np.abs(Y-l2))
			Ez1 = np.dot(Ez2,syn1.T) * sigmoid(l1,diff=True) #Error in output propagates, through syn1 weights and the sigmoid derivative, to z of hidden layer
			syn0 += alpha * np.dot(X.T,Ez1)
			syn1 += alpha * np.dot(l1.T,Ez2)


def sigmoid(a, diff=False):
	if diff==True:
		return a * (1 - a)
	return 1 / (1 + np.exp(-a))

if __name__ == "__main__":
	main()
