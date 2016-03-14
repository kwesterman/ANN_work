#!/usr/bin/python

#Generate data that follows a line y = mx + b

import numpy as np
import matplotlib.pyplot as plt

m = 5
b = 10

X = np.linspace(0, 20, 50) #50 samples in range [0,20]

lin_signal = m*X + b
sin_signal = 20*np.sin(0.6*X)
noise = np.random.normal(0, 2, 50) #Noise from normal distribution w/ mean 0 and sd 2

Y_lin = lin_signal + noise
Y_sin = sin_signal + noise

plt.scatter(X,Y_sin)
#plt.show()

lin_data = np.stack((X,Y_lin))
sin_data = np.stack((X,Y_sin))

np.save('lin_data.npy', lin_data.T)
np.save('sin_data.npy', sin_data.T)
