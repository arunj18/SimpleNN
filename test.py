import matplotlib.pyplot as plt
import numpy as np
from SimpleNN import SimpleNN 

def _activation(x):
    return 1/(1+np.exp(-x))

def _derivative(output):
    return output*(1-output)

if __name__ == '__main__':
    X = np.array([[0,0],[0,1],[1,0],[1,1]]).reshape((4,2))
    T = np.array([[0],[1],[1],[0]]).reshape((4,1))
    
    network = SimpleNN.Network(2)
    (iterations,errors) = (network.dense('hidden_layer_1',4,_activation,_derivative)
     .dense('output_layer',1,_activation,_derivative)
     .train(X,T,learning_rate = 0.1))
    print(iterations,errors)