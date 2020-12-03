import numpy as np
from copy import deepcopy as copy
from typing import Callable,Optional

class Network(object):
    def __init__(self,input_num: int) -> None:
        '''
        Description: Network initializer
        Parameters:
            input_num:
                Type: int
                Description: Number of inputs to the network
        '''
        self.network_dict = {}
        self.network_dict['input_num'] = input_num
        self.network_dict['begin'] = None
        self.network_dict['end'] = None
        self.layers = []
 
    def _forward_pass(self,input_num: int, 
                  no_neurons: int, 
                  X: np.ndarray,
                  activation: Callable[[np.ndarray],np.ndarray], 
                  W: Optional[np.ndarray] = None, 
                  b: Optional[np.ndarray] = None) -> np.ndarray:
        '''
        Description: Forward pass function for a layer which initializes W and b if not initialized
                    and performs the activation function
        Parameters:
            input_num: 
                Type: int
                Description: Number of inputs to the layer
            no_neurons:
                Type: int
                Description: Number of neurons in the layer
            X:
                Type: NumPy array of shape (input_num,1)
                Description: Input to the neuron from previous layer 
            activation:
                Type: function 
                Description: Function which performs the activation, takes NumPy array as input
            W:
                Type: NumPy array of shape (no_neurons,input_num) or None
                Description: Weights to the layer neurons from inputs
            b:
                Type: NumPy array of shape (no_neurons,1) or None
                Description: Bias to the layer neurons

        Returns: 
            Type: NumPy array
            Description: array with activation applied
        '''
        if W is None:
            W = np.random.rand(no_neurons,input_num)

        if b is None:
            b = np.random.rand(no_neurons,1)

        return activation(np.matmul(W,X) + b),W,b 
    
    def _update_Wb(self,W: np.ndarray,
               b: np.ndarray,
               dW: np.ndarray,
               db: np.ndarray,
               alpha: float) -> tuple(np.ndarray,np.ndarray) :
        '''
        Description: Function to update W and b, given dW, db and learning rate alpha
        Parameters:
            W:
                Type: NumPy array
                Description: Weights to the layer neurons from inputs
            b:
                Type: NumPy array of shape (no_neurons,1) or None
                Description: Bias to the layer neurons
            dW:
                Type: NumPy array of shape (no_neurons,input_num) or None
                Description: changes to weights
            db:
                Type: NumPy array of shape (no_neurons,1) or None
                Description: changes to the Bias
            alpha:
                Type: float
                Description: learning rate
        Returns:
            W:
                Type: NumPy array
                Description: Updated weights using learning rule
            b:
                Type: NumPy array
                Description: Updated biases using learning rule
        '''
        W = W + alpha*dW
        b = b + alpha*db
        return W,b

    def _feed_forward(self,X: np.ndarray) -> None:
        '''
        Description: Feed forward function which takes an input X and performs the forward pass through 
        the layers, stores it in the output of each layer
        Parameters:
            X:
                Type: NumPy array of shape (input_num,1)
                Description: Input to the neuron from previous layer 
        '''
        start = self.network_dict['begin'] # start from the beginning of the network
        dim_in = self.network_dict['input_num'] # dimension is the input dimension at first

        X_in = copy(X) #make a deep copy just to make sure no changes are made to original data

        while(start is not None):
            layer = self.network_dict[start] 
            layer['output'], layer['W'], layer['b'] = self._forward_pass(dim_in,
                                                                    layer['no_neurons'],
                                                                    X_in,
                                                                    layer['activation'],
                                                                    layer['W'],
                                                                    layer['b'])
            layer['input'] = X_in
            start = layer['next'] #move to the next layer
            X_in = layer['output']
            dim_in = layer['no_neurons'] #change input dimension to layer output neuron
    
    def _error_calc(self ,target: np.ndarray) -> tuple(np.ndarray,np.ndarray):
        '''
        Description: Error calculation function, takes target vector and calculates error and derivative of cost function
        Parameters:
            target:
                Type: NumPy array of shape (output_neuron,1)
        returns:
            error:
                Type: NumPy array
                Description: error in network output 
            deriv:
                Type: NumPy array
                Description: derivative of cost function that needs to be backpropagated
            
        '''
        end = self.network_dict['end']
        error = 0.5*np.square(target - self.network_dict[end]['output'])
        deriv = target-self.network_dict[end]['output']
        return error, deriv
    
    def _back_prop(self,error):
        '''
        Description: back propagation function, uses derivative of error calc to calculate deltas
            error:
                Type: NumPy array of shape (output_neuron,1)
                Description: Derivative of cost function 
            
        '''
        end = self.network_dict['end'] # back prop starts from the end

        error_back = copy(error)

        while (end is not None):
            layer = self.network_dict[end]
            if (layer['next'] is not None):
                 error_back = np.matmul(self.network_dict[layer['next']]['W'].T,
                                       self.network_dict[layer['next']]['error_back'])

            activ_deriv = layer['derivative'](layer['output'])

            if layer['dW'] is None:
                layer['dW'] = np.matmul(error_back * activ_deriv, 
                                        layer['input'].T)
            else:
                layer['dW'] += np.matmul(error_back * activ_deriv, 
                                        layer['input'].T)
            if layer['db'] is None:
                layer['db'] = error_back * activ_deriv
            else:
                layer['db'] += error_back * activ_deriv
                
            layer['error_back'] = error_back * activ_deriv
            end = layer['prev']

     
    def _make_updates(self,learning_rate):
        '''
        Description: function that makes changes to the weights of the network
        Parameters:
            learning_rate:
                Type: float
                Description: learning rate for the updates
            
        '''
        for key in self.layers:
            layer = self.network_dict[key]
            layer['W'],layer['b'] = self._update_Wb(layer['W'],layer['b'],layer['dW'],layer['db'],learning_rate)
            layer['dW'] = None
            layer['db'] = None
                
    def dense(self,name:str, 
              no_neurons: int,
              activation: Callable[[np.ndarray],np.ndarray], 
              derivative: Callable[[np.ndarray],np.ndarray]) -> None:
        '''
        Description: Function to add a new dense layer to the network
        Parameters:
            name: 
                Type: str
                Description: Name of the new layer of the network
            no_neurons:
                Type: int
                Description: Number of neurons in the layer
            activation:
                Type: function
                Description: function to calculate activation given the output
            derivative
                Type: function
                Description: function to calculate derivative given the output activations
        returns:
            object to allow chaining
        '''
        
        if name in self.network_dict.keys():
            raise NameError('layer with this name already exists!')
            
        if no_neurons <= 0:
            raise ValueError('layer cannot have less than one neuron!')
            
        self.network_dict[name] = {}
        self.network_dict[name]['no_neurons'] = no_neurons
        self.network_dict[name]['activation'] = activation
        self.network_dict[name]['derivative'] = derivative
        self.network_dict[name]['prev'] = self.network_dict['end']
        self.network_dict[name]['next'] = None
        if self.network_dict['end'] is not None:
            self.network_dict[self.network_dict['end']]['next'] = name
        self.network_dict['end'] = name
        self.network_dict[name]['W'] = None
        self.network_dict[name]['b'] = None
        self.network_dict[name]['dW'] = None
        self.network_dict[name]['db'] = None
        self.network_dict[name]['error_back'] = None
        self.network_dict[name]['output'] = None
        self.layers.append(name)
        
        
        if self.network_dict['begin'] is None:
            self.network_dict['begin'] = name
        return self
            
    def train(self,X: np.ndarray , target: np.ndarray ,learning_rate: float, max_epoch: int = 20000) -> tuple(list,list):
        '''
        Description: function to train the network
        Parameters:
            X:
                Type: NumPy array of shape (number of observations,input_num)
                Description: Input to the network
            target:
                Type: NumPy array of shape (number of observations,1)
                Description: Target value of output of network
            learning rate:
                Type: float
                Description: learning rate for the network
            max_epochs:
                type: int
                Description: max. number of epochs to run training for
        returns:
            epoch at which it converges, list with errors in each epoch
            
        '''
        errors = []
        for i in range(max_epoch):
            epoch_error = 0
            for j in range(X.shape[0]):
                self._feed_forward(X[j,np.newaxis].T)

                error,deriv = self._error_calc(target[j,np.newaxis])

                
                epoch_error += abs(error)
                self._back_prop(deriv)
                self._make_updates(learning_rate)


            errors.append(epoch_error[0,:])
            if epoch_error <= 0.01:
                return i+1,errors
    