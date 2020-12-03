# SimpleNN	

SimpleNN is a simple Neural Network implementation done via NumPy. It has a simple api and allows activations to be set on each layer of the network. 

## Install
SimpleNN requires NumPy to run, along with copy and typing to suggest types. 

To install using pip, simply run the following command:
```
pip install git+https://git@github.com/arunj18/SimpleNN.git
```
## Usage
### Network Definition
```
from SimpleNN import SimpleNN

def _activation(x): #simple sigmoid activation
    return 1/(1+np.exp(-x))

def _derivative(output): #sigmoid derivative
    return output*(1-output)


network = (Network(2).dense('hidden_layer_1',4,_activation,_derivative)
		    .dense('output_layer',1,_activation,_derivative))
```

The example code shows how to make a neural network with one input layer with 2 input neurons, one hidden layer with 4 neurons and an output layer with one neuron. The weights are not assigned until the first time the network is run. 

### Train the network
To train the network, one can simply run the train on the network as follows:
```
iterations,  = network.train(X,T,learning_rate = 0.1,max_epochs = 20000)
```

### Write your own error calculation function
SimpleNN uses squared as the cost function for the network. However, this can easily be changed by changing the network's _error_calc function as shown below:
```
def  _error_calc(self ,target: np.ndarray) -> tuple([np.ndarray,np.ndarray]):
	end = self.network_dict['end']
	error = 0.5*np.square(target - self.network_dict[end]['output']) # make your
	deriv = target-self.network_dict[end]['output']
	return  error, deriv

import types
network._error_calc = types.MethodType(_error_calc, network)
```

### Evaluate the network
Evaluating the network is pretty simple as well. It can be done in the following way:
```
predicted_ys = network.eval(X-eval) # will return a (obs x output_dims) 
```



Please reach out to me at arunjoh@gmail.com if you have any questions. 
Feel free to suggest edits and open pull requests for your improvements!

LICENSE: MIT
