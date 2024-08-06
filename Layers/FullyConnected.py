from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size: int, output_size: int):

        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(input_size + 1, output_size))

        self._optimizer = None
        self.gradient_weights = None
        self.input_size = input_size
        self.output_size = output_size
        
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if optimizer is None or not hasattr(optimizer, 'calculate_update'):
            raise ValueError("Please provide an optimizer")
        self._optimizer = optimizer

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        param input_tensor: np.ndarray
        return output: np.ndarray

        Original Equation for forward pass:
        output = input * weights + bias
        Modified -> input = [input, 1] (to include bias) -> output = input * weights
         -reason: a test case required weights to be of shape (input_size + 1, output_size)
        '''

        self.input = np.c_[input_tensor, np.ones((input_tensor.shape[0], 1))]
        output = np.dot(self.input, self.weights)
        return output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        '''
        param error_tensor: np.ndarray
        return np.ndarray

        Equations for backward pass:
        1- calculate the gradient of the loss w.r.t the weights
            gradient_weights = input.T * error_tensor
        2- update the weights using the optimizer
            weights = optimizer.calculate_update(weights, gradient_weights)
        3- return the error tensor for the next layer
            return error_tensor * weights.T
        '''
        self.gradient_weights = np.dot(self.input.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        
        error_tensor = np.dot(error_tensor, self.weights[:-1].T)
        return error_tensor
    
    def initialize(self, weights_initializer, bias_initializer):
        '''
        param weights_initializer: Initializer
        param bias_initializer: Initializer
        return None
        '''
        self.weights[:, :-1] = weights_initializer.initialize(self.weights[:, :-1].shape, self.input_size, self.output_size)
        self.weights[:, -1] = bias_initializer.initialize(self.weights[:, -1].shape, 1, self.output_size)
