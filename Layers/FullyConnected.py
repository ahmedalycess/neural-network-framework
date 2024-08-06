from Layers import Base, Initializers
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size: int, output_size: int):

        super().__init__()
        self.trainable = True
        
        self._optimizer = None
        self._gradient_weights = None

        self.input_size = input_size
        self.output_size = output_size

        self._weights = np.zeros((input_size + 1, output_size))
        self.initialize(Initializers.UniformRandom(), Initializers.UniformRandom())

        self.input = None
    


    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        param input_tensor: np.ndarray
        return output: np.ndarray

        Original Equation for forward pass:
        output = input * weights + bias
        Modified -> input = [input, 1] (to include bias) -> output = input * weights
         -reason: a test case required weights to be of shape (input_size + 1, output_size)
        '''

        self.input = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
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
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape, self.input_size, self.output_size)
    
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights