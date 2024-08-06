import numpy as np
from Layers.Base import BaseLayer
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the Flatten layer
        :params: input tensor
        :return: reshaped tensor
        '''
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)
    
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        '''
        Backward pass of the Flatten layer
        :params: error tensor
        :return: reshaped tensor
        '''
        return error_tensor.reshape(self.input_shape)