import numpy as np
from Layers.Base import BaseLayer
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        return input_tensor.reshape(batch_size ,-1)
    
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor.reshape(self.input_shape)