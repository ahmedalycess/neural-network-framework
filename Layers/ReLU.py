from Layers.Base import BaseLayer
import numpy as np
class ReLU(BaseLayer):
    '''
    ReLU activation function
    Forward pass:
     - output = max(0, input)
    Backward pass:
     - output = error_tensor * (input > 0)
    '''
    def __init__(self):
        super().__init__()
        self.output = None
    
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, input_tensor)
        return self.output
    
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor * (self.output > 0)