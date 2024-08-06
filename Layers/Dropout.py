import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability
        self.mask = None
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        # if testing, no dropout is applied -> mask is all ones 
        # else 
        #      mask is binomial distribution
        self.mask = np.ones(input_tensor.shape) if self.testing_phase else \
                    np.random.binomial(1, self.probability, input_tensor.shape) / self.probability

        return input_tensor * self.mask

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor * self.mask