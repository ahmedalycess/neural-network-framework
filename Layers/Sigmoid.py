import numpy as np
from Layers import Base


class Sigmoid(Base.BaseLayer):
    '''
    Sigmoid activation function
    Forward pass:
        - calculates the sigmoid of the input
        Equation:
            output = 1 / (1 + exp(-input))
    Backward pass:
        - calculates the gradient of the loss w.r.t the input
        Equation:
            error_tensor = output * (1 - output) * error_tensor
    '''    
    def __init__(self):
        super().__init__()
        self.output_tensor = None


    def forward(self, input_tensor):
        self.output_tensor = 1 / (1 + np.exp(-input_tensor))
        return self.output_tensor


    def backward(self, error_tensor):
        return self.output_tensor * (1 - self.output_tensor) * error_tensor