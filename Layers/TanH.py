import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    '''
    TanH activation function
    Forward pass:
        - calculates the hyperbolic tangent of the input
        Equation:
            output = tanh(input)
    Backward pass:
        - calculates the gradient of the loss w.r.t the input
        Equation:
            error_tensor = (1 - tanh^2(input)) * error_tensor
    '''
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        self.output_tensor = np.tanh(input_tensor)
        return self.output_tensor

    def backward(self, error_tensor):
        return (1 - np.square(self.output_tensor))*error_tensor