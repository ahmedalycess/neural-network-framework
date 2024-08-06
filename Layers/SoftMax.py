from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    '''
    SoftMax activation function
    Forward pass:
        - shifts inputs be zero-centered to avoid numerical instability
        - calculates the exponential of the shifted input
        - normalizes the exponential values
        Equation:
            output = exp(input - max(input)) / sum(exp(input - max(input)))
    Backward pass:
        - calculates the gradient of the loss w.r.t the input
        - calculates the new error_tensor --> error_tensor = gradients - self.output * sum(gradient)
        Equation:
            error_tensor = gradients - output * sum(gradient)
    '''
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, input_tensor):
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        
        gradients = self.output * error_tensor
        sum_gradient = np.sum(gradients, axis=1, keepdims=True)
        error_tensor = gradients - self.output * sum_gradient
        return error_tensor

