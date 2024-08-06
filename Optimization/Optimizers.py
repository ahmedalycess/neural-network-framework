import numpy as np

class Sgd:
    '''
        Stochastic Gradient Descent optimizer class
        
        Equation:
         w_new = w_old - learning_rate * gradient of loss function w.r.t weight tensor
        
        Position in the NN pipeline: 
         After the backward pass to update the weights of the model
    '''
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        return weight_tensor - self.learning_rate * gradient_tensor
    

