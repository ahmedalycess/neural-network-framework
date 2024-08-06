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

class SgdWithMomentum:

    def __init__(self, learning_rate: float, momentum_rate: float) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity
    
class Adam:

    def __init__(self, learning_rate: float = 0.001, mu: float = 0.9, rho: float = 0.999) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1 # bias correction term
        self.epsilon = 1e-8

    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:

        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.square(gradient_tensor)
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        self.k += 1
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon)

    

