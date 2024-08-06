import numpy as np

class Optimizer:
    '''
        Optimizer class skeleton
        Methods:
            - calculate_update:
                - calculates the update to the weight tensor
    '''
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer) -> None:
        self.regularizer = regularizer

    def apply_regularizer(self, weight_tensor: np.ndarray):
        return self.regularizer.calculate_gradient(weight_tensor) 


class Sgd(Optimizer):
    '''
        Stochastic Gradient Descent optimizer class
        
        Equation:
         w_new = w_old - learning_rate * gradient of loss function w.r.t weight tensor
        
        Position in the NN pipeline: 
         After the backward pass to update the weights of the model
    '''
    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:

        weight_tensor_copy = np.copy(weight_tensor) if type(weight_tensor) is np.ndarray else weight_tensor
        
        weight_tensor -= self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.apply_regularizer(weight_tensor_copy)
        return weight_tensor

class SgdWithMomentum(Optimizer):
    '''
        Stochastic Gradient Descent with Momentum optimizer class
        
        Equation:
         v = momentum_rate * v _ learning_rate * gradient of loss function w.r.t weight tensor
         w_new = w_old - v
        
        Position in the NN pipeline: 
         After the backward pass to update the weights of the model
    '''
    def __init__(self, learning_rate: float, momentum_rate: float) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        weight_tensor_copy = np.copy(weight_tensor) if type(weight_tensor) is np.ndarray else weight_tensor

        self.velocity = self.momentum_rate * self.velocity + self.learning_rate * gradient_tensor
        weight_tensor -= self.velocity
        
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.apply_regularizer(weight_tensor_copy)
        
        return weight_tensor
    
class Adam(Optimizer):
    '''
        Adam optimizer class
        
        Equations:
         v = beta1 * v + (1 - beta1) * gradient of loss function w.r.t weight tensor
         r = beta2 * r + (1 - beta2) * gradient of loss function w.r.t weight tensor^2
         v_hat = v / (1 - beta1^k)
         r_hat = r / (1 - beta2^k)
         w_new = w_old - learning_rate * v_hat / (sqrt(r_hat) + epsilon)
        
        Position in the NN pipeline: 
         After the backward pass to update the weights of the model
    '''
    def __init__(self, learning_rate: float = 0.001, mu: float = 0.9, rho: float = 0.999) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1 # bias correction term
        self.epsilon = np.finfo(float).eps

    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        weight_tensor_copy = np.copy(weight_tensor) if type(weight_tensor) is np.ndarray else weight_tensor
        
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.power(gradient_tensor, 2)
        
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))

        self.k += 1
        weight_tensor -= self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon)
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.apply_regularizer(weight_tensor_copy)
        return weight_tensor

    

