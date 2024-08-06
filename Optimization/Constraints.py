import numpy as np

class L2_Regularizer:
    '''
    L2 Regularizer
    Equation:
        loss = alpha * sum(weights^2)
    '''

    def __init__(self, alpha: float):
        self.alpha = alpha # regularization weight
    
    def calculate_gradient(self, weights: list[float]) -> list[float]:
        # calculate sub-gradient on the weights needed for the optimizer
        return  self.alpha * weights
    
    def norm(self, weights: list[float]) -> float:
        # norm enhancement loss
        return self.alpha * np.sum(np.square(weights))
    

class L1_Regularizer:
    '''
    L1 Regularizer
    Equation:
        loss = alpha * sum(|weights|)
    '''
    
    def __init__(self, alpha: float):
        self.alpha = alpha # regularization weight
    
    def calculate_gradient(self, weights: list[float]) -> list[float]:
        # calculate sub-gradient on the weights needed for the optimizer
        return self.alpha * np.sign(weights)
    
    def norm(self, weights: list[float]) -> float:
        # norm enhancement loss
        return self.alpha * np.sum(np.abs(weights))
