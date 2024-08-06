import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.input = None
        self.epsilon = np.finfo(float).eps
    
    def forward(self, input_tensor: np.ndarray, label_tensor: np.ndarray) -> float:
        '''
            forward pass of the cross entropy loss
            Literature: H(P, Q) = -sum(P * log(Q)) : P is label_tensor, Q is input_tensor
            Modification: 
                added epsilon to avoid log(0) --> log(0) = -inf
            Inputs: 
                input_tensor: np.ndarray
                label_tensor: np.ndarray
            Expected Output: 
                loss: float
        '''
        self.input = input_tensor
        input_tensor = input_tensor[label_tensor==1]
        loss = -np.sum(np.log(input_tensor + self.epsilon))
        return loss
        
    
    def backward(self, label_tensor) -> np.ndarray:
        '''
            backward pass of the cross entropy loss
            Literature: dH/dQ = -P/Q : P is label_tensor, Q is input_tensor
            Modification: 
                added epsilon to avoid division by zero
            Inputs: 
                label_tensor: np.ndarray
            Expected Output: 
                gradient: np.ndarray
        ''' 
        gradient = -label_tensor/(self.input + self.epsilon)
        return gradient
        