from Optimization.Optimizers import Sgd
from Optimization.Loss import CrossEntropyLoss
import numpy as np
from copy import deepcopy
class NeuralNetwork:
    '''
    Neural Network Skeleton
    Methods:
        - forward pass: 
            - calls the forward method of each layer in the architecture
            - returns the loss value from the loss layer
        - backward pass:
            - calls the backward method of each layer in the architecture
        - append_layer:
            - appends a layer to the architecture
        - train:
            - trains the network for a number of iterations
        - test:
            - tests the network on a given input_tensor
    '''

    def __init__(self, optimizer: Sgd) -> None:
        self.optimizer = optimizer
        self.loss: list[float] = []
        self.layers: list = [] 
        self.data_layer = None
        self.loss_layer: CrossEntropyLoss = None
    
    def forward(self) -> np.ndarray:
        input_tensor, self.label_tensor = deepcopy(self.data_layer.next())
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, deepcopy(self.label_tensor))
        return loss
    
    def backward(self) -> None:
        label_tensor = deepcopy(self.label_tensor)
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
    
    def append_layer(self, layer: object) -> None:
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)
    
    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()
    
    def test(self, input_tensor) -> np.ndarray:
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
        