import numpy as np
from Layers import Base, Helpers 
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels: int):
        super().__init__()
        self.trainable = True
        
        self.CNN_shape = None
        self.channels = channels
        self.weights, self.bias = self.initialize()
        
        
        self._optimizer = None

        self.moving_mean = None
        self.moving_var = None
        
        self.epsilon = np.finfo(float).eps
        self.decay = 0.8



        self.input_tensor = None
        self.input_tensor_hat = None

        self.gradient_weights = None
        self.gradient_bias = None
    
    def initialize(self, weights_initializer = None, bias_initializer = None):
        '''
        Initialize the weights and bias
        :param channels: number of channels
        :return: weights and bias
        '''
        weights = np.ones(self.channels)
        bias = np.zeros(self.channels)
        return weights, bias
        

    def forward(self, input_tensor):

        '''
        Forward pass of the BatchNormalization layer
        :param input_tensor: input tensor
        :return: output tensor
        '''

        # Reshape the input tensor if it is a CNN
        reshaped = False
        if input_tensor.ndim == 4:
            input_tensor = self.reformat(input_tensor)
            reshaped = True
        
        self.input_tensor = input_tensor # Save the input tensor for backpropagation

        # If testing phase, use the moving mean and variance else calculate the mean and variance and update the moving mean and variance
        if self.testing_phase:

            self.mean_b = self.moving_mean
            self.var_b = self.moving_var
        else:
            self.mean_b = np.mean(input_tensor, axis= 0)
            self.var_b = np.var(input_tensor, axis=0)
            
            # moving average: new_moving = decay * moving + (1 - decay) * current
            self.moving_mean = self.decay * self.moving_mean  + (1 - self.decay) * self.mean_b if \
                self.moving_mean is not None else self.mean_b
            self.moving_var =  self.decay * self.moving_var   + (1 - self.decay) * self.var_b  if \
                self.moving_var is not None else self.var_b
        
        # Normalize the input tensor and get the output tensor
        self.input_tensor_hat = (input_tensor - self.mean_b) / np.sqrt(self.var_b + self.epsilon)
        output_tensor = self.weights * self.input_tensor_hat + self.bias
        if reshaped:
            output_tensor = self.reformat(output_tensor)
        return output_tensor

    def backward(self, error_tensor):
        '''
        Backward pass of the BatchNormalization layer
        :param error_tensor: error tensor
        :return: gradient tensor
        '''
        
        # Reshape if CNN
        reshaped = False
        if error_tensor.ndim == 4:
            reshaped = True
            error_tensor = self.reformat(error_tensor)

        # Calculate the gradients for the weights and bias and update them
        self.gradient_weights = np.sum(error_tensor * self.input_tensor_hat, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        # Calculate the gradient tensor 
        gradient_tensor = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean_b, self.var_b)
        # Update the weights and bias using the optimizer
        if self._optimizer is not None:
            self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        # Reshape to original shape if CNN
        if reshaped:
            gradient_tensor = self.reformat(gradient_tensor)

        return gradient_tensor
    
    def reformat(self, tensor):
        '''
        Reshape the tensor to be used in the forward and backward pass
        :param tensor: input tensor
        :return: reshaped tensor
        '''
        if tensor.ndim == 4:
            self.CNN_shape = tensor.shape
            B, H, M, N = tensor.shape
            return tensor.reshape(B, H, M * N).transpose(0, 2, 1).reshape(B * M * N, H)
        else:
            B, H, M, N = self.CNN_shape
            return tensor.reshape(B, M * N, H).transpose(0, 2, 1).reshape(B, H, M, N)


    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        # create three optimizers for the weights, bias and the moving mean and variance 
        self._optimizer = copy.deepcopy(optimizer)
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)