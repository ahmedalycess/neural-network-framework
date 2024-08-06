from copy import deepcopy
from math import ceil
import numpy as np
from scipy.signal import correlate, convolve

from Layers.Base import BaseLayer
import warnings
from Layers.Initializers import UniformRandom
warnings.filterwarnings('ignore')

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.num_kernels = num_kernels
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape

        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self.bias = np.random.rand(num_kernels)

        self._optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor):
        '''
        Forward pass for the convolution layer
            Parameters:
                input_tensor: np.ndarray could be
                    - 1D --> b, c, y
                    - 2D --> b, c, y, x : batches, channels, y, x
            Output:
                out: np.ndarray
        '''
        self.input_tensor = input_tensor

        batches, channels, y = input_tensor.shape[:3] 
        y_out = ceil(y /self.stride_shape[0])
        if len(input_tensor.shape) == 3: 
            output_tensor = np.zeros((batches, self.num_kernels, y_out))
        else: 
            x_in = input_tensor.shape[3] 
            x_out = ceil(x_in/self.stride_shape[1]) 
            output_tensor = np.zeros((batches, self.num_kernels, y_out, x_out))

        
        for batch in range(batches):
            for kernal in range(self.num_kernels):
                val = correlate(input_tensor[batch], self.weights[kernal], "same") 
                val = val[val.shape[0] // 2] # get the middle value
                if (len(self.stride_shape) == 1):
                    val = val[::self.stride_shape[0]]  
                else:
                    val = val[::self.stride_shape[0], ::self.stride_shape[1]]
                output_tensor[batch, kernal] = val + self.bias[kernal]
        return output_tensor

    def backward(self,error_tensor):
        '''
        Backward pass for the convolution layer
            Parameters:
                error_tensor: np.ndarray
            Output:
                gradients: np.ndarray
        '''

        batches = np.shape(error_tensor)[0]  # [batches, channels, y, x]
        channels = self.convolution_shape[0] # [channels, y_kernal, x_kernal]

        weights = np.flip(np.swapaxes(self.weights, 0, 1), axis=1)
        batch_error = np.zeros((batches, self.num_kernels, *self.input_tensor.shape[2:]))
        gradients = np.zeros((batches, channels, *self.input_tensor.shape[2:]))
        for batch in range(batches):
            for channel in range(channels):
                if (len(self.stride_shape) == 1):
                    batch_error[:, :, ::self.stride_shape[0]] = error_tensor[batch]
                else:
                    batch_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[batch]
                output = convolve(batch_error[batch], weights[channel], "same")
                output = output[output.shape[0] // 2]
                gradients[batch, channel] = output

        self.gradient_weights, self.gradient_bias = self.get_gradients(error_tensor)
        if self.optimizer is not None:
            self.weights = deepcopy(self.optimizer).calculate_update(self.weights, self.gradient_weights)
            self.bias = deepcopy(self.optimizer).calculate_update(self.bias, self.gradient_bias)

        return gradients

    def get_gradients(self,error_tensor):
        '''
        Calculate the gradient of the weights and biases
            Parameters:
                error_tensor: np.ndarray
            Output:
                gradient_weights: np.ndarray
                gradient_bias: np.ndarray
        '''
        batches = np.shape(error_tensor)[0]
        channels = self.convolution_shape[0]
        gradient_weights = np.zeros((self.num_kernels, *self.convolution_shape))
        batch_error = np.zeros((batches, self.num_kernels, *self.input_tensor.shape[2:]))
        for batch in range(batches):
            if (len(self.stride_shape) == 1):
                batch_error[:, :, ::self.stride_shape[0]] = error_tensor[batch]
                gradient_bias = np.sum(error_tensor, axis=(0, 2))
                padding_width = ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2))
            else:
                batch_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[batch]
                gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
                padding_width =  ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2),
                                   (self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2))
            
            padded_input = np.pad(self.input_tensor[batch], padding_width, mode='constant', constant_values=0)
            dw_val = np.zeros((self.num_kernels, *self.convolution_shape))
            for kernal in range(self.num_kernels):
                for channel in range(channels):
                    dw_val[kernal, channel] = correlate(padded_input[channel], batch_error[batch][kernal], 'valid')
            gradient_weights += dw_val
        return gradient_weights, gradient_bias
    
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    @ property
    def optimizer(self):
        return self._optimizer

    @ optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
