import copy
import numpy as np
from Layers.Base import BaseLayer
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.FullyConnected import FullyConnected


class RNN(BaseLayer):
    '''
    RNN Layer
    '''

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        '''
        Constructor
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        '''

        super().__init__()
        self.trainable = True
        self._memorize = False  # Memorize the hidden state for the next batch
        self._optimizer = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_tensor = None
        self.total_time_steps: int = None

        self.hidden_t = None
        self.output_t = None

        self.hidden_t_prev = None

        self.fully_connected1_memory = None
        self.fully_connected1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)

        self.weights = self.fully_connected1.weights
        self.fc1_weights_gradient = None

        self.tanh = TanH()

        self.fully_connected2_memory = None
        self.fully_connected2 = FullyConnected(self.hidden_size, self.output_size)
        self.fc2_weights_gradient = None

        self.sigmoid = Sigmoid()

    def forward(self, input_tensor):
        """
        Forward Pass
        :param input_tensor: np.ndarray
        :return: np.ndarray

        Function:
        - Concatenate the input tensor with the hidden state
        - Pass the concatenated tensor through a fully connected layer
        - Pass the output of the fully connected layer through a tanh activation function

        - Pass the output of the tanh activation function through a fully connected layer
        - Pass the output of the fully connected layer through a sigmoid activation function

        - Store the output of the sigmoid activation function in the output tensor
        - Store the output of the tanh activation function in the hidden tensor
        """

        self.input_tensor = input_tensor
        self.total_time_steps = self.input_tensor.shape[0]
        self.output_t = np.zeros((self.total_time_steps, self.output_size))
        self.hidden_t = np.zeros((self.total_time_steps + 1, self.hidden_size))

        self.hidden_t[0] = self.hidden_t_prev if self._memorize and self.hidden_t_prev is not None \
            else np.zeros((1, self.hidden_size))

        self.fully_connected1_memory = []
        self.fully_connected2_memory = []

        for t in range(self.total_time_steps):
            input_hidden = np.hstack((input_tensor[t].reshape(1, -1), self.hidden_t[t].reshape(1, -1)))

            intermediate_output = self.fully_connected1.forward(input_hidden)
            self.fully_connected1_memory.append(self.fully_connected1.input)

            self.hidden_t[t + 1] = self.tanh.forward(intermediate_output)

            fully_connected2 = self.fully_connected2.forward(self.hidden_t[t + 1].reshape(1, -1))
            self.fully_connected2_memory.append(self.fully_connected2.input)
            self.output_t[t] = self.sigmoid.forward(fully_connected2)

        self.hidden_t_prev = self.hidden_t[self.total_time_steps]
        return self.output_t

    def backward(self, error_tensor):

        """
        Backward Pass
        :param error_tensor: np.ndarray
        :return: np.ndarray

        Function:
        - Follow the steps of the forward pass in reverse order
        - Update the weights of the fully connected layers
        """

        hidden_t_prev_error = np.zeros((1, self.hidden_size))
        output_error_tensor = np.zeros((self.total_time_steps, self.input_size))

        self.fc1_weights_gradient = np.zeros_like(self.fully_connected1.weights)
        self.fc2_weights_gradient = np.zeros_like(self.fully_connected2.weights)

        for t in reversed(range(self.total_time_steps)):
            self.sigmoid.output_tensor = self.output_t[t]
            d_sigmoid = self.sigmoid.backward(error_tensor[t].reshape(1, -1))

            self.fully_connected2.input = self.fully_connected2_memory[t]
            fc2_error = self.fully_connected2.backward(d_sigmoid)
            self.fc2_weights_gradient += self.fully_connected2.gradient_weights

            # Calculate the error after the tanh activation function
            self.tanh.output_tensor = self.hidden_t[t + 1]
            tanh_error = self.tanh.backward(hidden_t_prev_error + fc2_error)

            # Calculate the error after the fully connected layer
            self.fully_connected1.input = self.fully_connected1_memory[t]
            fc1_error = self.fully_connected1.backward(tanh_error)

            # Calculate the gradient of the weights
            self.fc1_weights_gradient += self.fully_connected1.gradient_weights

            input_error = fc1_error[:, :self.input_size]
            hidden_t_prev_error = fc1_error[:, self.input_size:]

            output_error_tensor[t] = input_error

        # Update the weights of the two fully connected layers
        if self.optimizer is not None:
            self.fully_connected1.weights = self._optimizer.calculate_update(self.fully_connected1.weights,
                                                                             self.fc1_weights_gradient)
            self.fully_connected2.weights = self._optimizer.calculate_update(self.fully_connected2.weights,
                                                                             self.fc2_weights_gradient)
        return output_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.fully_connected1.initialize(weights_initializer, bias_initializer)
        self.fully_connected2.initialize(weights_initializer, bias_initializer)
        self.weights = self.fully_connected1.weights

    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.fully_connected1.weights)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def weights(self):
        return self.fully_connected1.weights

    @weights.setter
    def weights(self, weights):
        self.fully_connected1.weights = weights

    @property
    def gradient_weights(self):
        return self.fc1_weights_gradient

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.fully_connected1._gradient_weights = gradient_weights
