import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    '''
    Pooling layer: Max pooling
    Methods:
        forward: Forward pass for the pooling layer
        backward: Backward pass for the pooling Layers
    '''

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.y_stride, self.x_stride = stride_shape
        self.y_pooling, self.x_pooling = pooling_shape

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        Forward pass for the pooling layer
            Parameters:
                input_tensor: 4D np.ndarray --> b, c, y, x : batches, channels, y, x
            Output:
                out: np.ndarray
        '''
        self.input_tensor_shape = (b, c, y, x) = input_tensor.shape
        y_pools = np.ceil((y - self.y_pooling + 1) / self.y_stride).astype(int)
        x_pools = np.ceil((x - self.x_pooling + 1) / self.x_stride).astype(int)
        output_tensor = np.zeros((b, c, y_pools, x_pools))
        self.x_max = np.zeros((b, c, y_pools, x_pools), dtype=int)
        self.y_max = np.zeros((b, c, y_pools, x_pools), dtype=int)

        # Calculate pooling windows
        y_pos = np.arange(y_pools) * self.y_stride
        x_pos = np.arange(x_pools) * self.x_stride
        y_pos_grid, x_pos_grid = np.meshgrid(y_pos, x_pos, indexing='ij')
        y_pos_grid = y_pos_grid.reshape(-1)
        x_pos_grid = x_pos_grid.reshape(-1)

        # Extract pooling regions
        pooled_regions = np.zeros((b, c, y_pools * x_pools, self.y_pooling, self.x_pooling))
        for idx, (y, x) in enumerate(zip(y_pos_grid, x_pos_grid)):
            pooled_regions[:, :, idx, :, :] = input_tensor[:, :, y:y+self.y_pooling, x:x+self.x_pooling]

        # Reshape to combine pooling regions
        pooled_regions = pooled_regions.reshape(b, c, y_pools, x_pools, -1)
        
        # Compute max indices and values
        max_indices = np.argmax(pooled_regions, axis=4)
        max_values = np.max(pooled_regions, axis=4)
        
        # Calculate the coordinates for the max values
        self.x_max = max_indices // self.x_pooling
        self.y_max = max_indices % self.x_pooling

        return max_values

    
    def backward(self, error_tensor):
        '''
        Backward pass for the pooling layer
            Parameters:
                error_tensor: np.ndarray
            Output:
                output_tensor: np.ndarray
        '''
        # Initialize the output tensor with zeros
        output_tensor = np.zeros(self.input_tensor_shape)

        # Get the shape of the pooling indices
        batch_size, channels, pooled_height, pooled_width = self.x_max.shape

        # Create a grid of indices for the output tensor
        batch_indices, channel_indices, i_indices, j_indices = np.meshgrid(
            np.arange(batch_size),
            np.arange(channels),
            np.arange(pooled_height),
            np.arange(pooled_width),
            indexing='ij'
        )

        # Calculate the indices in the input tensor where the max pooling occurred
        x_indices = i_indices * self.y_stride + self.x_max
        y_indices = j_indices * self.x_stride + self.y_max

        # Add the error tensor to the output tensor
        np.add.at(output_tensor, 
                (batch_indices, channel_indices, x_indices, y_indices), 
                error_tensor)

        return output_tensor




