import numpy as np

class Constant:
    def __init__(self, value = 0.1):
        self.value = value
    
    def initialize(self, weights_shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        return np.full(weights_shape, self.value)
    
class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.uniform(0, 1, weights_shape)
    

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(2 / (fan_in + fan_out))
        # zero mean gaussian
        return np.random.normal(0, limit, weights_shape)

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape: tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(2 / fan_in)
        # zero mean gaussian
        return np.random.normal(0, limit, weights_shape)
