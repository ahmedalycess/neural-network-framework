
class BaseLayer:
    '''
        Base class: inherited by all layers
    '''
    def __init__(self):
        self.trainable: bool = False
        self.weights: list[float] = []