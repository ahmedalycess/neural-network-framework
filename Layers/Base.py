
class BaseLayer:
    '''
        Base class: inherited by all layers
    '''
    def __init__(self):
        self.trainable: bool = False
        self.testing_phase: bool = False