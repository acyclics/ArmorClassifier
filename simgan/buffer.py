import numpy as np


class Buffer:
    """
        A buffer used to store old refined images
    """
    def __init__(self, size):
        self.size = size
        self.buffer = []
    
    def store(self, refined_image):
        if len(self.buffer) == self.size:
            random_idx = np.random.randint(0, self.size)
            self.buffer[random_idx] = refined_image
        else:
            self.buffer.append(refined_image)
    
    def get(self):
        random_idx = np.random.randint(0, len(self.buffer))
        return self.buffer[random_idx]
    
    @property
    def len(self):
        return len(self.buffer)
