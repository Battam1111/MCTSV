import numpy as np

class Normalizer:
    def __init__(self):
        self.rewards_history = []
        self.mean = 0
        self.std = 1
        self.epsilon = 1e-10

    def update(self, reward):
        self.rewards_history.append(reward)
        self.mean = np.mean(self.rewards_history)
        self.std = np.std(self.rewards_history) + self.epsilon

    def normalize(self, reward):
        return (reward - self.mean) / self.std
    
    def standardize(self, tensor):
        """标准化给定的张量到均值为0，标准差为1."""
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-5)