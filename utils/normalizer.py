import numpy as np

class Normalizer:
    def __init__(self):
        self.rewards_history = []
        self.sum_rewards = 0
        self.sum_of_squares = 0
        self.mean = 0
        self.std = 1
        self.epsilon = 1e-10

    def update(self, reward):
        self.rewards_history.append(reward)
        self.sum_rewards += reward
        self.sum_of_squares += reward ** 2
        
        n = len(self.rewards_history)
        self.mean = self.sum_rewards / n
        self.std = np.sqrt((self.sum_of_squares / n) - (self.mean ** 2)) + self.epsilon

    def normalize(self, reward):
        return (reward - self.mean) / self.std

    def standardize(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-5)
