# src/model.py
import numpy as np
from src import config

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(config.RANDOM_SEED)
        # He Initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.w3 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b3 = np.zeros((1, output_size))
    


    def train(self, x, y, epochs, learning_rate):
        # Your training loop from the notebook goes here
        pass

    def predict(self, x):
        # Your prediction function from the notebook goes here
        _, _, _, _, _, y_prediction = self.forward_pass(x)
        return (y_prediction > 0.5).astype(int)