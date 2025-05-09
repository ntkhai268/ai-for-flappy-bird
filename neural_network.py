import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=6, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def clone(self):
        clone = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        clone.W1 = np.copy(self.W1)
        clone.b1 = np.copy(self.b1)
        clone.W2 = np.copy(self.W2)
        clone.b2 = np.copy(self.b2)
        return clone

  
    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return a2[0][0]

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
