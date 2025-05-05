import random
from neural_network import *  # Giả sử bạn đã có NeuralNetwork như ở trên

class Individual:
    def __init__(self, genome=None, hidden_size=5):
        self.input_size = 9  # 6 đặc trưng + 1 bias
        self.hidden_size = hidden_size
        self.output_size = 1

        self.total_weights = self.input_size * self.hidden_size + self.hidden_size * self.output_size

        if genome is None:
            self.genome = [random.uniform(-1, 1) for _ in range(self.total_weights)]
        else:
            self.genome = genome

        self.fitness = 0
        self.neural_network = NeuralNetwork(self.genome, hidden_size=self.hidden_size)

    def evaluate_fitness(self, score):
        self.fitness = score
