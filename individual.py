from neural_network import NeuralNetwork
import numpy as np

class Individual:
    def __init__(self, input_size):
        self.brain = NeuralNetwork(input_size)
        self.fitness = 0
        self.score = 0
        self.survival_time = 0
        self.is_alive = True

    def clone(self):
        clone = Individual(self.brain.input_size)
        clone.brain = self.brain.clone()
        clone.score = self.score
        clone.survival_time = self.survival_time
        clone.fitness = self.fitness
        return clone

    def mutate(self, mutation_rate=0.1):
        self.brain.mutate(mutation_rate)

    def calculate_fitness(self):
        self.fitness = max(1, self.score * 100 + self.survival_time)
