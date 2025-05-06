from individual import *
import random

class Population:
    def __init__(self, size):
        """
        Khởi tạo một quần thể với số lượng cá thể nhất định.
        """
        self.size = size
        self.individuals = []  # Danh sách các cá thể
        self.generation = 0  # Thế hệ hiện tại

    def select(self):
        """
        Chọn lọc các cá thể tốt nhất để tạo ra thế hệ tiếp theo.
        """
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        selected_individuals = sorted_individuals[:self.size // 2]
        return selected_individuals

    def crossover(self, parent1, parent2):
        """
        Lai ghép các cá thể để tạo ra cá thể mới.
        """
        crossover_point = random.randint(0, len(parent1.genome) - 1)
        genome1 = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        genome2 = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]
        child1 = Individual(genome1)
        child2 = Individual(genome2)
        return child1, child2

    def mutate(self):
        """
        Đột biến các cá thể trong quần thể.
        """
        mutation_rate = 0.5
        for individual in self.individuals:
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, len(individual.genome) - 1)
                individual.genome[mutation_point] = random.uniform(-1, 1)
                individual.neural_network = NeuralNetwork(individual.genome)

    def evolve(self):
        """
        Tiến hóa quần thể qua các bước: elitism, chọn lọc, lai ghép, đột biến.
        """
        # Elitism: giữ lại cá thể tốt nhất
        best = max(self.individuals, key=lambda x: x.fitness)
        print(best.fitness)

        selected = self.select()
        new_individuals = []

        # Lai ghép theo cặp
        for i in range(0, len(selected) - 1, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            new_individuals.extend([child1, child2])

        # Bổ sung thêm cá thể để đủ số lượng
        while len(new_individuals) < self.size - 1:
            clone = random.choice(selected)
            new_individuals.append(Individual(clone.genome.copy()))

        # Cập nhật quần thể: giữ lại cá thể tốt nhất + phần còn lại
        self.individuals = [best] + new_individuals
        self.mutate()
        self.generation += 1
