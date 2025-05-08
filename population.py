from individual import Individual
import random

class Population:
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size
        self.individuals = [Individual(input_size) for _ in range(size)]
        self.generation = 1

    def evolve(self):
        for ind in self.individuals:
            ind.calculate_fitness()

        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        # Lấy 2 cá thể ưu tú nhất
        elite1, elite2 = self.individuals[:2]

        
        # Giữ lại 2 cá thể ưu tú nhất
        children = [elite1.clone(), elite2.clone()]
        
        # Tạo các cá thể con từ 2 cá thể ưu tú
        while len(children) < self.size:
            # Chọn ngẫu nhiên một trong hai cá thể ưu tú làm bố mẹ
            parent = random.choice([elite1, elite2]).clone()
            parent.mutate(mutation_rate=0.1)
            children.append(parent)

        self.individuals = children
        self.generation += 1
