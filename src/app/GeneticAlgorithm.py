
from types import FunctionType
from typing import Dict, List, Tuple
from .Chromosome import Chromosome, Limit
import random
import numpy as np

### Create initial population
### Calcute their fitnesses
### Select best fitnesses (rank-based selection)
### Recombine best fitnesses
### Mutate


class GeneticAlgorithm:
    def __init__(self, fitness_func: FunctionType, limits: Dict[str, Limit], *,
        population_size=10, mutation_rate=0.05, crossover_rate=0.9, generations=20):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.limits = limits
        self.fitness_func = fitness_func

        self.best_fitnesses: List[float] = []
        self.avg_fitnesses: List[float] = []
        self.worst_fitnesses: List[float] = []

        self.population: List[Chromosome] = [] 
        self.create_population()

    def start(self):
        print("Starting Genetic Algorithm...")

        print("Calculating Initial Fitnesses...")
        self.calculate_fitnesses()
        print("Initial Generation Fitnesses:")
        print(self.get_fitness_info())
        self.log_fitnesses()

        for i in range(self.generations):
            new_population: List[Chromosome] = []

            parents = self.rank_based_selection(self.population)
            for parent1, parent2 in parents:
                child1, child2 = parent1.crossover(parent2, self.crossover_rate)
                child1.mutate(self.mutation_rate)
                child2.mutate(self.mutation_rate)
                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population

            print(f"Calculating Fitnesses for Gen {i}")
            self.calculate_fitnesses()
            print(f"Gen {i} fitnesses:")
            print(self.get_fitness_info())
            self.log_fitnesses()

        max_index = np.argmax(self.population)
        print(f"Fittest chromosome was #{max_index} with a fitness of: {self.population[max_index].fitness}")

    def log_fitnesses(self):
        self.best_fitnesses.append(np.max(self.population))
        self.worst_fitnesses.append(np.min(self.population))
        self.avg_fitnesses.append(np.average(self.population))

    def get_fittest(self) -> Chromosome:
        max_index = np.argmax(self.population)
        return self.population[max_index]


    def get_fitness_info(self):
        ret = ""
        for index, chromosome in enumerate(self.population):
            ret += f"{index + 1}. {chromosome.fitness}\n"
        best_index = np.argmax(self.population)
        worst_index = np.argmin(self.population)
        ret += f"\nBest fitness is chromosome #{best_index} at {self.population[best_index].fitness}\n"
        ret += f"Worst fitness is chromosome #{worst_index} at {self.population[worst_index].fitness}\n"
        ret += f"Average fitness is {np.average([x.fitness for x in self.population])}\n"
        return ret

    def calculate_fitnesses(self):
        for chromosome in self.population:
            chromosome.fitness = self.fitness_func(chromosome)

    
    def rank_based_selection(self, population: List[Chromosome]) -> List[Tuple[Chromosome, Chromosome]]:
        population.sort(reverse=True)
        ranks = [index + 1 for index, x in enumerate(population)]
        weights = [1 / x for x in ranks] # Worse ranks get smaller weights
        parent_list = random.choices(population, weights=weights, k=len(population))
        
        # Change format to [[parent1, parent2], [parent3, parent4] ... etc]
        parents: List[Tuple[Chromosome, Chromosome]] = []
        for i in range(0, self.population_size, 2):
            parent1 = parent_list[i]
            parent2 = parent_list[i + 1]
            parents.append((parent1, parent2))

        return parents


    def create_population(self):
        for i in range(self.population_size):
            self.population.append(Chromosome(self.limits))
