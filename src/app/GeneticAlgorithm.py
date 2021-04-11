
from datetime import datetime
import os
from typing import Callable, Dict, List, Tuple
from .Chromosome import Chromosome, Limit
import random
import numpy as np

### Create initial population
### Calcute their fitnesses
### Select best fitnesses (rank-based selection)
### Recombine best fitnesses
### Mutate


class GeneticAlgorithm:
    def __init__(self, limits: Dict[str, Limit], fitness_func: Callable[[Chromosome], float], *,
        population_size=10, mutation_rate=0.05, crossover_rate=0.9, generations=20, log_file:str=None):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.limits = limits
        self.fitness_func = fitness_func
        self.log_file = f"{os.environ['WORKSPACE']}/{log_file}-{datetime.now().timestamp()}"
        open(self.log_file, 'a').close() ## Create log file
        

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
        self.log_fitnesses(generation=0)

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
            self.log_fitnesses(generation=(i + 1))

        max_index = np.argmax(self.population)
        print(f"Fittest chromosome was #{max_index} with a fitness of: {self.population[max_index].fitness}")
        print(f"Chromosome values:")
        print(self.population[max_index].values)

    def log_fitnesses(self, *, generation: int):
        best = max(self.population)
        self.best_fitnesses.append(best.fitness)
        self.worst_fitnesses.append(min(self.population).fitness)
        self.avg_fitnesses.append(np.average([x.fitness for x in self.population]))

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'a') as file:
                file.write("Generation,Fitness (R Square),MAE,Hidden Layers,Neurons Per Layer,Dropout,Initial Learn Rate\n")

        with open(self.log_file, 'a') as file:
            file.write(f"{generation},{best.fitness},{best.other['mae']},{best.values['hidden_layers']},{best.values['neurons_per_layer']},{best.values['dropout']},{best.values['inital_learn_rate']}\n")


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

