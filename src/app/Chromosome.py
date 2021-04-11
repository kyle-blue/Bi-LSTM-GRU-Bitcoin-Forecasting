import math
from typing import Any, Dict, List, Tuple
import random

class Limit():
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

class Chromosome():
    def __init__(self, limits:  Dict[str, Limit]):
        self.limits = limits # Gene upper and lower limits
        self.values: Dict[str, float] = {} # Gene Values
        self.fitness = 0.0 # Current Fitness
        self.other: Dict[str, Any] = {}

        self.init_values()

    def init_values(self):
        for key, limit in self.limits.items():
            self.values[key] = random.uniform(limit.lower, limit.upper)

    def mutate(self, mutation_rate: float):
        for key, limit in self.limits.items():
            if random.uniform(0, 1) < mutation_rate:
                self.values[key] = random.uniform(limit.lower, limit.upper)

    def crossover(self, other: "Chromosome", crossover_rate: float, *, type="UNIFORM") -> Tuple["Chromosome", "Chromosome"]:
        # May not crossover depending on crossover rate
        if random.uniform(0, 1) < crossover_rate:
            return self, other
                    
        child1 = Chromosome(self.limits)
        child2 = Chromosome(self.limits)
        
        for key in self.limits:
            parent1_gene = self.values[key]
            parent2_gene = other.values[key]
            
            ## Randomise which gene the children get
            num = random.randrange(0, 2) # Can be 0 or 1
            if num == 0:
                child1.values[key] = parent1_gene
                child2.values[key] = parent2_gene
            else:
                child1.values[key] = parent2_gene
                child2.values[key] = parent1_gene

        return child1, child2

    ## Math operations allow sorting
    def __lt__(self,other):
        return (self.fitness < other.fitness)
    def __le__(self,other):
        return (self.fitness <= other.fitness)
    def __gt__(self,other):
        return (self.fitness > other.fitness)
    def __ge__(self,other):
        return (self.fitness >= other.fitness)
    def __eq__(self,other):
        return (self.fitness == other.fitness)
    def __ne__(self,other):
        return (self.fitness != other.fitness)