import numpy as np


class Population:
    def __init__(self, factory, factory_args, pop_size, individuals=None):
        self.pop_size = pop_size

        if individuals is None:
            self.population = np.empty(pop_size, dtype=object)
            self.create_population(factory, factory_args)
        else:
            self.population = np.array(individuals, ndarray=object)

    def create_population(self, factory, factory_args):
        for i in range(self.pop_size):
            self.population[i] = factory.create(*factory_args)


class Individual:
    def __init__(self, factory, factory_args):
        self.fit = -1
        self.rank = -1
        self.indv = factory.create(*factory_args)

    def evaluate(self, *args):
        self.fit = self.indv.run(*args)
