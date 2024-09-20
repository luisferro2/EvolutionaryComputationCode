import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import math
import random
from enum import Enum

class Encoding(Enum):
    BINARY = 1
    REAL = 2

class Selection(Enum):
    ROULETTE = 1
    TOURNEY = 2

def debug(thing, title):
    print("--------------------------------------------")
    print(title)
    print(thing)
    print("--------------------------------------------")


# Función problema 1
def test_problem_1(individual):
    x1, x2 = individual
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2,

# Función de Rastrigin
def rastrigin(individual):
    A = 10
    n = len(individual)
    return A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in individual]),

def blen(li, lu):
    # num - the number to convert
    # li - limit inferior
    # lu - limit up
    # Using 4 digits of precision.
    n_digits = int(math.log2((lu - li) * 10 ** 4) + 0.99)
    return n_digits

def genetic_algorithm(li, 
                      lu, 
                      ps, 
                      encoding, 
                      selection,
                      pc,
                      pm):
    ''' John Holland
    
    ps - population size
    encoding - enum binary or real
    '''


    if encoding == Encoding.BINARY:
        ilen = blen(li, lu)
        P0 = initialize_randomb(ilen, ps)
    else:
        # REAL
        ...

    # while stopping condition not met
    while True:
        #   select parents (proportionate, random, tournament, and uniform-state)
        if  selection == Selection.ROULETTE:
            ...
        elif selection == Selection.TOURNEY:
            ...
        #   Crossover with crossover probability
        #   Mutation with mutation probability    
    
        # New population is the children 



def real2binary(num, li, lu):
    # num - the number to convert
    # li - limit inferior
    # lu - limit up
    # Using 4 digits of precision.
    n_digits = int(math.log2((lu - li) * 10 ** 4) + 0.99)
    debug(n_digits, 'digits')
    num_as_int = int((num - li) / 0.0001)
    debug(num_as_int, 'num in int')
    num_as_bin = bin(num_as_int)[2:].zfill(n_digits)
    return num_as_bin

'''xd = real2binary(-2.0480, -2.0480, 2.0480)
debug(xd, '-2.048 to binary 4 precision')'''

def initialize_randomb(ilen, ps):
    ''' Randomly initialize a population of binary individuals.
    
    ps - population size
    ilen - individual length
    '''
    population = []
    for x in range(ps):
        individual = ''
        for i in range(ilen):
            new_digit = random.choices(['0', '1'], weights=[0.5, 0.5], k=1)[0]
            individual += new_digit
        population += [individual]
    return population

'''xd = initialize_randomb(10, 10)
debug(xd, 'test population')'''

def roulette(fitness_list):
    ''' xd
    '''
    fitness_sum = sum(fitness_list)

    # Proportion fitnesses
    proportion_fs = [fitness / fitness_sum for fitness in fitness_list]
    debug(proportion_fs, 'proportions')

    chosen_index = random.choices(list(range(len(fitness_list))), 
                   weights=proportion_fs, k=1)
    return chosen_index

'''for i in range(5):
    xd = roulette([1, 2, 3, 4, 5])
    print('index', xd)
'''

def spx(parents):
    ''' Single point crossover

    '''
    P0, P1 = parents
    pc = 1#0.9

    child1, child2 = P0, P1

    chance = random.random()
    if chance < pc:    
        
        ilen = len(P0)
        position = random.choices(list(range(ilen - 1)), 
                    weights=[1 / (ilen - 1)] * (ilen - 1), k=1)[0]
        debug(position, 'position')
        left1 = P0[:position + 1] # exclusive upper limit
        left2 = P1[:position + 1]
        right1 = P0[position + 1:]
        right2 = P1[position + 1:]

        child1 = left1 + right2
        child2 = left2 + right1
    return (child1, child2)

'''xd = spx(['0100', '1011'])
debug(xd, 'children test')'''

def bin_mutation(individual):
    ilen = len(individual)
    position = random.choices(list(range(ilen)), 
                    weights=[1 / (ilen)] * (ilen), k=1)[0]
    debug(position, 'position')
    pm = 1 / ilen
    debug(pm, 'pm')
    chance = random.random()
    debug(chance, 'chance')
    if chance < pm:
        individual = individual[:position] + str(int(not int(individual[position]))) + \
            individual[position + 1:]
    return individual
xd = bin_mutation('00000')
debug(xd, 'mutation')

'''

# Setup para algoritmo genético
def setup_ga(toolbox, n, encoding):
    if encoding == 'binary':
        # Codificación binaria
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox.register("attr_bool", np.random.randint, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n)
    else:
        # Codificación real
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox.register("attr_float", np.random.uniform, -5.12, 5.12)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint if encoding == 'binary' else tools.cxSimulatedBinaryBounded, low=-5.12, up=5.12, eta=20.0)
    toolbox.register("mutate", tools.mutFlipBit if encoding == 'binary' else tools.mutPolynomialBounded, low=-5.12, up=5.12, eta=20.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", test_problem_1 if n == 2 else rastrigin)

    return toolbox

# Ejecutar algoritmo
def run_ga(toolbox, ngen=100, pop_size=50):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof

# Graficar convergencia
def plot_convergence(log, title):
    gen = log.select("gen")
    avg = log.select("avg")
    min_ = log.select("min")
    
    plt.plot(gen, avg, label='Promedio')
    plt.plot(gen, min_, label='Mínimo')
    plt.xlabel("Generaciones")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.show()

# Ejecutar para un problema de prueba
toolbox = base.Toolbox()
toolbox = setup_ga(toolbox, n=2, encoding='real')  # Cambiar 'real' a 'binary' según sea necesario
pop, log, hof = run_ga(toolbox)

plot_convergence(log, "Convergencia para el problema de prueba")

# Setup para algoritmo genético
def setup_ga(toolbox, n, encoding):
    if encoding == 'binary':
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox.register("attr_bool", np.random.randint, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n)
    else:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox.register("attr_float", np.random.uniform, -5.12, 5.12)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint if encoding == 'binary' else tools.cxSimulatedBinaryBounded, low=-5.12, up=5.12, eta=20.0)
    toolbox.register("mutate", tools.mutFlipBit if encoding == 'binary' else tools.mutPolynomialBounded, low=-5.12, up=5.12, eta=20.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", test_problem_1 if n == 2 else rastrigin)

    return toolbox

# Ejecutar algoritmo genético para una corrida
def run_ga(toolbox, ngen=100, pop_size=50):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=False)
    return pop, log, hof

# Realizar 20 corridas para cada problema y codificación
def run_20_experiments(toolbox, n_runs=20, ngen=100, pop_size=50):
    results = []
    for i in range(n_runs):
        pop, log, hof = run_ga(toolbox, ngen, pop_size)
        # Guardar los mejores valores de fitness de cada corrida
        best_fitness = hof[0].fitness.values[0]
        results.append(best_fitness)
    return results

# Graficar la convergencia (utilizando los logs de una de las corridas)
def plot_convergence(log, title):
    gen = log.select("gen")
    avg = log.select("avg")
    min_ = log.select("min")
    
    plt.plot(gen, avg, label='Promedio')
    plt.plot(gen, min_, label='Mínimo')
    plt.xlabel("Generaciones")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.show()

# Ejecutar el experimento completo para un problema y codificación específica
def experiment(toolbox, encoding, problem_name):
    print(f"Ejecutando 20 corridas para {problem_name} con codificación {encoding}")
    results = run_20_experiments(toolbox)
    mean_fitness = np.mean(results)
    std_fitness = np.std(results)
    min_fitness = np.min(results)
    max_fitness = np.max(results)
    
    print(f"Media: {mean_fitness}")
    print(f"Desviación estándar: {std_fitness}")
    print(f"Mínimo: {min_fitness}")
    print(f"Máximo: {max_fitness}")
    
    # Seleccionar la mejor corrida para graficar
    _, log, _ = run_ga(toolbox)
    plot_convergence(log, f"Convergencia - {problem_name} ({encoding})")

    return results

# Definir los problemas a ejecutar
problems = [
    ("Problema de prueba 1", 2),
    ("Rastrigin n=2", 2),
    ("Rastrigin n=5", 5)
]

# Configurar los experimentos y ejecutar para ambas codificaciones
for problem_name, n in problems:
    for encoding in ['binary', 'real']:
        toolbox = base.Toolbox()
        toolbox = setup_ga(toolbox, n, encoding)
        experiment(toolbox, encoding, problem_name)'''