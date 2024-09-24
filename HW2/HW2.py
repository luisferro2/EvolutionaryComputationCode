"""
Code for homework #2 Evolutionary Computing Class
Authors:
Luis Ignacio Ferro Salinas A01378248
Iván Miguel García López A01686450
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
# Just for easy read of the encodings and selections.
from enum import Enum
# saving figures
import os

# This is our code from homework number 1.
from classical_optimization import *


class Encoding(Enum):
    BINARY = 'BINARY'
    REAL = 'REAL'

class Selection(Enum):
    ROULETTE = 'ROULETTE'
    TOURNEY = 'TOURNEY'

def debug(thing, title):
    print("--------------------------------------------")
    print(title)
    print(thing)
    print("--------------------------------------------")


# Función problema 1
def test_problem_1(individual):
    x1, x2 = individual
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Función de Rastrigin
def rastrigin(individual):
    A = 10
    n = len(individual)
    return A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in individual])


######################################
# Functions for binary representation

def blen(li, lu):
    # num - the number to convert
    # li - limit inferior
    # lu - limit up
    # Using 4 digits of precision.
    n_digits = int(math.log2((lu - li) * 10 ** 4) + 0.99)
    return n_digits

def real2binary(num, li, lu):
    # num - the number to convert
    # li - limit inferior
    # lu - limit up
    # Using 4 digits of precision.
    n_digits = int(math.log2((lu - li) * 10 ** 4) + 0.99)
    #debug(n_digits, 'digits')
    num_as_int = int((num - li) / 0.0001)
    #debug(num_as_int, 'num in int')
    num_as_bin = bin(num_as_int)[2:].zfill(n_digits)
    return num_as_bin

'''xd = real2binary(-2.0480, -2.0480, 2.0480)
debug(xd, '-2.048 to binary 4 precision')'''

def initialize_randomb(vlen, n, ps, yl, yu):
    ''' Randomly initialize a population of binary individuals.
    
    ps - population size
    ilen - individual length
    '''

    population = []
    for x in range(ps):
        individual = ''
        while len(individual) < vlen * n:
            curr_var = ''
            for j in range(vlen):
                new_digit = random.choices(['0', '1'], weights=[0.5, 0.5], k=1)[0]
                curr_var += new_digit
            if binary2real(curr_var, yl) < yl or binary2real(curr_var, yl) > yu: 
                continue
            individual += curr_var

        population += [individual]
    return population

'''xd = initialize_randomb(10, 10)
debug(xd, 'test population')'''

def binary2real(ind, yl):
    return yl + 0.0001 * int(ind, 2)

'''xd = binary2real('0000000000100000', -2.048)
debug(xd, 'bin to real')'''

def roulette(fitness_list):
    ''' xd
    '''
    fitness_sum = sum(fitness_list)

    
    if fitness_sum == 0:
        proportion_fs = [1 / len(fitness_list)] * len(fitness_list)
    else:
        # Proportion fitnesses
        proportion_fs = [fitness / fitness_sum for fitness in fitness_list]
    #debug(proportion_fs, 'proportions')

    chosen_index = random.choices(list(range(len(fitness_list))), 
                   weights=proportion_fs, k=1)[0]
    #debug(chosen_index, 'chosen index')
    return chosen_index

'''for i in range(5):
    xd = roulette([1, 2, 3, 4, 5])
    print('index', xd)
'''

def spx(parents):
    ''' Single point crossover
    At this point the crossover has been determined
    '''
    P0, P1 = parents

    child1, child2 = P0, P1
    
    ilen = len(P0)
    

    ws = [1 / (ilen - 1)] * (ilen - 1)
    #debug(ws, 'weights')
    position = random.choices(list(range(ilen - 1)), 
                weights=ws, k=1)[0]
    #debug(position, 'position')
    #debug(parents, 'parents spx')
    
    left1 = P0[:position + 1] # exclusive upper limit
    left2 = P1[:position + 1]
    right1 = P0[position + 1:]
    right2 = P1[position + 1:]

    child1 = left1 + right2
    child2 = left2 + right1
    #debug((child1, child2), 'children spx')

    return (child1, child2)

'''xd = spx(['0100', '1011'])
debug(xd, 'children test')'''

def bin_mutation(individual):
    # At this point the mutation has been determined
    ilen = len(individual)
    ws = [1 / (ilen)] * ilen
    #debug(ws, 'weights')
    position = random.choices(list(range(ilen)), 
                    weights=ws, k=1)[0]
    individual = individual[:position] + str(int(not int(individual[position]))) + \
            individual[position + 1:]
    return individual
'''xd = bin_mutation('00000')
debug(xd, 'mutation')'''
###########################################


###########################################
# Functions for real representation

def initialize_randomr(n, ps, yl, yu):
    '''
    n - decision variables
    yl - y lower
    yu - y upper
    '''
    P = []
    for i in range(ps):
        # Current individual.
        curr_ind = []
        for i in range(n):
            #getcontext().prec = 3
            random_float = random.uniform(yl, yu)
            point_index = str(random_float).index('.')
            # Random float truncated
            random_floatt = float(str(random_float)[:point_index + 4])
            #random_decimal = Decimal(random_floatt)
            curr_ind += [random_floatt]
        curr_ind = np.array(curr_ind)
        P += [curr_ind]
    return P

'''xd = initialize_randomr(-2.0480, 2.0480)
print(xd)'''

def sbx(P1, P2, n_c=20):
    u = random.random()
    print('u', u)
    if u <= 0.5:
        beta_bar = (2 * u) ** (1 / n_c + 1)
    else:
        beta_bar = (1 / (2 * (1 - u))) ** (1 / n_c + 1)
    print('beta bar', beta_bar)

    H1 = 0.5 * ((P1 + P2) - beta_bar * (P2 - P1))
    H2 = 0.5 * ((P1 + P2) + beta_bar * (P2 - P1))

    return H1, H2

'''xd = sbx(np.array([1, 2, 3, 4]), np.array([2, 2, 2, 2]))
print(xd)'''

def pm(P, yl, yu, t):
    ''' Parameter mutation

    yl - y lower
    yu - y upper
    '''
    # Var to change index
    var_tci = random.choices(list(range(len(P))),
                             weights=[1 / len(P)] * len(P),
                             k=1)[0]
    debug(P, 'Parent')

    var_to_change = P[var_tci]
    debug(var_to_change, 'var to change')

    u = 0.72#random.random()

    debug(u, 'u')

    # Delta lower
    delta_l = min([(var_to_change - yl) / (yu - yl),
                   (yu - var_to_change) / (yu - yl)])
    debug(delta_l, 'delta l')

    eta_m = 100 + t

    debug(eta_m, 'eta m')

    if u <= 0.5:
        delta_q = (2 * u + (1 - 2 * u) * (1 - delta_l) ** (eta_m + 1))\
              ** (1 / (eta_m + 1)) - 1
    else:
        delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_l)\
                       ** (eta_m + 1)) ** (1 / (eta_m + 1))
    debug(delta_q, 'delta q')

    delta_max = yu - yl

    debug(delta_max, 'delta max')
    new_var = var_to_change + delta_q * delta_max
    P[var_tci] = new_var
    return P

'''xd = pm([2.3, 4.5, -1.2, 0.8], -2, 6, 20)
print(xd)'''

def binary_tourney(f_P, q):
    ''' Fitness Population f_P
    '''
    contestants = random.sample(f_P, q)
    
    '''random.choices([f_P],
                                 weights=[1 / len(f_P)] * len(f_P),
                                 k=q)
    '''
    debug(contestants, 'contestants')

    winner = min(contestants)

    debug(winner, 'winner')

    return f_P.index(winner) 

'''xd = binary_tourney([1, 2, 3], 2)
debug(xd, 'binary winner')'''


###########################################



def genetic_algorithm(func, 
                      yl, 
                      yu, 
                      encoding, 
                      selection,
                      n=0,
                      ps=100,
                      prob_c=0.9,
                      prob_m=0.001,
                      iterations=100,
                      verbose=False,
                      problem='?',
                      fig_n='?'):
    ''' John Holland
    
    ps - population size
    encoding - enum binary or real
    '''
    # Best fitnesses
    bfs = []

    # Initialize Population
    if encoding == Encoding.BINARY:
        vlen = blen(yl, yu)
        P = initialize_randomb(vlen, n, ps, yl, yu)
        #debug(P, 'initial population')
        # Fitness P
        # Genotype to phenotype necessary for binary encoding.
        f_P = []
        for individual in P:
            # Variables phenotypes.
            vars_pts = []
            # Variable start
            for vs in range(0, len(individual), vlen):
                # Current variable phenotype.
                curr_vpt = binary2real(individual[vs: vs + vlen], yl)
                vars_pts += [curr_vpt]
            vars_pts = np.array(vars_pts)
            #debug(vars_pts, 'variables phenotypes')
            curr_f = func(vars_pts)
            #debug(curr_f, 'current fitness')
            f_P += [curr_f]
    else:
        # REAL
        P = initialize_randomr(n, ps, yl, yu)
        #debug(P, 'initial population')
        # Fitness P
        f_P = [func(ind) for ind in P]
    f_max = max(f_P)
    f_Pa = [f_max - f for f in f_P]
    bfs += [min(f_P)]
    #debug(f_P, 'fitnesses original')
    #debug(f_Pa, 'fitnesses adjusted')

    

    # while stopping condition not met
    t = 0
    for i in range(iterations):

        if random.random() <= prob_c:
            #   select parents (proportionate, random, tournament, and uniform-state)
            # Number of times to get 2 children .
            children = []
            for j in range(ps // 2):
                #debug(selection, 'SELECTION HAPPENNING')
                if  selection == Selection.ROULETTE:
                    P1_ind = roulette(f_Pa)
                    P2_ind = roulette(f_Pa)
                    #debug((P1_ind, P2_ind), 'selected parents positions')
                elif selection == Selection.TOURNEY:
                    P1_ind = binary_tourney(f_P, q=2)
                    P2_ind = binary_tourney(f_P, q=2)
                    #debug((P1_ind, P2_ind), 'selected parents positions')#
                    
                P1 = P[P1_ind]
                P2 = P[P2_ind]
                #debug((P1, P2), 'selected parents')
        
                #debug('CROSSOVER HAPPENNING', 'Crossover')
                #   Crossover with crossover probability
                if encoding == Encoding.BINARY:
                    # The binary values may exceed the limit upwards.
                    H1, H2 = spx([P1, P2])
                    for vs in range(0, len(H1), vs):
                        # Fix the upper limit if necessary
                        if binary2real(H1[vs: vs + vlen], yl) > yu: 
                            H1 = H1[:vs] + real2binary(yu, yl, yu) + H1[vs + vlen + 1:]
                        if binary2real(H2[vs: vs + vlen], yl) > yu: 
                            H2 = H2[:vs] + real2binary(yu, yl, yu) + H2[vs + vlen + 1:]

                else:
                    # REAL
                    H1, H2 = sbx(P1, P2)
                children += [H1, H2]
        else:
            # No crossover.
            children = P
    
        #debug(children, 'children')


        #   Mutation with mutation probability
        # child index.
        for cind, child in enumerate(children):
            if random.random() <= prob_m:
                #debug('MUTATION HAPPENNING', 'mutation')
                #debug(child, 'before mutate')
                if encoding == Encoding.BINARY:
                    new_child = bin_mutation(child)
                    for vs in range(0, len(new_child), vs):
                        # Fix the upper limit if necessary
                        if binary2real(new_child[vs: vs + vlen], yl) > yu: 
                            new_child = new_child[:vs] + \
                                real2binary(yu, yl, yu) + \
                                new_child[vs + vlen + 1:]
                    children[cind] = new_child
                else:
                    children[cind] = pm(child, yl, yu, t)
                #debug(children[cind], 'after mutate')
                
        P = children

        if encoding == Encoding.BINARY:
            # Fitness P
            # Genotype to phenotype necessary for binary encoding.
            f_P = []
            all_vars = []
            for individual in P:
                # Variables phenotypes.
                vars_pts = []
                # Variable start
                for vs in range(0, len(individual), vlen):
                    # Current variable phenotype.
                    curr_vpt = binary2real(individual[vs: vs + vlen], yl)
                    vars_pts += [curr_vpt]
                vars_pts = np.array(vars_pts)
                all_vars += [vars_pts]
                #debug(vars_pts, 'variables phenotypes')
                curr_f = func(vars_pts)
                #debug(curr_f, 'current fitness')
                f_P += [curr_f]
        else:
            # REAL
            # Fitness P
            f_P = [func(ind) for ind in P]
        f_max = max(f_P)
        f_Pa = [f_max - f for f in f_P]
        bfs += [min(f_P)]
        debug(all_vars[f_P.index(min(f_P))], 'best variables')
        debug(f_P, 'fitnesses original')
        debug(f_Pa, 'fitnesses adjusted')


        t += 1
    
    if verbose:
        plt.plot(bfs)
        title = f'Genetic_{problem}_encoding_{encoding.value}_selection_{selection.value}_{fig_n}'
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.savefig(os.path.join('figures', f'{encoding.value}', f'{problem}', title))
        plt.clf()

''' 
prob_c = 0.9
prob_m = 1 / n (length of binary individual, or n of variables)
nc = 20
nm = 20
4 digits precision
20 runs 
    rastrigin n=2, n=5
    test problem 1
graph best fitness (y) vs generation (x) verbose mode

compare to hill climb, gradient and Newton on test problem 1 and rastrigin n=2

'''
########################################################
# Homework requirements
limits_p1 = [-2.048, 2.048]
limits_rastrigin = [-5.12, 5.12]

#######################################################
# BINARY
# Length for binary
L1 = int(math.log2((limits_p1[1] - limits_p1[0]) * 10 ** 4) + 0.99)
for i in range(20):
    my_bfs = genetic_algorithm(test_problem_1,
                            limits_p1[0],
                            limits_p1[1],
                            encoding=Encoding.BINARY,
                            selection=Selection.ROULETTE,
                            n=2,
                            ps=100,
                            prob_c=0.9,
                            prob_m=1 / L1,
                            iterations=100,
                            verbose=True,
                            problem='problem1',
                            fig_n=i)
    
# Length for binary
L23 = int(math.log2((limits_rastrigin[1] - limits_rastrigin[0]) * 10 ** 4) + 0.99)
for i in range(20):
    my_bfs = genetic_algorithm(rastrigin,
                            limits_rastrigin[0],
                            limits_rastrigin[1],
                            encoding=Encoding.BINARY,
                            selection=Selection.ROULETTE,
                            n=2,
                            ps=100,
                            prob_c=0.9,
                            prob_m=1 / L23,
                            iterations=100,
                            verbose=True,
                            problem='rastrigin2',
                            fig_n=i)

for i in range(20):
    my_bfs = genetic_algorithm(rastrigin,
                            limits_rastrigin[0],
                            limits_rastrigin[1],
                            encoding=Encoding.BINARY,
                            selection=Selection.ROULETTE,
                            n=3,
                            ps=100,
                            prob_c=0.9,
                            prob_m=1 / L23,
                            iterations=100,
                            verbose=True,
                            problem='rastrigin3',
                            fig_n=i)
#######################################################
# REAL
'''
for i in range(20):
    my_bfs = genetic_algorithm(test_problem_1,
                            limits_p1[0],
                            limits_p1[1],
                            encoding=Encoding.REAL,
                            selection=Selection.TOURNEY,
                            n=2,
                            ps=100,
                            prob_c=0.9,
                            prob_m=1 / 2,
                            iterations=100,
                            verbose=True,
                            problem='problem1',
                            fig_n=i)
    
for i in range(20):
    my_bfs = genetic_algorithm(rastrigin,
                            limits_rastrigin[0],
                            limits_rastrigin[1],
                            encoding=Encoding.REAL,
                            selection=Selection.TOURNEY,
                            n=2,
                            ps=100,
                            prob_c=0.9,
                            prob_m=1 / 2,
                            iterations=100,
                            verbose=True,
                            problem='rastrigin2',
                            fig_n=i)

for i in range(20):
    my_bfs = genetic_algorithm(rastrigin,
                            limits_rastrigin[0],
                            limits_rastrigin[1],
                            encoding=Encoding.REAL,
                            selection=Selection.TOURNEY,
                            n=3,
                            ps=100,
                            prob_c=0.9,
                            prob_m=1 / 3,
                            iterations=100,
                            verbose=True,
                            problem='rastrigin3',
                            fig_n=i)'''

########################################################
########################################################


ps_poses = [50, 100, 250]
#iter_poses = [50, 100, 150]
prob_c_poses = [0.8, 0.9, 0.99]
prob_m_poses = [0.01, 0.1, 0.15]



# Best config average
best_cavg = float('Inf')
'''
# Simple grid search beibi
for curr_ps in ps_poses:
    for curr_prob_c in prob_c_poses:
        for curr_prob_m in prob_m_poses:
            config_avg = 0
            for i in range(3):
                # New population is the children 
                my_bfs = genetic_algorithm(test_problem_1,
                                    limits_p1[0],
                                    limits_p1[1],
                                    encoding=Encoding.BINARY,
                                    selection=Selection.ROULETTE,
                                    n=2,
                                    ps=curr_ps,
                                    prob_c=curr_prob_c,
                                    prob_m=curr_prob_m,
                                    iterations=100,
                                    verbose=False)# of decision variables
                combed_f = sum(my_bfs)
                config_avg += combed_f
            config_avg /= 3

            if  config_avg < best_cavg:
                best_cavg = config_avg
                best_config = [curr_ps,
                                curr_prob_c,
                                curr_prob_m]'''

my_bfs = genetic_algorithm(test_problem_1,
                            limits_p1[0],
                            limits_p1[1],
                            encoding=Encoding.BINARY,
                            selection=Selection.ROULETTE,
                            n=2,
                            ps=250,#best_config[0],
                            prob_c=0.99,#best_config[1],
                            prob_m=0.1,#best_config[2],
                            iterations=100,
                            verbose=False)# of decision variables

#debug(best_config, 'best config')



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