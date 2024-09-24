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
    #print('u', u)
    if u <= 0.5:
        beta_bar = (2 * u) ** (1 / n_c + 1)
    else:
        beta_bar = (1 / (2 * (1 - u))) ** (1 / n_c + 1)
    #print('beta bar', beta_bar)

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
    #debug(P, 'Parent')

    var_to_change = P[var_tci]
    #debug(var_to_change, 'var to change')

    u = 0.72#random.random()

    #debug(u, 'u')

    # Delta lower
    delta_l = min([(var_to_change - yl) / (yu - yl),
                   (yu - var_to_change) / (yu - yl)])
    #debug(delta_l, 'delta l')

    eta_m = 100 + t

    #debug(eta_m, 'eta m')

    if u <= 0.5:
        delta_q = (2 * u + (1 - 2 * u) * (1 - delta_l) ** (eta_m + 1))\
              ** (1 / (eta_m + 1)) - 1
    else:
        delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_l)\
                       ** (eta_m + 1)) ** (1 / (eta_m + 1))
    #debug(delta_q, 'delta q')

    delta_max = yu - yl

    #debug(delta_max, 'delta max')
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
    #debug(contestants, 'contestants')

    winner = min(contestants)

    #debug(winner, 'winner')

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
                      problem='?',):
    ''' John Holland
    
    ps - population size
    encoding - enum binary or real
    '''

    if verbose:
        debug(f'{encoding.value} and {selection.value}', 'RUNNING GENETIC ALGORITHM')
    # Best fitnesses
    bfs = []

    # Initialize Population
    if encoding == Encoding.BINARY:
        vlen = blen(yl, yu)
        P = initialize_randomb(vlen, n, ps, yl, yu)
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

            f_max = max(f_P)
            f_Pa = [f_max - f for f in f_P]
    else:
        # REAL
        P = initialize_randomr(n, ps, yl, yu)
        # Fitness P
        f_P = [func(ind) for ind in P]
    if verbose:
        debug(P, 'Initial population')
        
    bfs += [min(f_P)]
    if verbose:
        debug(f_P, 'Fitnesses')

    # while stopping condition not met
    t = 0
    for i in range(iterations):
        if verbose:
            debug('', f'GENERATION {i}')
        children = []
        # Number of times to get 2 children to equal population size.
        for j in range(ps // 2):
            if verbose:
                debug(selection, 'SELECTION HAPPENNING')
            if  selection == Selection.ROULETTE:
                P1_ind = roulette(f_Pa)
                P2_ind = roulette(f_Pa)
            elif selection == Selection.TOURNEY:
                P1_ind = binary_tourney(f_P, q=2)
                P2_ind = binary_tourney(f_P, q=2)
            if verbose:
                debug((P1_ind, P2_ind), 'Selected parents positions')
                
            P1 = P[P1_ind]
            P2 = P[P2_ind]
            if verbose:
                debug((P1, P2), 'Selected parents')

            if random.random() <= prob_c:
                # Select parents (proportionate, random, tournament, and uniform-state)
                if verbose:
                    debug('', 'CROSSOVER HAPPENNING')
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
                children += [P1, P2]
            if verbose:
                debug(children, 'Children')
        
        # Mutation with mutation probability.
        # child index.
        for cind, child in enumerate(children):
            if random.random() <= prob_m:
                if verbose:
                    debug('', 'MUTATION HAPPENNING')
                    debug(child, 'Before mutate')
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
                if verbose:
                    debug(children[cind], 'After mutate')
                
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
                curr_f = func(vars_pts)
                f_P += [curr_f]
                f_max = max(f_P)
                f_Pa = [f_max - f for f in f_P]
        else:
            # REAL
            # Fitness P
            f_P = [func(ind) for ind in P]
            all_vars = P
        
        bfs += [min(f_P)]
        if verbose:
            debug(all_vars[f_P.index(min(f_P))], 'best variables')
            debug(f_P, 'Fitnesses')

        t += 1
    
    return bfs
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


def main(verbose=True, runs_genetic=20):
    ########################################################
    # Homework requirements.
    limits_p1 = [-2.048, 2.048]
    limits_rastrigin = [-5.12, 5.12]
    if verbose:
        debug('\n', 'FIRST SECTION TOY EXAMPLES FOR THE THREE PROBLEMS')
    #######################################################
    # BINARY
    
    # Problem 1 binary for table.
    p1b_table = []
    r2b_table = []
    r3b_table = []

    # Minimum fitness sum for problem 1 binary.
    min_p1b = float('Inf')
    best_p1b = -1
    # Best for problem 1 binary.
    min_r2b = float('Inf')
    best_r2b = -1
    min_r3b = float('Inf')
    best_r3b = -1


    # Length for binary
    L1 = int(math.log2((limits_p1[1] - limits_p1[0]) * 10 ** 4) + 0.99)

    # Toy example
    if verbose:
        debug('\n', 'PROBLEM 1')
    curr_bfs = genetic_algorithm(test_problem_1,
                                limits_p1[0],
                                limits_p1[1],
                                encoding=Encoding.BINARY,
                                selection=Selection.ROULETTE,
                                n=2,
                                ps=4,
                                prob_c=0.9,
                                prob_m=1 / L1,
                                iterations=3,
                                verbose=True,
                                problem='problem1')

    for i in range(runs_genetic):
        curr_bfs = genetic_algorithm(test_problem_1,
                                limits_p1[0],
                                limits_p1[1],
                                encoding=Encoding.BINARY,
                                selection=Selection.ROULETTE,
                                n=2,
                                ps=100,
                                prob_c=0.9,
                                prob_m=1 / L1,
                                iterations=100,
                                verbose=False,
                                problem='problem1')
        p1b_table += [min(curr_bfs)]

        # Current fitness sum.
        curr_fs = sum(curr_bfs)
        if curr_fs < min_p1b:
            min_p1b = curr_fs
            best_p1b = curr_bfs

    # Length for binary
    L23 = int(math.log2((limits_rastrigin[1] - limits_rastrigin[0]) * 10 ** 4) + 0.99)
    for i in range(runs_genetic):
        curr_bfs = genetic_algorithm(rastrigin,
                                limits_rastrigin[0],
                                limits_rastrigin[1],
                                encoding=Encoding.BINARY,
                                selection=Selection.ROULETTE,
                                n=2,
                                ps=100,
                                prob_c=0.9,
                                prob_m=1 / (L23 * 2),
                                iterations=100,
                                verbose=False,
                                problem='rastrigin2',)
        r2b_table += [min(curr_bfs)]

        # Current fitness sum.
        curr_fs = sum(curr_bfs)
        if curr_fs < min_r2b:
            min_r2b = curr_fs
            best_r2b = curr_bfs

    for i in range(runs_genetic):
        curr_bfs = genetic_algorithm(rastrigin,
                                limits_rastrigin[0],
                                limits_rastrigin[1],
                                encoding=Encoding.BINARY,
                                selection=Selection.ROULETTE,
                                n=5,
                                ps=100,
                                prob_c=0.9,
                                prob_m=1 / (L23 * 5),
                                iterations=100,
                                verbose=False,
                                problem='rastrigin3',)
        r3b_table += [min(curr_bfs)]

        # Current fitness sum.
        curr_fs = sum(curr_bfs)
        if curr_fs < min_r3b:
            min_r3b = curr_fs
            best_r3b = curr_bfs
                
    #######################################################
    # REAL

    # Problem 1 real for table.
    p1r_table = []
    r2r_table = []
    r3r_table = []

    # Minimum fitness sum for problem 1 real.
    min_p1r = float('Inf')
    best_p1r = -1
    # Best for problem 1 real.
    min_r2r = float('Inf')
    best_r2r = -1
    min_r3r = float('Inf')
    best_r2r = -1

    for i in range(runs_genetic):
        curr_bfs = genetic_algorithm(test_problem_1,
                                limits_p1[0],
                                limits_p1[1],
                                encoding=Encoding.REAL,
                                selection=Selection.TOURNEY,
                                n=2,
                                ps=100,
                                prob_c=0.9,
                                prob_m=1 / 2,
                                iterations=100,
                                verbose=False,
                                problem='problem1',)
        p1r_table += [min(curr_bfs)]

        # Current fitness sum.
        curr_fs = sum(curr_bfs)
        if curr_fs < min_p1r:
            min_p1r = curr_fs
            best_p1r = curr_bfs


    # Toy example
    if verbose:
        debug('\n', 'RASTRIGIN N=2')
    curr_bfs = genetic_algorithm(rastrigin,
                                limits_rastrigin[0],
                                limits_rastrigin[1],
                                encoding=Encoding.REAL,
                                selection=Selection.TOURNEY,
                                n=2,
                                ps=4,
                                prob_c=0.9,
                                prob_m=1 / 2,
                                iterations=3,
                                verbose=True,
                                problem='rastrigin2',)

    for i in range(runs_genetic):
        curr_bfs = genetic_algorithm(rastrigin,
                                limits_rastrigin[0],
                                limits_rastrigin[1],
                                encoding=Encoding.REAL,
                                selection=Selection.TOURNEY,
                                n=2,
                                ps=100,
                                prob_c=0.9,
                                prob_m=1 / 2,
                                iterations=100,
                                verbose=False,
                                problem='rastrigin2',)
        r2r_table += [min(curr_bfs)]

        # Current fitness sum.
        curr_fs = sum(curr_bfs)
        if curr_fs < min_r2r:
            min_r2r = curr_fs
            best_r2r = curr_bfs

    # Toy example.
    if verbose:
        debug('\n', 'RASTRIGIN N=5')
    curr_bfs = genetic_algorithm(rastrigin,
                                limits_rastrigin[0],
                                limits_rastrigin[1],
                                encoding=Encoding.REAL,
                                selection=Selection.TOURNEY,
                                n=5,
                                ps=4,
                                prob_c=0.9,
                                prob_m=1 / 5,
                                iterations=3,
                                verbose=True,
                                problem='rastrigin5',)

    for i in range(runs_genetic):
        curr_bfs = genetic_algorithm(rastrigin,
                                limits_rastrigin[0],
                                limits_rastrigin[1],
                                encoding=Encoding.REAL,
                                selection=Selection.TOURNEY,
                                n=5,
                                ps=100,
                                prob_c=0.9,
                                prob_m=1 / 5,
                                iterations=100,
                                verbose=False,
                                problem='rastrigin5',)
        r3r_table += [min(curr_bfs)]

        # Current fitness sum.
        curr_fs = sum(curr_bfs)
        if curr_fs < min_r3r:
            min_r3r = curr_fs
            best_r3r = curr_bfs

    if verbose:
        debug('\n', 'SECOND SECTION OVERLEAF TABLE')

    if verbose:
        plt.style.use('ggplot')
        # Best pair for problem.
        for best_pair_p in [(best_p1b, best_p1r, 'problem1'),
                          (best_r2b, best_r2r, 'rastrigin2'),
                          (best_r3b, best_r3r, 'rastrigin5')]:
            
            best_bin, best_real, problem = best_pair_p
            plt.plot(best_bin, label=f'{Encoding.BINARY.value} with {Selection.ROULETTE.value}')
            plt.plot(best_real, label=f'{Encoding.REAL.value} with {Selection.TOURNEY.value}')
            title = f'Genetic_for_{problem}'
            plt.title(title)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.savefig(os.path.join('figures', problem, title))
            plt.clf()

        debug('BINARY PROBLEM 1 & REAL PROBLEM 1 & BINARY RASTRIGIN 2 & REAL \
                RASTRIGIN 2 & BINARY RASTRIGIN 3 & REAL RASTRIGIN 3',
                'THE VALUES TO FILL TABLE IN OVERLEAF')
        for ind, cp1b, cp1r, cr2b, cr2r, cr3b, cr3r in zip(list(range(20)), p1b_table, 
                            p1r_table,
                            r2b_table,
                            r2r_table,
                            r3b_table,
                            r3r_table):
            debug(f'{ind} & {cp1b} & {cp1r} & {cr2b} & {cr2r} & {cr3b} & {cr3r}\\\\', '')

        debug(f'Mean & {np.mean(p1b_table)} & {np.mean(p1r_table)} & {np.mean(r2b_table)} \
                {np.mean(r2r_table)} & {np.mean(r3b_table)} & {np.mean(r3r_table)}\\\\', 
                'MEAN')
        debug(f'Standard Deviation & {np.std(p1b_table)} & {np.std(p1r_table)} & {np.std(r2b_table)} \
                {np.std(r2r_table)} & {np.std(r3b_table)} & {np.std(r3r_table)}\\\\', 
                'STD')
        debug(f'Min & {np.min(p1b_table)} & {np.min(p1r_table)} & {np.min(r2b_table)} \
                {np.min(r2r_table)} & {np.min(r3b_table)} & {np.min(r3r_table)}\\\\', 
                'MIN')
        debug(f'Max & {np.max(p1b_table)} & {np.max(p1r_table)} & {np.max(r2b_table)} \
                {np.max(r2r_table)} & {np.max(r3b_table)} & {np.max(r3r_table)}\\\\', 
                'MAX')
    

    # For sympy.
    x1, x2 = symbols("x1 x2")
    # problem 1 min 0 at (1, 1)
    expression1 = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    constraints1 = [(-2.048, 2.048), (-2.048, 2.048)]
    x_init1 = np.array([0, 1.5])
    # rastrigin 2 min 0 at (0, 0)
    expression2 = 10 * 2 + (x1**2 - 10 * cos(2 * pi * x1)) + (x2**2 - 10 * cos(2 * pi * x2))
    constraints2 = [(-5.12, 5.12), (-5.12, 5.12)]
    x_init2 = np.array([-2, 2])

    # Problem 1 comparison
    x_list11, iters1 = gradient_descent(
        expression1, [x1, x2], x_init1, constraints1, t_init=10
    )

    x_list12, iters12 = newton(
        expression1, [x1, x2], 1e-6, x_init1, constraints1, with_wolfe=False, verbose=False
    )

    x_list13, iters13 = hill_climber(expression1, x_init1, [x1, x2], constraints1)

    # Best problem 1 gradient Newton and Hill.
    best_p1g = [test_problem_1(x) for x in x_list11]
    best_p1n = [test_problem_1(x) for x in x_list12]
    best_p1h = [test_problem_1(x) for x in x_list13]

    # Rastrigin 2 comparison
    x_list21, iters1 = gradient_descent(
        expression2, [x1, x2], x_init2, constraints2, t_init=1
    )
    

    x_list22, iters12 = newton(
        expression2, [x1, x2], 1e-6, x_init2, constraints2, with_wolfe=True, t_init=250, verbose=False
    )

    x_list23, iters13 = hill_climber(expression2, x_init2, [x1, x2], constraints2, step_size=2)

    

    # Best rastrigin 2 gradient Newton and Hill.
    best_r2g = [rastrigin(x) for x in x_list21]
    best_r2n = [rastrigin(x) for x in x_list22]
    best_r2h = [rastrigin(x) for x in x_list23]


    if verbose:
        debug('\n', 'THIRD SECTION SAVING FIGURES COMPARING GENETIC WITH OPTIMIZATION IN FOLDER "/figures"')
    for best_quintuple_p in [(best_p1b, best_p1r, best_p1g, best_p1n, best_p1h, 'problem1'),
                          (best_r2b, best_r2r, best_r2g, best_r2n, best_r2h, 'rastrigin2'),]:
            
            best_bin, best_real, best_grad, best_newton, best_hill, problem = best_quintuple_p
            title = f'Genetic_vs_Optimization_{problem}'
            plt.title(title)
            plt.bar(['Binary',
                     'Real',
                     'Gradient Descent',
                     'Newton', 
                     'Hill Climb'], 
                     [best_bin[-1],
                      best_real[-1],
                      best_grad[-1],
                      best_newton[-1],
                      best_hill[-1]],
                      color=['yellow',
                              'blue', 
                              'purple', 
                              'red', 
                              'green'])
            
            plt.savefig(os.path.join('figures', problem, title))

            plt.ylabel('The absolute error')

            


main(verbose=True, runs_genetic=20)
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