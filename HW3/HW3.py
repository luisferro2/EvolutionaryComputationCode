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

# For saving figures path.
import os

# This is our code from homework number 1.
#from classical_optimization import *

class Encoding(Enum):
    BINARY = "BINARY"
    REAL = "REAL"


class Selection(Enum):
    ROULETTE = "ROULETTE"
    TOURNEY = "TOURNEY"
    SURVIVAL = "SURVIVAL"


def debug(thing, title):
    print("--------------------------------------------")
    print(title)
    print(thing)
    print("--------------------------------------------")


def test_problem_1(individual):
    x1, x2 = individual
    return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2


def rastrigin(individual):
    A = 10
    n = len(individual)
    return A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in individual])


######################################
# Functions for binary representation.


def blen(li, lu):
    """Calculates binary length given limits with 4 digits precision.
    li - limit inferior
    lu - limit up
    """

    n_digits = int(math.log2((lu - li) * 10**4) + 0.99)
    return n_digits


def real2binary(num, li, lu):
    """
    num - the number to convert
    li - limit inferior
    lu - limit up
    Using 4 digits of precision.
    """

    n_digits = int(math.log2((lu - li) * 10**4) + 0.99)
    num_as_int = int((num - li) / 0.0001)
    num_as_bin = bin(num_as_int)[2:].zfill(n_digits)
    return num_as_bin


"""xd = real2binary(-2.0480, -2.0480, 2.0480)
debug(xd, '-2.048 to binary 4 precision')"""


def initialize_randomb(vlen, n, ps, yl, yu):
    """Randomly initialize a population of binary individuals.
    vlen - variable length
    n - number of decision variables
    ps - population size
    yl - lower limit
    yu - upper limit
    """

    population = []
    for x in range(ps):
        individual = ""
        while len(individual) < vlen * n:
            curr_var = ""
            for j in range(vlen):
                new_digit = random.choices(["0", "1"], weights=[0.5, 0.5], k=1)[0]
                curr_var += new_digit
            if binary2real(curr_var, yl) < yl or binary2real(curr_var, yl) > yu:
                continue
            individual += curr_var
        population += [individual]
    return population


"""xd = initialize_randomb(10, 10)
debug(xd, 'test population')"""


def binary2real(ind, yl):
    """
    ind - binary string representing individual
    yl - lower limit
    """

    return yl + 0.0001 * int(ind, 2)


"""xd = binary2real('0000000000100000', -2.048)
debug(xd, 'bin to real')"""


def roulette(fitness_list):
    fitness_sum = sum(fitness_list)
    if fitness_sum == 0:
        proportion_fs = [1 / len(fitness_list)] * len(fitness_list)
    else:
        # Proportion fitnesses
        proportion_fs = [fitness / fitness_sum for fitness in fitness_list]
    chosen_index = random.choices(
        list(range(len(fitness_list))), weights=proportion_fs, k=1
    )[0]
    return chosen_index


"""for i in range(5):
    xd = roulette([1, 2, 3, 4, 5])
    print('index', xd)
"""


def spx(parents):
    """Single point crossover
    At this point the crossover has been determined
    """

    P0, P1 = parents
    child1, child2 = P0, P1
    ilen = len(P0)
    ws = [1 / (ilen - 1)] * (ilen - 1)
    position = random.choices(list(range(ilen - 1)), weights=ws, k=1)[0]
    left1 = P0[: position + 1]  # exclusive upper limit
    left2 = P1[: position + 1]
    right1 = P0[position + 1 :]
    right2 = P1[position + 1 :]
    child1 = left1 + right2
    child2 = left2 + right1
    return (child1, child2)


"""xd = spx(['0100', '1011'])
debug(xd, 'children test')"""


def bin_mutation(individual):
    """
    At this point the mutation has been determined
    """

    ilen = len(individual)
    ws = [1 / (ilen)] * ilen
    position = random.choices(list(range(ilen)), weights=ws, k=1)[0]
    individual = (
        individual[:position]
        + str(int(not int(individual[position])))
        + individual[position + 1 :]
    )
    return individual


"""xd = bin_mutation('00000')
debug(xd, 'mutation')"""

###########################################

###########################################
# Functions for real representation.


def initialize_randomr(n, ps, yl, yu):
    """Initialize random population of individuals with real encoding.
    n - number of decision variables
    ps - population size
    yl - lower limit
    yu - upper limit
    """

    P = []
    for i in range(ps):
        # Current individual
        curr_ind = []
        for i in range(n):
            random_float = random.uniform(yl, yu)
            point_index = str(random_float).index(".")
            # Random float truncated
            random_floatt = float(str(random_float)[: point_index + 4])
            curr_ind += [random_floatt]
        curr_ind = np.array(curr_ind)
        P += [curr_ind]
    return P


"""xd = initialize_randomr(-2.0480, 2.0480)
print(xd)"""


def sbx(P1, P2, n_c=20):
    """Simulated binary crossover.
    P1 - Parent 1
    P2 - Parent 2
    n_c - parameter to calculate u
    """

    u = random.random()
    if u <= 0.5:
        beta_bar = (2 * u) ** (1 / n_c + 1)
    else:
        beta_bar = (1 / (2 * (1 - u))) ** (1 / n_c + 1)
    H1 = 0.5 * ((P1 + P2) - beta_bar * np.abs(P2 - P1))
    H2 = 0.5 * ((P1 + P2) + beta_bar * np.abs(P2 - P1))
    return H1, H2


"""xd = sbx(np.array([1, 2, 3, 4]), np.array([2, 2, 2, 2]))
print(xd)"""


def pm(P, yl, yu, t):
    """Parameter mutation.
    yl - lower limit
    yu - upper limit
    """

    # Var to change index
    var_tci = random.choices(list(range(len(P))), weights=[1 / len(P)] * len(P), k=1)[0]
    var_to_change = P[var_tci]
    u = random.random()
    # Delta lower
    delta_l = min([(var_to_change - yl) / (yu - yl), (yu - var_to_change) / (yu - yl)])
    eta_m = 100 + t
    if u <= 0.5:
        delta_q = (2 * u + (1 - 2 * u) * (1 - delta_l) ** (eta_m + 1)) ** (
            1 / (eta_m + 1)
        ) - 1
    else:
        delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_l) ** (eta_m + 1)) ** (
            1 / (eta_m + 1)
        )
    delta_max = yu - yl
    new_var = var_to_change + delta_q * delta_max
    P[var_tci] = new_var
    return P


"""xd = pm([2.3, 4.5, -1.2, 0.8], -2, 6, 20)
print(xd)"""


def binary_tourney(f_P, q):
    """Selection of an index of fitness list by binary tournament.
    f_P - Fitnesses for the population
    q - Number of contestants
    """

    contestants = random.sample(f_P, q)
    winner = min(contestants)
    return f_P.index(winner)


"""xd = binary_tourney([1, 2, 3], 2)
debug(xd, 'binary winner')"""

def diff_mutation(ps, P, r3, F=1):

    all_different = False
    while not all_different:
        r1 = random.randint(0, ps - 1)
        r2 = random.randint(0, ps - 1)
        # In this case r3 is not random per the algorithm description
        #r3 = random.randint(0, ps - 1)
        if r1 != r2:
            all_different = True
    # debug((r1, r2), 'indices')
    # debug(P[r3], 'target')
    # debug(F * (P[r1] - P[r2]), 'change')
    u = P[r3] + F * (P[r1] - P[r2])
    return u

'''xd = diff_mutation(3, [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])], F=1)
debug(xd, 'diff mutation')'''

def diff_bin_crossover(n, pr):
    
    jstar = random.randint(0, n - 1) #0-indexed
    #debug(jstar, 'jstar')
    J = {jstar}
    for j in range(0, n):
        if random.random() < pr and j != jstar:
            J = J.union({j})
    return J

'''xd = diff_bin_crossover(2, 0.7)
debug(xd, 'binomial crossover')'''

###########################################


def genetic_algorithm(
    func,
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
    problem="?",
):
    """Genetic algorithm, originally dneoted reproductive plans by John Holland.
    func - The fitness function
    yl - lower limit
    yu - upper limit
    encoding - type of encoding enumerator
    selection - type of selection enumerator
    n - number of variables
    ps - population size
    prob_c - probability for crossover
    prob_m - probability for mutation
    iterations - the number of iterations
    verbose - To debug the intermediate steps of the algorithm
    """

    if verbose:
        debug(f"{encoding.value} and {selection.value}", "RUNNING GENETIC ALGORITHM")
    # Best fitnesses
    bfs = []
    # Best variables
    bvs = []

    # Initialize Population
    if encoding == Encoding.BINARY:
        vlen = blen(yl, yu)
        P = initialize_randomb(vlen, n, ps, yl, yu)
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
                curr_vpt = binary2real(individual[vs : vs + vlen], yl)
                vars_pts += [np.array(curr_vpt)]
            vars_pts = np.array(vars_pts)
            all_vars += [vars_pts]
            #debug(vars_pts, 'variables phenotypes')
            curr_f = func(vars_pts)
            f_P += [curr_f]
            f_max = max(f_P)
            f_Pa = [f_max - f for f in f_P]
    else:
        # REAL
        P = initialize_randomr(n, ps, yl, yu)
        # Fitness P
        f_P = [func(ind) for ind in P]
        all_vars = P
    if verbose:
        debug(P, "Initial population")

    bfs += [min(f_P)]
    bvs += [all_vars[f_P.index(min(f_P))]]
    
    if verbose:
        debug(f_P, "Fitnesses")

    t = 0
    for i in range(iterations):
        if verbose:
            debug("", f"GENERATION {i}")
        children = []
        # Number of times to get 2 children to equal population size.
        for j in range(ps // 2):
            if verbose:
                debug(selection, "SELECTION HAPPENNING")
            if selection == Selection.ROULETTE:
                P1_ind = roulette(f_Pa)
                P2_ind = roulette(f_Pa)
            elif selection == Selection.TOURNEY:
                P1_ind = binary_tourney(f_P, q=2)
                P2_ind = binary_tourney(f_P, q=2)
            if verbose:
                debug((P1_ind, P2_ind), "Selected parents positions")

            P1 = P[P1_ind]
            P2 = P[P2_ind]
            if verbose:
                debug((P1, P2), "Selected parents")

            if random.random() <= prob_c:
                # Select parents (proportionate, random, tournament, and uniform-state).
                if verbose:
                    debug("", "CROSSOVER HAPPENNING")
                # Crossover with crossover probability.
                if encoding == Encoding.BINARY:
                    # The binary values may exceed the limit upwards.
                    H1, H2 = spx([P1, P2])
                    for vs in range(0, len(H1), vs):
                        # Fix the upper limit if necessary.
                        if binary2real(H1[vs : vs + vlen], yl) > yu:
                            H1 = H1[:vs] + real2binary(yu, yl, yu) + H1[vs + vlen + 1 :]
                        if binary2real(H2[vs : vs + vlen], yl) > yu:
                            H2 = H2[:vs] + real2binary(yu, yl, yu) + H2[vs + vlen + 1 :]
                else:
                    # REAL
                    H1, H2 = sbx(P1, P2)
                children += [H1, H2]
            else:
                # No crossover.
                children += [P1, P2]
            if verbose:
                debug(children, "Children")

        # Mutation with mutation probability.
        # child index.
        for cind, child in enumerate(children):
            if random.random() <= prob_m:
                if verbose:
                    debug("", "MUTATION HAPPENNING")
                    debug(child, "Before mutate")
                if encoding == Encoding.BINARY:
                    new_child = bin_mutation(child)
                    for vs in range(0, len(new_child), vs):
                        # Fix the upper limit if necessary
                        if binary2real(new_child[vs : vs + vlen], yl) > yu:
                            new_child = (
                                new_child[:vs]
                                + real2binary(yu, yl, yu)
                                + new_child[vs + vlen + 1 :]
                            )
                    children[cind] = new_child
                else:
                    children[cind] = pm(child, yl, yu, t)
                if verbose:
                    debug(children[cind], "After mutate")

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
                    curr_vpt = binary2real(individual[vs : vs + vlen], yl)
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
        bvs += [all_vars[f_P.index(min(f_P))]]
        if verbose:
            debug(all_vars[f_P.index(min(f_P))], "best variables")
            debug(f_P, "Fitnesses")

        t += 1

    return bfs, bvs


def diff_genetic_algorithm(
    func,
    yl,
    yu,
    encoding,
    selection,
    n=0,
    ps=100,
    prob_c=0.9,
    #prob_m=0.001,
    iterations=100,
    verbose=False,
    problem="?",
):
    """
    func - The fitness function
    yl - lower limit
    yu - upper limit
    encoding - type of encoding enumerator
    selection - type of selection enumerator
    n - number of variables
    ps - population size
    prob_m - probability for mutation
    iterations - the number of iterations
    verbose - To debug the intermediate steps of the algorithm
    """

    if verbose:
        debug(f"{encoding.value} and {selection.value}", "RUNNING DIFFERENTIAL GENETIC ALGORITHM")
    # Best fitnesses
    bfs = []
    # Best variables
    bvs = []
    
    # REAL
    P = initialize_randomr(n, ps, yl, yu)
    # Fitness P
    f_P = [func(ind) for ind in P]
    if verbose:
        debug(P, "Initial population")

    bfs += [min(f_P)]
    bvs += [P[f_P.index(min(f_P))]]
    if verbose:
        debug(f_P, "Fitnesses")

    t = 0
    for i in range(iterations):
        if verbose:
            debug("", f"GENERATION {i}")
        children = []
        # Number of times to get 2 children to equal population size.

        for j in range(ps):
            # Mutation.
            # child index.
            if verbose:
                debug(P, "population")
                debug("", "MUTATION HAPPENNING")
                debug(j, 'Modifying individual with index')
                
            #children[cind] = pm(child, yl, yu, t)
            ui = diff_mutation(ps, P, j, 1)
            if verbose:
                debug(ui, 'ui')
                debug(ui, 'modified individual ui')

            # Select parents (proportionate, random, tournament, and uniform-state).
            if verbose:
                debug("", "CROSSOVER HAPPENNING")
            # Crossover with crossover probability.
            inds_2_change = diff_bin_crossover(n, prob_c)
            if verbose:
                debug(inds_2_change, 'indices to change')
                debug(P[j], 'member from population being modified')
            child = P[j].copy()
            for ind_2_change in inds_2_change:
                child[ind_2_change] = ui[ind_2_change]
            if verbose:
                debug(child, 'child after crossover')
            if func(child) < func(P[j]):
                children += [child]
            else:
                children += [P[j]]
            if verbose:
                debug(children, 'children')
        if verbose:
            debug(P, 'old population')
            debug(children, 'new population')
        P = children
        # REAL
        # Fitness P
        f_P = [func(ind) for ind in P]
        all_vars = P

        bfs += [min(f_P)]
        bvs += [P[f_P.index(min(f_P))]]
        if verbose:
            debug(all_vars[f_P.index(min(f_P))], "best variables")
            debug(f_P, "Fitnesses")

        t += 1

    return bfs, bvs


def plot_contour(func, constraints, x_list, title):
    """To plot the contour and the evaluated points."""

    delta = 0.025
    x = np.arange(constraints[0][0], constraints[0][1], delta)
    y = np.arange(constraints[1][0], constraints[1][1], delta)
    X, Y = np.meshgrid(x, y)

    Xin = np.array([[xi, yi] for xi, yi in zip(x, y)])
    Z = func([X, Y])
    plt.contourf(X, Y, Z)
    plt.colorbar()
    x1_list = [x[0] for x in x_list[:]]
    x2_list = [x[1] for x in x_list[:]]

    plt.plot(
        x1_list,
        x2_list,
        "r-.",
    )
    (lines,) = plt.plot(x_list[0][0], x_list[0][1], "ro", label="start")
    (linee,) = plt.plot(x_list[-1][0], x_list[-1][1], "r*", label="end")
    plt.legend()
    plt.savefig(f"figures/{title}.jpg")
    plt.clf()



def main(verbose=True, runs_genetic=20):
    """"""

    limits_p1 = [-2.048, 2.048]
    limits_rastrigin = [-5.12, 5.12]

    # Limits problem 1 for contour
    limits_p1_4c = [[-2.048, 2.048], [-2.048, 2.048]]
    limits_rastrigin_4c = [[-5.12, 5.12], [-5.12, 5.12]]
    #######################################################
    # BINARY

    # Length for binary
    L1 = int(math.log2((limits_p1[1] - limits_p1[0]) * 10**4) + 0.99)
    L2 = int(math.log2((limits_rastrigin[1] - limits_rastrigin[0]) * 10**4) + 0.99)

    min_bp1 = float("Inf")
    for i in range(runs_genetic):
        curr_bfs_bp1, curr_bvs_bp1 = genetic_algorithm(
            test_problem_1,
            limits_p1[0],
            limits_p1[1],
            encoding=Encoding.BINARY,
            selection=Selection.ROULETTE,
            n=2,
            ps=100,
            prob_c=0.5,
            prob_m=1 / (L1 * 2),
            iterations=100,
            verbose=False,
            problem="P1",
        )
        # Current fitness sum.
        curr_fs = sum(curr_bfs_bp1)
        if curr_fs < min_bp1:
            min_bp1 = curr_fs
            bfs_bp1 = curr_bfs_bp1
            bvs_bp1 = curr_bvs_bp1

    min_br2 = float("Inf")
    for i in range(runs_genetic):
        curr_bfs_br2, curr_bvs_br2 = genetic_algorithm(
            rastrigin,
            limits_rastrigin[0],
            limits_rastrigin[1],
            encoding=Encoding.BINARY,
            selection=Selection.ROULETTE,
            n=2,
            ps=100,
            prob_c=0.5,
            prob_m=1 / (L2 * 2),
            iterations=100,
            verbose=False,
            problem="rastrigin",
        )
        # Current fitness sum.
        curr_fs = sum(curr_bfs_br2)
        if curr_fs < min_br2:
            min_br2 = curr_fs
            bfs_br2 = curr_bfs_br2
            bvs_br2 = curr_bvs_br2
        
    
    ##################################################
    # REAL TRADITIONAL
    
    min_rtp1 = float("Inf")
    for i in range(runs_genetic):
        # Best fitnesses real traditional problem 1.
        curr_bfs_rtp1, curr_bvs_rtp1 = genetic_algorithm(
            test_problem_1,
            limits_p1[0],
            limits_p1[1],
            encoding=Encoding.REAL,
            selection=Selection.TOURNEY,
            n=2,
            ps=100,
            prob_c=0.5,
            iterations=100,
            verbose=False,
            problem="P1",
        )
        # Current fitness sum.
        curr_fs = sum(curr_bfs_rtp1)
        if curr_fs < min_rtp1:
            min_rtp1 = curr_fs
            bfs_rtp1 = curr_bfs_rtp1
            bvs_rtp1 = curr_bvs_rtp1

    min_rtr2 = float("Inf")
    for i in range(runs_genetic):
        curr_bfs_rtr2, curr_bvs_rtr2 = genetic_algorithm(
            rastrigin,
            limits_rastrigin[0],
            limits_rastrigin[1],
            encoding=Encoding.REAL,
            selection=Selection.TOURNEY,
            n=2,
            ps=100,
            prob_c=0.5,
            iterations=100,
            verbose=False,
            problem="rastrigin",
        )
        # Current fitness sum.
        curr_fs = sum(curr_bfs_rtr2)
        if curr_fs < min_rtr2:
            min_rtr2 = curr_fs
            bfs_rtr2 = curr_bfs_rtr2
            bvs_rtr2 = curr_bvs_rtr2
    

    ##################################################
    # REAL DIFFERENTIAL

    min_rdp1 = float("Inf")
    for i in range(runs_genetic):
        # Best fitnesses real differential problem 1
        curr_bfs_rdp1, curr_bvs_rdp1 = diff_genetic_algorithm(
            test_problem_1,
            limits_p1[0],
            limits_p1[1],
            encoding=Encoding.REAL,
            selection=Selection.SURVIVAL,
            n=2,
            ps=100,
            prob_c=0.5,
            iterations=100,
            verbose=False,
            problem="P1",
        )
        # Current fitness sum.
        curr_fs = sum(curr_bfs_rdp1)
        if curr_fs < min_rdp1:
            min_rdp1 = curr_fs
            bfs_rdp1 = curr_bfs_rdp1
            bvs_rdp1 = curr_bvs_rdp1
    
    min_rdr2 = float("Inf")
    for i in range(runs_genetic):
        curr_bfs_rdr2, curr_bvs_rdr2 = diff_genetic_algorithm(
            rastrigin,
            limits_rastrigin[0],
            limits_rastrigin[1],
            encoding=Encoding.REAL,
            selection=Selection.SURVIVAL,
            n=2,
            ps=100,
            prob_c=0.5,
            iterations=100,
            verbose=False,
            problem="rastrigin",
        )
        # Current fitness sum.
        curr_fs = sum(curr_bfs_rdr2)
        if curr_fs < min_rdr2:
            min_rdr2 = curr_fs
            bfs_rdr2 = curr_bfs_rdr2
            bvs_rdr2 = curr_bvs_rdr2

    for quintuple in ((bfs_bp1, 
                       Encoding.BINARY.value, 
                       Selection.ROULETTE.value, 
                       "Rosenbrock", 
                       bvs_bp1),#encoding, selection, problem), 
                     (bfs_br2, 
                      Encoding.BINARY.value, 
                      Selection.ROULETTE.value, 
                      "Rastrigin", 
                      bvs_br2), 
                     (bfs_rtp1, 
                      Encoding.REAL.value, 
                      Selection.TOURNEY.value, 
                      "Rosenbrock", 
                      bvs_rtp1), 
                     (bfs_rtr2, 
                      Encoding.REAL.value, 
                      Selection.TOURNEY.value, 
                      "Rastrigin", 
                      bvs_rtr2), 
                     (bfs_rdp1, 
                      Encoding.REAL.value + '_Differential', 
                      Selection.SURVIVAL.value, 
                      "Rosenbrock", 
                      bvs_rdp1), 
                     (bfs_rdr2, 
                      Encoding.REAL.value + '_Differential', 
                      Selection.SURVIVAL.value, 
                      "Rastrigin", 
                      bvs_rdr2)):
        curr_bfs, encoding, selection, problem, curr_bvs = quintuple
        curr_lab = f"{encoding}_with_{selection}"
        plt.plot(
                curr_bfs,
                label=curr_lab,
            )
        title = f"Genetic_for_{problem}"
        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        #plt.show()
        plt.savefig(os.path.join("figures", title + "_" + curr_lab))
        plt.clf()
        # Problem to function
        p2f = {"Rosenbrock": test_problem_1, "Rastrigin": rastrigin}
        p2cs = {"Rosenbrock": limits_p1_4c, "Rastrigin": limits_rastrigin_4c}
        plot_contour(p2f[problem], p2cs[problem], curr_bvs, title + "_" + curr_lab)

main(verbose=True)
########################################################
