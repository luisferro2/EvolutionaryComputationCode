"""
Code for homework #1 Evolutionary Computing Class
Authors:
Luis Ignacio Ferro Salinas A01378248
Iván Miguel García López A01686450
"""

import numpy as np
from sympy import *

def debug(thing, title):
    print("--------------------------------------------")
    print(title)
    print(thing)
    print("--------------------------------------------")

def evaluate(expression, x, my_actual_symbs):
    """To evaluate expressions of sympy."""
    for i, symb in enumerate(my_actual_symbs):
        expression = expression.subs(symb, x[i])
    return expression


def gradient2(f, x, my_actual_symbs):
    """Computes the 2nd gradient or Hessian matrix."""
    n_vars = x.shape[0]
    grad_expressions = []
    grad2 = np.zeros((n_vars, n_vars))

    grad2_expressions = []
    for i in range(n_vars):
        grad2_expressions += [[None] * n_vars]

    for i in range(n_vars):
        part_df = diff(f, my_actual_symbs[i])
        grad_expressions += [part_df]

    for i, part_df in enumerate(grad_expressions):
        for j, symb in enumerate(my_actual_symbs):
            part_df2 = diff(part_df, symb)
            grad2_expressions[i][j] = part_df2

    for i in range(n_vars):
        for j in range(n_vars):
            part_df2 = grad2_expressions[i][j]
            if isinstance(part_df2, np.float64):
                continue
            for k, symb in enumerate(my_actual_symbs):
                part_df2 = part_df2.subs(symb, x[k])
            grad2[i, j] = part_df2

    return grad2


def gradient(f, x, my_actual_symbs):
    """Computes the first derivative vector or the Jacobian."""
    n_vars = x.shape[0]

    grad = np.zeros((n_vars,))
    for i in range(n_vars):
        part_df = diff(f, my_actual_symbs[i])
        for j, symb in enumerate(my_actual_symbs):
            part_df = part_df.subs(symb, x[j])
        grad[i] = part_df
    return grad


def constrained(x, constraints):
    """True if the decision variables lie within constraints, False otherwise"""
    for i in range(len(x)):
        if x[i] < constraints[i][0] or x[i] > constraints[i][1]:
            return False
    return True


def line_search_wolfe(
    func, my_actual_symbs, x, p, constraints, c1=1e-4, c2=0.9, t_init=1
):
    """
    Step size control using Wolfe conditions.

    :param func: Objective function to minimize.
    :param x: Current solution.
    :param p: Direction of descent.
    :param c1: Constant for the first Wolfe condition.
    :param c2: Constant for the second Wolfe condition.
    :param t_init: Initial step.
    :return: t: Optimal step size.
    """
    t = t_init
    flag = False
    grad_f_x = gradient(func, x, my_actual_symbs)

    l1 = evaluate(func, x + t * p, my_actual_symbs)
    r1 = evaluate(func, x, my_actual_symbs) + c1 * t * np.dot(grad_f_x, p)
    l2 = np.dot(gradient(func, x + t * p, my_actual_symbs), p)
    r2 = c2 * np.dot(grad_f_x, p)
    while not flag:
        # Condition 1
        if l1 > r1 or not constrained(x + t * p, constraints):
            t *= 0.5
        # Condition 2
        elif l2 < r2:
            if not constrained(x + t * 2 * p, constraints):
                break
            t *= 2.0
        else:
            flag = True

        l1 = evaluate(func, x + t * p, my_actual_symbs)
        r1 = evaluate(func, x, my_actual_symbs) + c1 * t * np.dot(grad_f_x, p)
        l2 = np.dot(gradient(func, x + t * p, my_actual_symbs), p)
        r2 = c2 * np.dot(grad_f_x, p)

    return t


def gradient_descent(
    func, my_actual_symbs, x_init, constraints, max_iter=1000, tol=1e-6, t_init=1, verbose=False
):
    """
    Gradient descent method with step size control using Wolfe conditions.

    :param func: Objective function to minimize.
    :param x_init: Initial values x.
    :param max_iter: Maximum number of iterations.
    :param tol: Tolerance for convergence.
    :return: x_optimal: Optimal point found.
    """
    x_list = [x_init]

    x = x_init

    for i in range(max_iter):
        # 0-indexed
        if i >= 19:
            break
        
        if verbose:
            debug(i + 1, 'Iteration')
            debug(x, 'x values')

        grad = gradient(func, x, my_actual_symbs)
        
        if verbose:
            debug(grad, 'Gradient')
            debug(np.linalg.norm(grad), 'Gradient Norm')
        
        if np.linalg.norm(grad) < tol:
            break

        p = -grad
        t = line_search_wolfe(func, my_actual_symbs, x, p, constraints, t_init=t_init)
        if verbose:
            debug(p, 'Direction p')
            debug(t, 'Step size calculated with Wolfe conditions t')
        if not constrained(x + t * p, constraints):
            break
        x = x + t * p
        x_list += [x]

    if verbose:
        x1, x2 = x_list[-1]
        debug((float(format(x1, ".10f")), float(format(x2, ".10f"))), 'THE OPTIMAL POINT FOUND AT:')

    return x_list, i + 1


def newton(
    f,
    my_actual_symbs,
    tol,
    x_init,
    constraints,
    with_wolfe=False,
    t_init=None,
    verbose=False,
):
    """
    Calculates minimum of function f using Newton's method.

    :param f: Objective function to minimize.
    :param tol: Tolerance for convergence.
    :param x_init: Initial value for independent variables.
    :return: x: Values of the solution.
    """

    x_list = [x_init]
    x = x_init
    c = 0

    while tol < np.linalg.norm(gradient(f, x, my_actual_symbs)):
        if c >= 20:
            break
        if verbose:
            debug(c + 1, "Iteration")
            debug(np.linalg.norm(gradient(f, x, my_actual_symbs)), "Gradient norm")
        grad1 = gradient(f, x, my_actual_symbs)
        grad2 = gradient2(f, x, my_actual_symbs)

        if verbose:
            debug(x, "x values")
            debug(grad1, "1st derivative (Jacobian)")
            debug(grad2, "2nd derivative (Hessian)")

        inv_grad2 = -np.linalg.inv(grad2)

        if verbose:
            debug(inv_grad2, "negative inverse of Hessian")

        p = np.dot(inv_grad2, grad1)

        if verbose:
            debug(p, "Direction p")

        if with_wolfe:
            t = line_search_wolfe(f, my_actual_symbs, x, p, constraints, t_init=t_init)
        else:
            t = 1

        if verbose and with_wolfe:
            debug(t, "Step size obtained with Wolfe conditions t")
        elif verbose and not with_wolfe:
            debug(t, "Step size without using Wolfe conditions t")
        if not constrained(x + t * p, constraints):
            break

        x = x + t * p
        x_list += [x]

        c += 1
    if verbose:
        x1, x2 = x_list[-1]
        debug((float(format(x1, ".10f")), float(format(x2, ".10f"))), 'THE OPTIMAL POINT FOUND AT:')

    return x_list, c


def hill_climber(
    func, x_init, my_actual_symbs, constraints, step_size=0.25, max_iter=2500, verbose=False
):
    """
    Continuous Hill Climber method for single objective minimization problems.

    :param func: Objective function to minimize.
    :param x_init: Initial guess.
    :param step_size: Size of the perturbation applied at each step.
    :param max_iter: Maximum number of iterations.
    :return: x_optimal: The optimal point found.
    """
    x_list = [x_init]
    x = x_init
    f_val = evaluate(func, x, my_actual_symbs)

    for i in range(max_iter):
        if verbose:
            debug(i + 1, 'Iteration')
            debug(x, 'x values')

        # Generate a random step
        x_new = x + step_size * (2 * np.random.rand(len(x)) - 1)
        f_val_new = evaluate(func, x_new, my_actual_symbs)

        if not constrained(x_new, constraints):
            break

        # If the new solution is better, update
        if f_val_new < f_val:
            x, f_val = x_new, f_val_new
            x_list += [x]

    if verbose:
        x1, x2 = x_list[-1]
        debug((float(format(x1, ".10f")), float(format(x2, ".10f"))), 'THE OPTIMAL POINT FOUND AT:')
    return x_list, max_iter