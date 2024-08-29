"""
Code for homework #1 Evolutionary Computing Class
Authors:
Luis Ignacio Ferro Salinas
Iván
"""

import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from scipy import optimize

# Global minimums obtained via Wolfram Alpha.
min1 = [np.array([0, 0])]
min2 = [np.array([-0.08984201310031806242249056062, 0.7126564030207396333972658142]),
        np.array([0.08984201310031806242249056062, -0.7126564030207396333972658142])]
min3 = [np.array([0, 0])]


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
        # debug(type(part_df), 'type expression')
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
    Control del tamaño de paso usando las condiciones de Wolfe.

    :param func: Función objetivo a minimizar.
    :param x: Solución actual.
    :param p: Dirección de descenso.
    :param c1: Constante para la primera condición de Wolfe.
    :param c2: Constante para la segunda condición de Wolfe.
    :param t_init: Paso inicial.
    :return: t: Tamaño de paso óptimo.
    """
    t = t_init
    flag = False
    grad_f_x = gradient(func, x, my_actual_symbs)

    l1 = evaluate(func, x + t * p, my_actual_symbs)
    # print(l1, 'l1')
    r1 = evaluate(func, x, my_actual_symbs) + c1 * t * np.dot(grad_f_x, p)
    # print(r1, 'r1')
    l2 = np.dot(gradient(func, x + t * p, my_actual_symbs), p)
    # print(l2, 'l2')
    r2 = c2 * np.dot(grad_f_x, p)
    # print(r2, 'r2')
    while not flag:
        # debug(t, 't')
        # debug(x + t * p, 'x + t * p')
        if l1 > r1 or not constrained(x + t * p, constraints):
            # debug(x + t * p, 'x plus step direction')

            # if t < 1e-2:
            #    break
            t *= 0.5
        elif l2 < r2:
            # debug(, 'larger')
            if not constrained(x + t * 2 * p, constraints):
                break
            t *= 2.0
        else:
            flag = True

        l1 = evaluate(func, x + t * p, my_actual_symbs)
        # print('------------------------------------------------')
        # print(l1, 'l1')
        r1 = evaluate(func, x, my_actual_symbs) + c1 * t * np.dot(grad_f_x, p)
        # print(r1, 'r1')
        l2 = np.dot(gradient(func, x + t * p, my_actual_symbs), p)
        # print(l2, 'l2')
        r2 = c2 * np.dot(grad_f_x, p)
        # print(r2, 'r2')

    return t


def gradient_descent(
    func, my_actual_symbs, x_init, constraints, max_iter=1000, tol=1e-6, t_init=1
):
    """
    Método de descenso de gradiente con control del tamaño de paso usando condiciones
    de Wolfe.

    :param func: Función objetivo a minimizar.
    :param x_init: Suposición inicial.
    :param max_iter: Número máximo de iteraciones.
    :param tol: Tolerancia para la convergencia.
    :return: x_optimal: El punto óptimo encontrado.
    """
    x_list = [x_init]

    x = x_init
    # debug(x, 'x')

    for i in range(max_iter):
        grad = gradient(func, x, my_actual_symbs)

        # print(np.linalg.norm(grad), 'grad norm')
        if np.linalg.norm(grad) < tol:
            break

        p = -grad
        t = line_search_wolfe(func, my_actual_symbs, x, p, constraints, t_init=t_init)
        # debug(p, 'direction p')
        # debug(t, 't')
        # debug(x + t * p, 'new x')
        if not constrained(x + t * p, constraints):
            break
        x = x + t * p
        x_list += [x]
        # debug(x, 'x')
    return x_list, i + 1


def newton(f, my_actual_symbs, tol, x_init, constraints, with_wolfe=False, t_init=None):
    """
    Calcula el mínimo de la función f usando el
    método de Newton.

    :param f: Función objetivo a minimizar.
    :param tol: Tolerancia para la convergencia.
    :param x_init: Valor inicial de las variables independientes.
    :return: x: Los parámetros de la solución.
    """

    x_list = [x_init]
    x = x_init
    c = 0

    # debug(tol >= np.linalg.norm(gradient(f, x, my_actual_symbs)), 'tol >= gradient norm')
    # debug(tol, 'tol')

    while tol < np.linalg.norm(gradient(f, x, my_actual_symbs)):
        # debug(c, 'iteration')
        # debug(np.linalg.norm(gradient(f, x, my_actual_symbs)), 'norm grad')
        grad1 = gradient(f, x, my_actual_symbs)
        grad2 = gradient2(f, x, my_actual_symbs)
        # debug(x, 'x values')

        # debug(grad1, 'gradient')
        # debug(grad2, 'gradient 2')
        inv_grad2 = -np.linalg.inv(grad2)
        # debug(inv_grad2, 'inverse grad 2')
        p = np.dot(inv_grad2, grad1)
        # debug(p, 'direction p')
        if with_wolfe:
            t = line_search_wolfe(f, my_actual_symbs, x, p, constraints, t_init=t_init)
        else:
            t = 1
        # debug(t, 't')

        if not constrained(x + t * p, constraints):
            break
        x = x + t * p
        x_list += [x]

        # debug(np.linalg.norm(grad1), 'gradient norm')
        c += 1

    return x_list, c


def hill_climber(
    func, x_init, my_actual_symbs, constraints, step_size=1, max_iter=2500
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

    for _ in range(max_iter):
        # Generate a random step
        x_new = x + step_size * (2 * np.random.rand(len(x)) - 1)
        f_val_new = evaluate(func, x_new, my_actual_symbs)

        # If the new solution is better, update
        if f_val_new < f_val:
            x, f_val = x_new, f_val_new
            x_list += [x]

    return x_list, max_iter


# Main process

# Initial values
x_init1 = np.array([-4.0, 4.0])
x_init2 = np.array([0.5, 1.0])
x_init3 = np.array([-2.0, 2.0])

# Constraints for decision variables.
constraints1 = [(-6, 6), (-6, 6)]
constraints2 = [(-3, 3), (-2, 2)]
constraints3 = [(-5.12, 5.12), (-5.12, 5.12)]

# For sympy.
x1, x2 = symbols("x1 x2")

# Expression 1 is multiplied by -1 to convert to minimization problem
expression1 = -(-2 * x1**2 + 3 * x1 * x2 - 1.5 * x2**2 - 1.3)
expression2 = (
    (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
)
expression3 = 10 * 2 + (x1**2 - 10 * cos(2 * pi * x1)) + (x2**2 - 10 * cos(2 * pi * x2))


# Also as functions for easier evaluation.
def func1(x):
    x1, x2 = x
    return -(-2 * x1**2 + 3 * x1 * x2 - 1.5 * x2**2 - 1.3)


def func2(x):
    x1, x2 = x
    return (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


def func3(x):
    x1, x2 = x
    return (
        10 * 2
        + (x1**2 - 10 * np.cos(2 * np.pi * x1))
        + (x2**2 - 10 * np.cos(2 * np.pi * x2))
    )


def plot_contour(func, constraints, x_list, title):
    """To plot the contour and the evaluated points."""

    delta = 0.025
    x = np.arange(constraints[0][0], constraints[0][1], delta)
    y = np.arange(constraints[1][0], constraints[1][1], delta)
    X, Y = np.meshgrid(x, y)

    Xin = np.array([[xi, yi] for xi, yi in zip(x, y)])
    Z = func([X, Y])
    # debug(Z.shape, 'Z shape')
    # fig, ax = plt.subplots()
    plt.contourf(X, Y, Z)
    plt.colorbar()
    # cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    # ax.clabel(CS, inline=False, fontsize=10)
    # plt.set_title('Simplest default with labels')
    x1_list = [x[0] for x in x_list[:]]
    x2_list = [x[1] for x in x_list[:]]

    plt.plot(
        x1_list,
        x2_list,
        "r-.",
    )
    (lines,) = plt.plot(x_list[0][0], x_list[0][1], "ro", label="start")
    (linee,) = plt.plot(x_list[-1][0], x_list[-1][1], "r*", label="end")
    # lines.set_color('black')
    # linee.set_color('black')
    plt.legend()
    plt.savefig(f"figures/{title}.jpg")
    plt.clf()


def print_4_table(func, final_points, algo, mins_x, iters):
    """For easier integration with overleaf report."""

    titles = "Algorithm & Point Found & Evaluation & Optimal point & Real Minimum & \
2-norm error between points & iterations"
    
    errors = [float(format(np.linalg.norm(min - final_points), '.10f')) \
              for min in mins_x]
    real_mins = [func(min_x) for min_x in mins_x]
    real_mins_fmt = [float(format(rmin, '.10f')) for rmin in real_mins]

    mins_x_fmt = [(float(format(x1, '.10f')), 
                   float(format(x2, '.10f'))) for x1, x2 in mins_x]

    curr_str = (
        algo
        + " & "
        + str(final_points)
        + " & "
        + format(func1(final_points), '.10f')
        + " & "
        + str(mins_x_fmt)
        + " & "
        + str(real_mins_fmt)
        + " & "
        + str(errors)
        + " & "
        + str(iters)
    )

    debug(curr_str, titles)


# Gradient descent
# 0.8 for all 3 to work
# t = 10 finds faster
x_list1, iters1 = gradient_descent(
    expression1, [x1, x2], x_init1, constraints1, t_init=10
)
# t = 8.1 finds faster
x_list2, iters2 = gradient_descent(
    expression2, [x1, x2], x_init2, constraints2, t_init=8.1
)
# t = 1 finds minimum
x_list3, iters3 = gradient_descent(
    expression3, [x1, x2], x_init3, constraints3, t_init=1
)
plot_contour(func1, constraints1, x_list1, "function_1_gradient_descent")
print_4_table(func1, x_list1[-1], "gradient descent", min1, iters1)
plot_contour(func2, constraints2, x_list2, "function_2_gradient_descent")
print_4_table(func2, x_list2[-1], "gradient descent", min2, iters2)
plot_contour(func3, constraints3, x_list3, "function_3_gradient_descent")
print_4_table(func3, x_list3[-1], "gradient descent", min3, iters3)

# Newton without wolfe
x_list12, iters12 = newton(expression1, [x1, x2], 1e-6, x_init1, constraints1, with_wolfe=False)
x_list22, iters22 = newton(expression2, [x1, x2], 1e-6, x_init2, constraints2, with_wolfe=False)
x_list32, iters32 = newton(expression3, [x1, x2], 1e-6, x_init3, constraints3, with_wolfe=False)
plot_contour(func1, constraints1, x_list12, "function_1_newton_no_wolfe")
print_4_table(func1, x_list12[-1], "newton", min1, iters12)
plot_contour(func2, constraints2, x_list22, "function_2_newton_no_wolfe")
print_4_table(func2, x_list22[-1], "newton", min2, iters22)
plot_contour(func3, constraints3, x_list32, "function_3_newton_no_wolfe")
print_4_table(func3, x_list32[-1], "newton", min3, iters32)

# Newton Method with wolfe
# 1 for all to work
# t = 1
x_list12, iters12 = newton(expression1, [x1, x2], 1e-6, x_init1, constraints1, with_wolfe=True, t_init=1)
# t = 0.6 finds minimum
x_list22, iters22 = newton(expression2, [x1, x2], 1e-6, x_init2, constraints2, with_wolfe=True, t_init=0.6)
# t = 250 finds minimum in 338 iterations
x_list32, iters32 = newton(expression3, [x1, x2], 1e-6, x_init3, constraints3, with_wolfe=True, t_init=250)
plot_contour(func1, constraints1, x_list12, "function_1_newton_wolfe")
print_4_table(func1, x_list12[-1], "newton", min1, iters12)
plot_contour(func2, constraints2, x_list22, "function_2_newton_wolfe")
print_4_table(func2, x_list22[-1], "newton", min2, iters22)
plot_contour(func3, constraints3, x_list32, "function_3_newton_wolfe")
print_4_table(func3, x_list32[-1], "newton", min3, iters32)

# Hill Climber
x_list13, iters13 = hill_climber(expression1, x_init1, [x1, x2], constraints1)
x_list23, iters23 = hill_climber(expression2, x_init2, [x1, x2], constraints2)
x_list33, iters33 = hill_climber(expression3, x_init3, [x1, x2], constraints3)
plot_contour(func1, constraints1, x_list13, "function_1_hill_climber")
print_4_table(func1, x_list13[-1], "hill climber", min1, iters13)
plot_contour(func2, constraints2, x_list23, "function_2_hill_climber")
print_4_table(func2, x_list23[-1], "hill climber", min2, iters23)
plot_contour(func3, constraints3, x_list33, "function_3_hill_climber")
print_4_table(func3, x_list33[-1], "hill climber", min3, iters33)
