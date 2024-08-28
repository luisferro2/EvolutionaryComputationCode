import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from scipy import optimize

min1 = np.array([0, 0])
min2 = np.array([-0.08984201310031806242249056062, 0.7126564030207396333972658142])
min3 = np.array([0, 0])


def debug(thing, title):
    print('--------------------------------------------')
    print(title)
    print(thing)
    print('--------------------------------------------')

def evaluate(expression, x, my_actual_symbs):
    #print('my actual symbs', my_actual_symbs)
    for i, symb in enumerate(my_actual_symbs):
        expression = expression.subs(symb, x[i])
    return expression

def gradient2(f, x, my_actual_symbs):
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
    #debug(grad2, 'grad 2 expressions')

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
    n_vars = x.shape[0]
    
    grad = np.zeros((n_vars,))
    for i in range(n_vars):
        part_df = diff(f, my_actual_symbs[i])
        #debug(type(part_df), 'type expression')
        for j, symb in enumerate(my_actual_symbs):
            part_df = part_df.subs(symb, x[j])
        grad[i] = part_df
    return grad

def constrained(x, constraints):
    for i in range(len(x)):
        if x[i] < constraints[i][0] or x[i] > constraints[i][1]:
            return False
    return True

def line_search_wolfe(func, 
                      my_actual_symbs, 
                      x, 
                      p, 
                      constraints,
                      c1=1e-4, 
                      c2=0.9, 
                      t_init=0.8):
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
    #print(l1, 'l1')
    r1 = evaluate(func, x, my_actual_symbs) + c1 * t * np.dot(grad_f_x, p)
    #print(r1, 'r1')
    l2 = np.dot(gradient(func, x + t * p, my_actual_symbs), p)
    #print(l2, 'l2')
    r2 = c2 * np.dot(grad_f_x, p)
    #print(r2, 'r2')
    while not flag:
        #debug(t, 't')
        #debug(x + t * p, 'x + t * p')
        if l1 > r1 or not constrained(x + t * p, constraints):
            #debug(x + t * p, 'x plus step direction')
            
            #if t < 1e-2:
            #    break
            t *= 0.5
        elif l2 < r2:
            #debug(, 'larger')
            if not constrained(x + t * 2 * p, constraints):
                break
            t *= 2.0
        else:
            flag = True

        l1 = evaluate(func, x + t * p, my_actual_symbs)
        #print('------------------------------------------------')
        #print(l1, 'l1')
        r1 = evaluate(func, x, my_actual_symbs) + c1 * t * np.dot(grad_f_x, p)
        #print(r1, 'r1')
        l2 = np.dot(gradient(func, x + t * p, my_actual_symbs), p)
        #print(l2, 'l2')
        r2 = c2 * np.dot(grad_f_x, p)
        #print(r2, 'r2')
            
    return t


def gradient_descent(func, 
                     my_actual_symbs, 
                     x_init, 
                     constraints,
                     max_iter=1000, 
                     tol=1e-6):
    """
    Método de descenso de gradiente con control del tamaño de paso usando condiciones de Wolfe.
    
    :param func: Función objetivo a minimizar.
    :param x_init: Suposición inicial.
    :param max_iter: Número máximo de iteraciones.
    :param tol: Tolerancia para la convergencia.
    :return: x_optimal: El punto óptimo encontrado.
    """
    x_list = [x_init]

    x = x_init
    debug(x, 'x')
    
    for _ in range(max_iter):
        grad = gradient(func, x, my_actual_symbs)

        #print(np.linalg.norm(grad), 'grad norm')
        if np.linalg.norm(grad) < tol:
            break
        
        p = -grad
        t = line_search_wolfe(func, my_actual_symbs, x, p, constraints)
        debug(p, 'direction p')
        debug(t, 't')
        debug(x + t * p, 'new x')
        if not constrained(x + t * p, constraints):
            break
        x = x + t * p
        x_list += [x]
        debug(x, 'x')
    return x_list

def newton(f, 
           my_actual_symbs, 
           tol, 
           x_init, 
           constraints):
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

    #debug(tol >= np.linalg.norm(gradient(f, x, my_actual_symbs)), 'tol >= gradient norm')
    #debug(tol, 'tol')
    

    while tol < np.linalg.norm(gradient(f, x, my_actual_symbs)):
        debug(c, 'iteration')
        debug(np.linalg.norm(gradient(f, x, my_actual_symbs)), 'norm grad')
        grad1 = gradient(f, x, my_actual_symbs)
        grad2 = gradient2(f, x, my_actual_symbs)
        debug(x, 'x values')
        
        #debug(grad1, 'gradient')
        #debug(grad2, 'gradient 2')
        inv_grad2 = -np.linalg.inv(grad2)
        debug(inv_grad2, 'inverse grad 2')
        p = np.dot(inv_grad2, grad1)
        debug(p, 'direction p')
        t = line_search_wolfe(f, my_actual_symbs, x, p, constraints)
        debug(t, 't')
        
        if not constrained(x + t * p, constraints):
            break
        x = x + t * p
        x_list += [x]
        
        #debug(np.linalg.norm(grad1), 'gradient norm')
        c += 1
    
    return x_list

def hill_climber(func, 
                 x_init, 
                 my_actual_symbs, 
                 constraints, 
                 step_size=0.01, 
                 max_iter=10000):
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


x_init1 = np.array([-4.0, 4.0])
x_init2 = np.array([0.5, 1.0])
x_init3 = np.array([-2.0, 2.0])

constraints1 = [(-6, 6), (-6, 6)]
constraints2 = [(-3, 3), (-2, 2)]
constraints3 = [(-5.12, 5.12), (-5.12, 5.12)]

x1, x2 = symbols('x1 x2')

# Expression 1 is multiplied by -1 to convert to minimization problem
expression1 = -(-2 * x1 ** 2 + 3 * x1 * x2 - 1.5 * x2 ** 2 - 1.3)
expression2 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2 + x1 * x2 + \
    (-4 + 4 * x2 ** 2) * x2 ** 2
expression3 = 10 * 2 + (x1 ** 2 - 10 * cos(2 * pi * x1)) + \
    (x2 ** 2 - 10 * cos(2 * pi * x2))



def func1(x):
    x1, x2 = x
    return -(-2 * x1 ** 2 + 3 * x1 * x2 - 1.5 * x2 ** 2 - 1.3)
def func2(x):
    x1, x2 = x
    return (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2 + x1 * x2 + \
    (-4 + 4 * x2 ** 2) * x2 ** 2
def func3(x):
    x1, x2 = x
    return 10 * 2 + (x1 ** 2 - 10 * np.cos(2 * np.pi * x1)) + \
    (x2 ** 2 - 10 * np.cos(2 * np.pi * x2))

# Plot for expression 1
def plot_contour(func, constraints, x_list):
    delta = 0.025
    x = np.arange(constraints[0][0], constraints[0][1], delta)
    y = np.arange(constraints[1][0], constraints[1][1], delta)
    X, Y = np.meshgrid(x, y)

    Xin = np.array([[xi, yi] for xi, yi in zip(x, y)])
    Z = func([X, Y])
    #debug(Z.shape, 'Z shape')
    #fig, ax = plt.subplots()
    plt.contourf(X, Y, Z)
    plt.colorbar()
    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    #ax.clabel(CS, inline=False, fontsize=10)
    #plt.set_title('Simplest default with labels')
    x1_list = [x[0] for x in x_list[:]]
    x2_list = [x[1] for x in x_list[:]]
    
    plt.plot(x1_list, x2_list, 'r-.', )
    lines, = plt.plot(x_list[0][0], x_list[0][1], 'bo', label='start')
    linee, = plt.plot(x_list[-1][0], x_list[-1][1], 'b*', label='end')
    lines.set_color('black')
    linee.set_color('black')
    plt.legend()
    plt.show()

def print_4_table(func, final_points, algo, min_x, iters):

    titles = 'Algorithm & Point Found & Evaluation & Optimal point & Real Minimum & 2-norm error between points & iterations'
    
    curr_str = algo + ' & ' + str(final_points) + ' & ' +\
        str(func1(final_points)) + ' & ' + str(min_x) + ' & ' + str(func(min_x)) + \
        ' & ' + str(np.linalg.norm(min1 - final_points)) + ' & ' + str(iters)
            
    debug(curr_str, titles)
# 0.8 for all 3 to work 
# t = 10 finds faster
#x_list1 = gradient_descent(expression1, [x1, x2], x_init1, constraints1)
# t = 8.1 finds faster
#x_list2 = gradient_descent(expression2, [x1, x2], x_init2, constraints2)
# t = 1 finds minimum
#x_list3 = gradient_descent(expression3, [x1, x2], x_init3, constraints3)
# print(f"gradient descent optimal point for expression 1: ", x_list1)
# print(f"gradient descent optimal point for expression 2: ", x_list2)
# print(f"gradient descent optimal point for expression 3: ", x_list3)
#debug(len(x_list2), 'iterations')
# plot_contour(func1, constraints1, x_list1)
# plot_contour(func2, constraints2, x_list2)
#plot_contour(func3, constraints3, x_list3)

# 1 for all to work (without wolfe)
# t = 1
#x_list12 = newton(expression1, [x1, x2], 1e-6, x_init1, constraints1)
# t = 0.6 finds minimum
#x_list22 = newton(expression2, [x1, x2], 1e-6, x_init2, constraints2)
# t = 250 finds minimum in 338 iterations
#x_list32 = newton(expression3, [x1, x2], 1e-6, x_init3, constraints3)
#print(f"newton optimal point for expression 1: ", x_list12)
#print(f"newton optimal point for expression 2: ", x_list22)
#print(f"newton optimal point for expression 3: ", x_list32)
#debug(len(x_list32), 'iterations')
#plot_contour(func1, constraints1, x_list12)
#plot_contour(func2, constraints2, x_list22)
#plot_contour(func3, constraints3, x_list32)

x_list13, iters13 = hill_climber(expression1, x_init1, [x1, x2], constraints1)
#x_list23 = hill_climber(expression2, x_init2, [x1, x2], constraints2)
#x_list33 = hill_climber(expression3, x_init3, [x1, x2], constraints3)
# print(f"hill optimal point for expression 2: ", x_list23)
# print(f"hill optimal point for expression 3: ", x_list33)
# debug(len(x_list13), 'iterations')
plot_contour(func1, constraints1, x_list13)
#plot_contour(func2, constraints2, x_list23)
#plot_contour(func3, constraints3, x_list33)
print_4_table(func1, x_list13[-1], 'hill climber', min1, iters13)



