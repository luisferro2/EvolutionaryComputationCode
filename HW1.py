import numpy as np
from sympy import *
import time

def evaluate(expression, x, my_actual_symbs):
    #print('my actual symbs', my_actual_symbs)
    for i, symb in enumerate(my_actual_symbs):
        expression = expression.subs(symb, x[i])
    return expression

def debug(thing, title):
    print('--------------------------------------------')
    print(title)
    print(thing)
    print('--------------------------------------------')

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
        debug(t, 't')
        debug(x + t * p, 'x + t * p')
        if l1 > r1: #or not constrained(x + t * p, constraints):
            #debug(x + t * p, 'x plus step direction')
            
            #if t < 1e-10:
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
        print('------------------------------------------------')
        print(l1, 'l1')
        r1 = evaluate(func, x, my_actual_symbs) + c1 * t * np.dot(grad_f_x, p)
        print(r1, 'r1')
        l2 = np.dot(gradient(func, x + t * p, my_actual_symbs), p)
        print(l2, 'l2')
        r2 = c2 * np.dot(grad_f_x, p)
        print(r2, 'r2')
            
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
        debug(x, 'x')
        
        
    
    return x

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
    x = x_init
    c = 0

    #debug(tol >= np.linalg.norm(gradient(f, x, my_actual_symbs)), 'tol >= gradient norm')
    #debug(tol, 'tol')
    #debug(np.linalg.norm(gradient(f, x, my_actual_symbs)), 'norm grad')

    while tol < np.linalg.norm(gradient(f, x, my_actual_symbs)):
        debug(c, 'iteration')
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
        
        if not constrained(x + t * p, constraints):
            break
        x = x + p
        
        #debug(np.linalg.norm(grad1), 'gradient norm')
        c += 1
    
    return x

def hill_climber(func, 
                 x_init, 
                 my_actual_symbs, 
                 constraints, 
                 step_size=0.01, 
                 max_iter=1000):
    """
    Continuous Hill Climber method for single objective minimization problems.
    
    :param func: Objective function to minimize.
    :param x_init: Initial guess.
    :param step_size: Size of the perturbation applied at each step.
    :param max_iter: Maximum number of iterations.
    :return: x_optimal: The optimal point found.
    """
    x = x_init
    f_val = evaluate(func, x, my_actual_symbs)
    
    for _ in range(max_iter):
        # Generate a random step
        x_new = x + step_size * (2 * np.random.rand(len(x)) - 1)
        f_val_new = evaluate(func, x_new, my_actual_symbs)
        
        # If the new solution is better, update
        if f_val_new < f_val:
            x, f_val = x_new, f_val_new
    
    return x


x_init1 = np.array([-4.0, 4.0])
x_init2 = np.array([0.5, 1.0])
x_init3 = np.array([-2.0, 2.0])

constraints1 = [(-6, 6), (-6, 6)]
constraints2 = [(-3, 3), (-2, 2)]
constraints3 = [(float('-Inf'), float('Inf')), (float('-Inf'), float('Inf'))]

x1, x2 = symbols('x1 x2')

expression1 = -2 * x1 ** 2 + 3 * x1 * x2 - 1.5 * x2 ** 2 - 1.3
expression2 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2 + x1 * x2 + \
    (-4 + 4 * x2 ** 2) * x2 ** 2
expression3 = 10 * 2 + (x1 ** 2 - 10 * cos(2 * pi * x1)) + \
    (x2 ** 2 - 10 * cos(2 * pi * x2))

x_optimal1 = gradient_descent(expression1, [x1, x2], x_init1, constraints1)
x_optimal2 = gradient_descent(expression2, [x1, x2], x_init2, constraints2)
x_optimal3 = gradient_descent(expression3, [x1, x2], x_init3, constraints3)
print(f"gradient descent optimal point for expression 1: ", x_optimal1)
print(f"gradient descent optimal point for expression 2: ", x_optimal2)
print(f"gradient descent optimal point for expression 3: ", x_optimal3)

x_optimal12 = newton(expression1, [x1, x2], 1e-6, x_init1, constraints1)
x_optimal22 = newton(expression2, [x1, x2], 1e-6, x_init2, constraints2)
x_optimal32 = newton(expression3, [x1, x2], 1e-6, x_init3, constraints3)
print(f"newton optimal point for expression 1: ", x_optimal12)
print(f"newton optimal point for expression 2: ", x_optimal22)
print(f"newton optimal point for expression 3: ", x_optimal32)

x_optimal13 = hill_climber(expression1, x_init1, [x1, x2], constraints1)
x_optimal23 = hill_climber(expression2, x_init2, [x1, x2], constraints2)
x_optimal33 = hill_climber(expression3, x_init3, [x1, x2], constraints3)
print(f"hill optimal point for expression 1: ", x_optimal13)
print(f"hill optimal point for expression 2: ", x_optimal23)
print(f"hill optimal point for expression 3: ", x_optimal33)