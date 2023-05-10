import numpy as np
from scipy import optimize
import sympy as sm
from sympy import *
from types import SimpleNamespace
import matplotlib as plt
from ipywidgets import interact, FloatSlider

class ModelProjectClass:
    
    def __init__(self, n):
        """ setup model """

        # a. creating namespaces
        self.par = SimpleNamespace()

        # b. defining the cost function
        def c(q):
            return self.par.c * np.sum(q)
        self.cost = c

        # c. dDefininf the revenue
        def r(p):
            return p * self.q(p)
        self.rev = r

    # defining the inverse demand funtion
    def q(self, p):
        return (self.par.a - p) / self.par.b / self.par.n

    def EQ(self, a, b, c, n):
        """ Finding equilibrium for all firms """

        # a. setting parameters
        self.par.a = a
        self.par.b = b
        self.par.c = c
        self.par.n = n

        # b. optimal EQ amounts given the optimal price
        initial_guess = np.ones(n) / n
        p_opt = optimize.minimize(lambda p: -self.rev(p) + self.cost(np.ones(n) * self.q(p)), x0=self.par.a, bounds=[(0, None)])
        
        return np.maximum(self.q(p_opt.x) * n, 0)