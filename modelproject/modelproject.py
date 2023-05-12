import numpy as np
from scipy import optimize
import sympy as sm
from sympy import *
from types import SimpleNamespace
import matplotlib as plt


class ModelProjectClass:
    
    def __init__(self, n):
        """ setup model """

        # a. creating namespaces
        self.par = SimpleNamespace()

        # b. defining the cost function
    def cost(self, q):
        return self.par.c * np.sum(q)
    #self.cost = c

    # c. dDefininf the revenue
    def rev(self,p):
        # Imposing restrictions that the price cannot be negative
        if p >= 0:
            return p * self.q(p)
        else:
            return -np.inf
    


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
        
        sol_q = np.maximum(self.q(p_opt.x) * n, 0)
        sol_p = self.par.a - self.par.b * sol_q * self.par.n
 
        return sol_q, sol_p, self.par.a, self.par.b, self.par.c, self.par.n