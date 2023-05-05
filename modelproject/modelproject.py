import numpy as np
from scipy import optimize
import sympy as sm
from types import SimpleNamespace 

class modelprojectClass:

    def __init__(self):
        """ setup model """

    # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. setting  endogenous variabels
        par.a = 1
        par.b = 1
        par.c = 1

        # c. setting endogenous variables 
        def p(self, x1, x2): 
            x = x1 + x2
            return par.a - par.b * x
        
        def C(x):
            C = par.c * x
            return C
    
    def solve_ss(alpha, c):  
    

    # a. Objective function, depends on k (endogenous) and c (exogenous).
        f = lambda k: k**alpha - c
        obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
        result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
        return result