from types import SimpleNamespace

import numpy as np

from scipy import optimize

import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)
        par.lnwFwH_vec = np.log(par.wF_vec/par.wM)
        par.sigma_vec = [0.5, 1, 1.5]
        par.alpha_vec = [0.25, 0.50, 0.75]

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF
        
        # b. home production
         # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
            
        elif par.sigma == 0:
            H = np.minimum(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
            
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        sol.LM = LM[j]
        opt.HM = HM[j]
        sol.LF = LF[j]
        opt.HF = HF[j]
        
        opt.lnHFHM = np.log(opt.HF/opt.HM)
        opt.HF_div_HM = opt.HF/opt.HM
        opt.alpha = par.alpha
        opt.sigma = par.sigma
        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
        
        opt.par_list = [opt.HM, opt.HF, opt.HF_div_HM, opt.lnHFHM, opt.alpha, opt.sigma]
        return opt.par_list

    
    def solve(self,do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        def obj(x):
            return -self.calc_utility(x[0], x[1], x[2], x[3])

        xguess = [12, 12, 12, 12]
        constraint1 = ({"type": "ineq", "fun": lambda x: -x[0]-x[1]+24})
        constraint2 = ({"type": "ineq", "fun": lambda x: -x[2]-x[3]+24})
        constraints = [constraint1, constraint2]
        bounds = [(0,24)]*4
        results = optimize.minimize(obj,xguess,method='SLSQP',  bounds = bounds, constraints = constraints)
        #opt.results = results.x
        sol.LM = results.x[0]
        sol.HM = results.x[1]
        sol.LF = results.x[2]
        sol.HF = results.x[3]

        opt.lnHFHM = np.log(sol.HF/sol.HM)
        
        return opt.lnHFHM
    
    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        
        xguess = [12, 12, 12, 12]
        par = self.par
        sol = self.sol

        objective = (sol.beta0 - par.beta0_target)**2 + (sol.beta1 - par.beta1_target)**2
        results = optimize.minimize(objective,xguess,args=(par,),method='Nelder-Mead')
        
        return results