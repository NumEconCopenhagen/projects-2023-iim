# Importing packages
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np


class ModelProjectClass:
    def cournotmodel(self, n, a, b, c):
        """ setup cournot model"""
        # Create a list to hold the quantities produced for each firm
        q_liste = np.ones(n)

        # Create the demand function
        def demand(q_liste, a, b):
            return a - np.matmul(b * np.ones(len(q_liste)), q_liste)

        # Create the profit function
        def profit(q_liste, firm, a, b, c):
            return (demand(q_liste, a, b) - c) * q_liste[firm]

        # Create the profit max function
        def profitmax(q_liste, firm, a, b, c):
            def opt_response(q_new):
                q_liste_new = q_liste.copy()
                q_liste_new[firm] = q_new
                return -profit(q_liste_new, firm, a, b, c)

        # Each firms optimal response (in quantities)
            q_opt_response = optimize.brute(opt_response, ((0, 100,),))

            return q_opt_response

        # Create best response
        def profitmax_vector(q_liste, a, b, c):
            
            # List to store each firms BR
            BR = []

            # Iterate over each firm and append in the BR list
            for firm in range(len(q_liste)):
                BR.append(profitmax(q_liste, firm, a, b, c))

            # Convert the list to a 1-dimensional array
            BR = np.array(BR).reshape(-1)

            return q_liste - BR

        # Optimize to find the optimal equilibrium quantities
        EQ = optimize.fsolve(profitmax_vector, q_liste, args=(a, b, c))

        return EQ

    def Results(self, n, a, b, c):
        # Create empty lists to store the results
        EQ_list = []
        n_list = []

        # Loop over n firms and append the results
        for n_val in range(2, n + 1):
            n_list.append(n_val)

            # Find the optimal quantities responding to n number of firms and append the results
            EQ_iter = self.cournotmodel(n_val, a, b, c)
            EQ_list.append(EQ_iter[0])

        return EQ_list, n_list

    # Set up for the bar plot
    def plot1(self, a, b, c):
        n = 2
        EQ = self.cournotmodel(n, a, b, c)

        plt.bar(range(n), EQ)
        plt.xlabel("Firms")
        plt.ylabel("Optimal Quantity")
        plt.title("Optimal Quantities for Two Firms")
        plt.xticks(range(n), ['Firm 1', 'Firm 2'])
        plt.ylim(0, max(EQ) * 1.2)
        plt.show()

    # Set up for the line plot
    def plot2(self, EQ_list, n_list):
        plt.plot(n_list, EQ_list)
        plt.xlabel("N Number of Firms")
        plt.ylabel("Optimal Quantity")
        plt.title("Optimal Quantities for N Numbers of Firms")
        plt.show()