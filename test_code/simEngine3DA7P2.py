""" Peforms the inverse dynamics analysis for a
revolute joint pendulum
"""

import numpy as np
from Utilities.RigidBody import RigidBody
from Utilities.kinematic_identities import p_from_A, A_from_p, a_dot_from_p_dot, a_ddot, G_from_p
from GCons.Revolute import Revolute
from GCons.DP1 import DP1
from GCons.P_norm import P_norm
import matplotlib.pyplot as plt

def solveIVP():
    ###### Backwards Euler ################

    # solve a linear system for delta
    # update, until updates are small enough

    h = 0.001
    max_iterations = 10
    epsilon = 10 ** -10

    time = []
    x_val = []
    y_val = []

    # Initial Conditions
    x_initial= 1
    y_initial= 2

    time.append(0.)
    x_val.append(x_initial)
    y_val.append(y_initial)

    xn_prev = x_initial
    yn_prev = y_initial

    # solve for 20 seconds
    simulation_length=20

    for tt in np.arange(h, simulation_length, h):
        # Solve for the timestep
        print(tt)

        num_iterations = 0

        curr_norm = 10 ** 10 # Start with large number so first iteration always executes

        # Starting point for new values
        xn = xn_prev
        yn = yn_prev

        while np.abs(curr_norm) > (epsilon) and num_iterations<max_iterations:

            # Calculate the function to solve for the new values via Newton-Raphson
            g = np.zeros((2,1))
            g[0,0] = xn * (1+h) + (4*xn*yn*h)/(1+np.power(xn,2))-xn_prev
            g[1,0] = yn -h*xn + (xn*yn*h)/(1+np.power(xn,2))-yn_prev

            # Calculate the Jacobian
            J = np.zeros((2,2))
            J[0,0]=(1+h)+4*yn*h*(1-np.power(xn,2))/np.power(1+np.power(xn,2),2)
            J[0,1]=4*xn*h/(1+np.power(xn,2))
            J[1, 0] = -h + yn * h * (1 - np.power(xn, 2)) / np.power(1 + np.power(xn, 2), 2)
            J[1, 1] = 1 + xn * h / (1 + np.power(xn, 2))

            # Calculate Correction
            delta_xy = np.linalg.solve(J,-g)

            # Update guesses
            xn = xn + delta_xy[0]
            yn = yn + delta_xy[1]

            curr_norm = np.linalg.norm(delta_xy)

            num_iterations += 1

        # Save data for plotting
        x_val.append(xn)
        y_val.append(yn)
        time.append(tt)

        # For next iteration
        xn_prev = xn
        yn_prev = yn


    #Plot the results
    plt.figure(0)
    plt.plot(time,x_val, label='x')
    plt.plot(time,y_val, label='y')
    plt.title("Initial Value Problem")
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    solveIVP()