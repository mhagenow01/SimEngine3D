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

def convergenceAnalysis():

    ###### Backwards Euler ################

    # solve a linear system for delta
    # update, until updates are small enough

    h_vals = np.logspace(-3, -1, 20) # h values distributed logarithmically to make slope plot cleaner
    BDF_errors = []
    BE_errors = []


    for h in h_vals:
        print("h:",h)
        ############# BACKWARDS EULER ###################
        max_iterations = 10

        # Initial Conditions (Note: This is an LTV System)
        y_initial= 1
        t_initial = 1

        yn_prev = y_initial

        # solve for 10 seconds
        simulation_length=10

        final_val = 0 # this makes sure the exact solution uses whatever our last t value is if not exactly 10

        for tt in np.arange(t_initial+h, simulation_length+h, h):
            # Solve for the timestep
            final_val = tt

            num_iterations = 0
            curr_norm = 10 ** 10 # Start with large number so first iteration always executes

            # Starting point for new values
            yn = yn_prev

            while  num_iterations<max_iterations: # choosing fixed iterations for convergence comparison
                # Calculate the function to solve for the new values via Newton-Raphson
                g = np.zeros((1,1))
                g[0,0] = yn + h * yn ** 2 + h/(tt ** 4) - yn_prev

                # Calculate the Jacobian
                J = np.zeros((1,1))
                J[0,0]=1+2*yn*h

                # Calculate Correction
                delta_xy = np.linalg.solve(J,-g)

                # Update guesses
                yn = yn + delta_xy[0]

                curr_norm = delta_xy[0]

                num_iterations += 1
            yn_prev = yn


        # Value from analytical solution to generate error
        closed_form_value = 1/(final_val)+1/(final_val **2) * np.tan((1/final_val) + np.pi - 1)

        # Add backwards euler for this value of h
        BE_errors.append(np.abs(closed_form_value-yn))


        ###### 4th Order BDF Method ###################################
        max_iterations = 10

        # Initial Conditions (Note: This is an LTV System)

        t_initial = 1

        # Prime the method with the last 4 values
        # I chose to seed forward meaning give it the first four values after 1 second
        # You could also do it backwards if desired... (ie the four values before 1 second)
        prev_time_1 = t_initial+3*h
        yn_prev_1 = 1/(prev_time_1)+1/(prev_time_1 **2) * np.tan((1/prev_time_1) + np.pi - 1)
        prev_time_2 = t_initial+2*h
        yn_prev_2 = 1 / (prev_time_2) + 1 / (prev_time_2 ** 2) * np.tan((1 / prev_time_2) + np.pi - 1)
        prev_time_3 = t_initial+1*h
        yn_prev_3 = 1 / (prev_time_3) + 1 / (prev_time_3 ** 2) * np.tan((1 / prev_time_3) + np.pi - 1)
        prev_time_4 = t_initial
        yn_prev_4 = 1 / (prev_time_4) + 1 / (prev_time_4 ** 2) * np.tan((1 / prev_time_4) + np.pi - 1)

        # solve for 10 seconds
        simulation_length = 10

        final_val = 0.0 # this makes sure the exact solution uses whatever our last t value is if not exactly 10
        for tt in np.arange(t_initial + 4*h, simulation_length + h, h):
            # Solve for the timestep
            final_val = tt

            num_iterations = 0

            # Starting point for new values
            yn = yn_prev_1

            while num_iterations < max_iterations: # choosing fixed iterations for convergence comparison
                # Calculate the function to solve for the new values via Newton-Raphson
                g = np.zeros((1, 1))
                g[0, 0] = yn -(48/25)*yn_prev_1 +(36/25)*yn_prev_2-(16/25)*yn_prev_3+(3/25)*yn_prev_4+\
                          (12/25)*h * yn ** 2 + (12/25)*h / (tt ** 4)

                # Calculate the Jacobian
                J = np.zeros((1, 1))
                J[0, 0] = 1 + (24/25) * yn * h

                # Calculate Correction
                delta_xy = np.linalg.solve(J, -g)

                # Update guesses
                yn = yn + delta_xy[0]

                curr_norm = delta_xy[0]

                num_iterations += 1

            # Update last 4
            yn_prev_4 = yn_prev_3
            yn_prev_3 = yn_prev_2
            yn_prev_2 = yn_prev_1
            yn_prev_1 = yn

        # Calculate BDF error for plot

        closed_form_value = 1 / (final_val) + 1 / (final_val ** 2) * np.tan((1 / final_val) + np.pi - 1)
        BDF_errors.append(np.abs(closed_form_value - yn))


    #Plot the results on a linear scale
    plt.figure(0)
    plt.plot(h_vals,BE_errors, label='backwards euler')
    plt.plot(h_vals,BDF_errors, label='BDF')
    plt.title("Convergence Analysis")
    plt.xlabel('h step size')
    plt.ylabel('IVP error')
    plt.legend()

    # Print the results on a log-log scale
    plt.figure(1)
    plt.plot(h_vals,BE_errors, label='backwards euler')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.plot(h_vals,BDF_errors, label='BDF')
    plt.title("Convergence Analysis")
    plt.xlabel('h step size')
    plt.ylabel('IVP error')
    plt.legend()

    # Fit lines to the log-log plots to look at the slope
    mbe, bbe = np.polyfit(np.log(h_vals),np.log(BE_errors), 1)
    mbdf,bbdf = np.polyfit(np.log(h_vals),np.log(BDF_errors), 1)

    print("BE Log slope:", mbe)
    print("BDF Log slope:", mbdf)

    plt.show()


if __name__ == "__main__":
    convergenceAnalysis()