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

def setUpPendulum():

    ###### Define the two bodies ################
    # Body j is going to be the ground and as such doesn't have any generalized coordinates
    j = RigidBody()  # defaults are all zero
    s_bar_j_q = np.array([0., 0., 0.]).reshape((3,1))


    # Initial configuration for body i
    r_i = np.array([0., 2*np.sqrt(2)/2, -2*np.sqrt(2)/2]).reshape((3,1))

    # Need to convert A TO P for use in this formulation
    A_i_initial = np.array([[0., 0., 1],[np.sqrt(2)/2, np.sqrt(2)/2, 0],[-np.sqrt(2)/2, np.sqrt(2)/2, 0]])
    p_i_initial = p_from_A(A_i_initial)
    i = RigidBody(r_i,p_i_initial) # velocities as defaults
    s_bar_i_q = np.array([-2., 0., 0.]).reshape((3,1))


    # Define the local vectors
    c_bar_j = np.array([1., 0., 0.]).reshape((3,1))
    a_bar_i = np.array([1., 0., 0.]).reshape((3,1))
    b_bar_i = np.array([0., 1., 0.]).reshape((3,1))

    # Driving constraint will use y' and y to define trig function
    d_bar_j = np.array([0., -1., 0.]).reshape((3,1))

    ###### Define the geometric constraints ################
    RJ = Revolute(i, s_bar_i_q, a_bar_i, b_bar_i, j, s_bar_j_q, c_bar_j, j_ground=True)

    ###### Define the driving constraints for time zero ################
    # Function: theta = pi/4 * cos(2t)

    dp1 = DP1(i, a_bar_i, j, d_bar_j, 0, 0, 0, j_ground=True)

    p_norm_i = P_norm(i)


    ###### Inverse Dynamics (and also kinematics) Analysis ##########

    timestep = 0.01
    simulation_length = 10

    times = []
    torques_1 = []
    torques_2 = []
    torques_3 = []
    torques_4 = []

    for t in np.arange(0, simulation_length, timestep):
        # Keep track of progress
        print(t)

        # Calculate function values for the given time-step
        f = np.sin(np.pi / 4 * np.cos(2 * t))
        f_dot = -np.cos(np.pi / 4 * np.cos(2 * t)) * np.pi / 2 * np.sin(2 * t)
        f_ddot = -(np.power(np.pi, 2) / 4) * np.sin(np.pi / 4 * np.cos(2 * t)) * np.power(np.sin(2 * t), 2) - \
                 np.pi * np.cos(np.pi / 4 * np.cos(2 * t)) * np.cos(2 * t)

        RJ.update(i, j)
        dp1.update(i, j, f, f_dot, f_ddot)
        p_norm_i.update(i)

        # Newton-Raphson to try and converge on the values of pi and ri
        # Hard-coded to 10 iterations. Could also do based on a tolerance.
        num_iterations=10
        for ii in range(0,num_iterations):
            # Current guesses for generalized coordinates
            q0 = np.concatenate((i.r,i.p),axis=0)


            PHI = np.concatenate((RJ.phi(), dp1.phi(), p_norm_i.phi()), axis=0)
            phi_q_rj = np.concatenate((RJ.phi_r(), RJ.phi_p()), axis=1)
            phi_q_dp1 = np.concatenate((dp1.phi_r(), dp1.phi_p()), axis=1)
            phi_q_p_norm = np.concatenate((p_norm_i.phi_r(), p_norm_i.phi_p()), axis=1)
            PHI_Q = np.concatenate((phi_q_rj, phi_q_dp1, phi_q_p_norm), axis=0)

            # Assess new guess
            q1 = q0 - np.linalg.inv(PHI_Q) @ PHI
            i.r = (q1[0:3][:]).reshape((3,1))
            i.p = (q1[3:][:]).reshape((4,1))

            RJ.update(i, j)
            dp1.update(i, j, f, f_dot, f_ddot)
            p_norm_i.update(i)



        # Now solve for the velocities
        NU = np.concatenate((RJ.nu(), dp1.nu(), p_norm_i.nu()), axis=0)
        q_dot = np.linalg.solve(PHI_Q, NU)
        r_dot = (q_dot[0:3][:]).reshape((3,1))
        p_dot = (q_dot[3:][:]).reshape((4,1))




        # Update Velocities for Acceleration Analysis
        i.r_dot=r_dot
        i.p_dot=p_dot
        RJ.update(i, j)
        dp1.update(i, j, f, f_dot, f_ddot)

        # Now solve for the accelerations
        GAMMA = np.concatenate((RJ.gamma(), dp1.gamma(), p_norm_i.gamma()), axis=0)
        q_ddot = np.linalg.solve(PHI_Q, GAMMA)
        r_ddot = (q_ddot[0:3][:]).reshape((3, 1))
        p_ddot = (q_ddot[3:][:]).reshape((4, 1))


        # With the accelerations, we can solve for the lagrange multipliers

        # mass matrrix
        m = np.power(0.05,2) * 4 * 7800
        M = m*np.eye(3)

        # Inertia matrix
        J_bar = np.zeros((3,3))
        J_bar[0, 0] = (1/12) * m * (np.power(0.05, 2) + np.power(0.05, 2))
        J_bar[1, 1] = (1/12) * m * (np.power(0.05, 2) + np.power(4, 2))
        J_bar[2, 2] = (1/12) * m * (np.power(0.05, 2) + np.power(4, 2))

        Jp = 4 * G_from_p(i.p).transpose() @ J_bar @ G_from_p(i.p)


        RHS = np.zeros((7,1))
        RHS[0:3,:] = - M @ r_ddot
        RHS[3:,:] = -Jp @ p_ddot

        LHS = np.zeros((7, 7))
        PHI_Q_kinematic = np.concatenate((phi_q_rj, phi_q_dp1), axis=0)
        LHS[:, 0:6] = PHI_Q_kinematic.transpose()
        LHS[3:, 6] = i.p.reshape((4,))

        lagrange = np.linalg.solve(LHS, RHS)

        # With the lagrange multipliers, we can solve for any required torques
        req_torque = dp1.phi_p().reshape((1,4)).transpose() @ lagrange[5].reshape((1,1))
        torques_1.append(req_torque[0])
        torques_2.append(req_torque[1])
        torques_3.append(req_torque[2])
        torques_4.append(req_torque[3])
        times.append(t)

    #Plot the results
    plt.figure(0)
    plt.plot(times,torques_1, label='e0')
    plt.plot(times,torques_2, label='e1')
    plt.plot(times,torques_3, label='e2')
    plt.plot(times,torques_4, label='e3')
    plt.title("Required Torques")
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()


    plt.show()



if __name__ == "__main__":
    setUpPendulum()