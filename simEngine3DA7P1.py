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

def inverseDyanmics():

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

    timestep = 0.001
    simulation_length = 10

    times = []

    # data output for torques in generalized coordinates
    torques_1 = []
    torques_2 = []
    torques_3 = []
    torques_4 = []

    # data output for reaction forces for driving constraint
    forces_1 = []
    forces_2 = []
    forces_3 = []

    # data output for driving constraint torques in local frame
    torques_l1 = []
    torques_l2 = []
    torques_l3 = []

    # data output for overall reaction forces for body i
    react_x = []
    react_y = []
    react_z = []


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

        # mass matrix
        m = np.power(0.05,2) * 4 * 7800
        M = m*np.eye(3)

        # Inertia matrix
        J_bar = np.zeros((3,3))
        J_bar[0, 0] = (1/12) * m * (np.power(0.05, 2) + np.power(0.05, 2))
        J_bar[1, 1] = (1/12) * m * (np.power(0.05, 2) + np.power(4, 2))
        J_bar[2, 2] = (1/12) * m * (np.power(0.05, 2) + np.power(4, 2))

        Jp = 4 * G_from_p(i.p).transpose() @ J_bar @ G_from_p(i.p)


        RHS = np.zeros((7,1))
        F = np.array([0., 0., -9.81*m]).reshape((3,1))

        # F = np.array([0., 0., 0.]).reshape((3,1)) # uncomment if no gravity is desired

        # Set up the linear system to be solved for the lagrange multipliers from the EOM
        RHS[0:3,:] = F - M @ r_ddot
        RHS[3:,:] = -Jp @ p_ddot
        LHS = np.zeros((7, 7))
        PHI_Q_kinematic = np.concatenate((phi_q_rj, phi_q_dp1), axis=0)
        LHS[:, 0:6] = PHI_Q_kinematic.transpose()
        LHS[3:, 6] = i.p.reshape((4,))

        lagrange = np.linalg.solve(LHS, RHS)

        # With the lagrange multipliers, we can solve for any required torques

        # Get the reaction forces associated with the driving constraint (should be zero)
        req_force = -dp1.phi_r().reshape((1,3)).transpose() @ lagrange[5].reshape((1,1))
        forces_1.append(req_force[0])
        forces_2.append(req_force[1])
        forces_3.append(req_force[2])

        # Get the reaction torques associated with the driving constraint
        # both in the p-generalized coordinates and the local frame PI_BAR = 0.5*G*Phi^T
        req_torque = -dp1.phi_p().reshape((1,4)).transpose() @ lagrange[5].reshape((1,1))
        req_torque_local = -0.5*G_from_p(i.p) @ dp1.phi_p().reshape((1,4)).transpose() @ lagrange[5].reshape((1,1))
        torques_1.append(req_torque[0])
        torques_2.append(req_torque[1])
        torques_3.append(req_torque[2])
        torques_4.append(req_torque[3])
        torques_l1.append(req_torque_local[0])
        torques_l2.append(req_torque_local[1])
        torques_l3.append(req_torque_local[2])

        # Also get the overall reaction forces for body i to verify they are logical
        PHI_Q_kinematic = np.concatenate((phi_q_rj, phi_q_dp1, phi_q_p_norm), axis=0)
        react_forces = -PHI_Q_kinematic.transpose() @ lagrange
        react_x.append(react_forces[0])
        react_y.append(react_forces[1])
        react_z.append(react_forces[2])

        times.append(t)

    # # Plot Generalized torques - disabled by default
    # plt.figure(0)
    # plt.plot(times,torques_1, label='e0')
    # plt.plot(times,torques_2, label='e1')
    # plt.plot(times,torques_3, label='e2')
    # plt.plot(times,torques_4, label='e3')
    # plt.title("Required Torques (Generalized)")
    # plt.xlabel('Time (s)')
    # plt.ylabel('Torque (Nm)')
    # plt.legend()

    # # Plot driving constraint reaction forces - disabled by default
    # plt.figure(1)
    # plt.plot(times, forces_1, label='x')
    # plt.plot(times, forces_2, label='y')
    # plt.plot(times, forces_3, label='z')
    # plt.title("Reaction Forces")
    # plt.xlabel('Time (s)')
    # plt.ylabel('Force (N)')
    # plt.legend()

    # Plot required torques in the body i frame
    plt.figure(2)
    plt.plot(times,torques_l1, label='x')
    plt.plot(times,torques_l2, label='y')
    plt.plot(times,torques_l3, label='z')
    plt.title("Required Torques (Local)")
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()

    # Plot body i overall reaction forces
    plt.figure(3)
    plt.plot(times,react_x, label='x')
    plt.plot(times,react_y, label='y')
    plt.plot(times,react_z, label='z')
    plt.title("Body i Forces")
    plt.xlabel('Time (s)')
    plt.ylabel('Force (Nm)')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    inverseDyanmics()