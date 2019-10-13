""" Function to set up a revolute joint pendulum
 with the parameters for HW #6 and perform the kinematic analysis
"""

import numpy as np
from Utilities.RigidBody import RigidBody
from Utilities.kinematic_identities import p_from_A, A_from_p, a_dot_from_p_dot, a_ddot
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
    r_i = np.array([0., np.sqrt(2)/2, -np.sqrt(2)/2]).reshape((3,1))

    # Need to convert A TO P for use in this formulation
    A_i_initial = np.array([[0., 0., 1],[np.sqrt(2)/2, np.sqrt(2)/2, 0],[-np.sqrt(2)/2, np.sqrt(2)/2, 0]])
    p_i_initial = p_from_A(A_i_initial)
    i = RigidBody(r_i,p_i_initial) # velocities as defaults
    s_bar_i_q = np.array([-1., 0., 0.]).reshape((3,1))


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


    ###### Kinematics Analysis ##########

    timestep = 0.001
    simulation_length = 10

    positions_x=[]
    positions_y=[]
    positions_z=[]
    velocities_x=[]
    velocities_y=[]
    velocities_z=[]
    accel_x=[]
    accel_y=[]
    accel_z=[]
    q_positions_x=[]
    q_positions_y=[]
    q_positions_z=[]
    q_velocities_x=[]
    q_velocities_y=[]
    q_velocities_z=[]
    q_accel_x=[]
    q_accel_y=[]
    q_accel_z=[]

    times = []

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
        # Hard-coded to 10 iterations
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

        # Position Analysis for O'
        positions_x.append(i.r[0])
        positions_y.append(i.r[1])
        positions_z.append(i.r[2])
        times.append(t)

        # Position Analysis for Q
        pos_q = i.r + A_from_p(i.p) @ s_bar_i_q
        q_positions_x.append(pos_q[0])
        q_positions_y.append(pos_q[1])
        q_positions_z.append(pos_q[2])

        # Now solve for the velocities
        NU = np.concatenate((RJ.nu(), dp1.nu(), p_norm_i.nu()), axis=0)
        q_dot = np.linalg.solve(PHI_Q, NU)
        r_dot = (q_dot[0:3][:]).reshape((3,1))
        p_dot = (q_dot[3:][:]).reshape((4,1))
        velocities_x.append(r_dot[0])
        velocities_y.append(r_dot[1])
        velocities_z.append(r_dot[2])

        # Velocity Analysis for Q
        vel_q = r_dot + a_dot_from_p_dot(i.p,s_bar_i_q,p_dot)
        q_velocities_x.append(vel_q[0])
        q_velocities_y.append(vel_q[1])
        q_velocities_z.append(vel_q[2])

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
        accel_x.append(r_ddot[0])
        accel_y.append(r_ddot[1])
        accel_z.append(r_ddot[2])

        # Acceleration Analysis for Q
        acc_q = r_ddot + a_ddot(i.p, s_bar_i_q, p_dot, p_ddot)
        q_accel_x.append(acc_q[0])
        q_accel_y.append(acc_q[1])
        q_accel_z.append(acc_q[2])


    #Plot the results
    plt.figure(0)
    plt.plot(times,positions_x, label='x')
    plt.plot(times,positions_y, label='y')
    plt.plot(times,positions_z, label='z')
    plt.title("Position of Point O'")
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    plt.figure(1)
    plt.plot(times, velocities_x, label='x')
    plt.plot(times, velocities_y, label='y')
    plt.plot(times, velocities_z, label='z')
    plt.title("Velocity of Point O'")
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    plt.figure(2)
    plt.plot(times, accel_x, label='x')
    plt.plot(times, accel_y, label='y')
    plt.plot(times, accel_z, label='z')
    plt.title("Acceleration of Point O'")
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()

    # Plot the results
    plt.figure(3)
    plt.plot(times, q_positions_x, label='x')
    plt.plot(times, q_positions_y, label='y')
    plt.plot(times, q_positions_z, label='z')
    plt.title("Position of Point Q")
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    plt.figure(4)
    plt.plot(times, q_velocities_x, label='x')
    plt.plot(times, q_velocities_y, label='y')
    plt.plot(times, q_velocities_z, label='z')
    plt.title("Velocity of Point Q")
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    plt.figure(5)
    plt.plot(times, q_accel_x, label='x')
    plt.plot(times, q_accel_y, label='y')
    plt.plot(times, q_accel_z, label='z')
    plt.title("Acceleration of Point Q")
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()

    plt.show()


    ###### Print the Desired Initial Properties ########################
    NU = np.concatenate((RJ.nu(), dp1.nu().reshape((1,1))), axis=0)
    # print("NU:")
    # print(NU)

    GAMMA = np.concatenate((RJ.gamma(), dp1.gamma()), axis=0)
    # print("GAMMA:")
    # print(GAMMA)


if __name__ == "__main__":
    setUpPendulum()