""" Peforms a dynamics analysis for a
revolute joint pendulum
"""

import numpy as np
from Utilities.RigidBody import RigidBody
from Utilities.kinematic_identities import p_from_A, A_from_p, a_dot_from_p_dot, a_ddot, G_from_p, E_from_p
from GCons.Revolute import Revolute
from GCons.DP1 import DP1
from GCons.P_norm import P_norm
import matplotlib.pyplot as plt
import time

def dynamicsAnalysis():
    ###### SIMULATION PARAMETERS #################
    sim_length = 10.
    h = 0.001 # step for solver

    start = time.time()

    ###### Define the two bodies ################
    # Body j is going to be the ground and as such doesn't have any generalized coordinates
    j = RigidBody()  # defaults are all zero
    s_bar_j_q = np.array([0., 0., 0.]).reshape((3,1))

    # Initial configuration for body i
    r_i = np.array([0., 2*np.sqrt(2)/2, -2*np.sqrt(2)/2]).reshape((3,1))
    r_dot_i = np.array([0., 0., 0.]).reshape((3,1))  # no initial velocity

    # Need to convert A TO P for use in this formulation
    A_i_initial = np.array([[0., 0., 1],[np.sqrt(2)/2, np.sqrt(2)/2, 0],[-np.sqrt(2)/2, np.sqrt(2)/2, 0]])
    p_i_initial = p_from_A(A_i_initial)
    p_dot_i = np.array([0., 0., 0., 0.]).reshape((4,1)) # orientation is not initially changing

    # mass matrix
    m_i = np.power(0.05, 2) * 4 * 7800
    M_i = m_i * np.eye(3)

    # Inertia matrix
    J_bar_i = np.zeros((3, 3))
    J_bar_i[0, 0] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(0.05, 2))
    J_bar_i[1, 1] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(4, 2))
    J_bar_i[2, 2] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(4, 2))

    i = RigidBody(r_i,p_i_initial,r_dot_i,p_dot_i,m_i) # velocities as defaults
    s_bar_i_q = np.array([-2., 0., 0.]).reshape((3,1))

    # Define the local vectors of the revolute joint
    c_bar_j = np.array([1., 0., 0.]).reshape((3,1))
    a_bar_i = np.array([1., 0., 0.]).reshape((3,1))
    b_bar_i = np.array([0., 1., 0.]).reshape((3,1))

    ###### Define the geometric constraints ################
    RJ = Revolute(i, s_bar_i_q, a_bar_i, b_bar_i, j, s_bar_j_q, c_bar_j, j_ground=True)
    RJ.update(i,j)

    p_norm_i = P_norm(i)

    # Compute the initial conditions for acceleration and the lagrange multipliers
    # by solving a linear system

    LHS = np.zeros((13,13))
    LHS[0:3,0:3]=M_i
    LHS[0:3,8:13]=RJ.phi_r().transpose()
    Jp = 4 * G_from_p(i.p).transpose() @ J_bar_i @ G_from_p(i.p)

    LHS[3:7,3:7] = Jp
    LHS[3:7,7] = i.p.reshape((4,))
    LHS[3:7,8:13] = RJ.phi_p().transpose()

    LHS[7,3:7]=i.p.reshape((1,4))

    LHS[8:13,0:3]=RJ.phi_r()
    LHS[8:13,3:7]=RJ.phi_p()

    Fg = np.array([0., 0.,-9.81 * m_i]).reshape((3,1))

    RHS = np.zeros((13,1))
    RHS[0:3,0] = Fg.reshape((3,))
    RHS[3:7,0] = np.zeros((4,)) # no torques
    RHS[7,0] = p_norm_i.gamma()
    RHS[8:13,0]=RJ.gamma().reshape((5,))

    initial_conds = np.linalg.solve(LHS,RHS)

    # Lists for plotting
    x_i = []
    y_i = []
    z_i = []
    w_x = []
    w_y = []
    w_z = []
    tor_x = []
    tor_y = []
    tor_z = []
    norm_vel_constraint = []
    times = []


    # Keep track of terms for the BDF
    r_i_prev = np.zeros((3,1))
    r_dot_i_prev = np.zeros((3,1))
    p_i_prev = np.zeros((4, 1))
    p_dot_i_prev = np.zeros((4, 1))

    r_i_ddot = initial_conds[0:3, 0].reshape((3, 1))
    p_i_ddot = initial_conds[3:7, 0].reshape((4, 1))
    lagrangeP = initial_conds[7, 0]
    lagrange = initial_conds[8:13, 0].reshape((5, 1))


    printIter = 0
    for tt in np.arange(h,sim_length,h):


        # Step 0  - Prime the solver
        # Don't actually need to do anything in this implementation

        # Need two previous sets of values for the second order BDF
        r_i_two_prev = r_i_prev
        r_dot_i_two_prev = r_dot_i_prev
        p_i_two_prev = p_i_prev
        p_dot_i_two_prev = p_dot_i_prev

        r_i_prev = i.r
        r_dot_i_prev = i.r_dot
        p_i_prev = i.p
        p_dot_i_prev = i.p_dot

        # Step 1 - Compute position and velocity using BDF

        if tt==h: #first iteration
            # Use BDF of order 1 - compute static terms
            cr = r_i_prev + h*r_dot_i_prev
            crdot = r_dot_i_prev
            cp = p_i_prev + h * p_dot_i_prev
            cpdot = p_dot_i_prev
            beta_0 = 1

        else: # every iteration after the first iteration
            # Use BDF of order 2 - compute static terms
            cr = (4/3)*r_i_prev - (1/3)*r_i_two_prev + (8/9)*h*r_dot_i_prev - (2/9)*h*r_dot_i_two_prev
            crdot = (4/3)*r_dot_i_prev-(1/3)*r_dot_i_two_prev
            cp = (4 / 3) * p_i_prev - (1 / 3) * p_i_two_prev + (8 / 9) * h * p_dot_i_prev - (
                        2 / 9) * h * p_dot_i_two_prev
            cpdot = (4 / 3) * p_dot_i_prev - (1 / 3) * p_dot_i_two_prev
            beta_0 = (2/3)

        if (printIter % 100) == 0:

            print("r:" , i.r.transpose(), " t:",tt)
            print('v:', i.r_dot.transpose())
            print('a:', r_i_ddot.transpose())
            # g_temp = calculateResidual(r_i_ddot, p_i_ddot, lagrange, lagrangeP, i, j, RJ, M_i, J_bar_i, cr, crdot, cp, cpdot,
            #                   beta_0, h, p_norm_i, Fg)
            # print("g:" , g_temp.transpose(), " t:",tt)

        printIter = printIter+1

        # Compute non-linear residual
        iterations = 0
        correction = np.ones((13,1)) # seed so first iteration runs

        # Calculate the jacobian for the iterative process (only done once per timestep)

        PSI = computeJacobian(r_i_ddot, p_i_ddot, lagrange, lagrangeP, i, j, RJ, M_i, J_bar_i, cr, crdot, cp, cpdot,
                              beta_0, h, p_norm_i)

        while iterations < 10 and np.linalg.norm(correction)>0.01:


            g = calculateResidual(r_i_ddot, p_i_ddot, lagrange, lagrangeP, i, j, RJ, M_i, J_bar_i, cr, crdot, cp, cpdot,
                                  beta_0, h, p_norm_i, Fg)

            correction = np.linalg.solve(PSI , -g)


            r_i_ddot = r_i_ddot + correction[0:3,0].reshape((3,1))
            p_i_ddot = p_i_ddot + correction[3:7,0].reshape((4,1))
            lagrangeP = lagrangeP + correction[7]
            lagrange = lagrange + correction[8:13].reshape((5,1))
            iterations = iterations+1

        i.r = cr + (beta_0 ** 2) * (h ** 2) * r_i_ddot
        i.p = cp + (beta_0 ** 2) * (h ** 2) * p_i_ddot
        i.r_dot = crdot + beta_0 * h * r_i_ddot
        i.p_dot = cpdot + beta_0 * h * p_i_ddot


        x_i.append(i.r[0])
        y_i.append(i.r[1])
        z_i.append(i.r[2])

        w_global = 2 * E_from_p(i.p) @ i.p_dot
        w_x.append(w_global[0])
        w_y.append(w_global[1])
        w_z.append(w_global[2])

        vel_constraint_violation = np.concatenate((RJ.phi_r(),RJ.phi_p()),axis=1) @ np.concatenate((i.r_dot,i.p_dot),axis=0)-RJ.nu()
        norm_vel_constraint.append(np.linalg.norm(vel_constraint_violation))

        req_torque_global = -0.5 * E_from_p(i.p) @ RJ.phi_p().reshape((5, 4)).transpose() @ lagrange
        tor_x.append(req_torque_global[0])
        tor_y.append(req_torque_global[1])
        tor_z.append(req_torque_global[2])

        times.append(tt)
        # print("z:",i.r[2])

    # Plotting of Positions and Angular Velocities and Constraint Violations
    plt.figure(0)
    plt.plot(times, x_i, label='x')
    plt.plot(times, y_i, label='y')
    plt.plot(times, z_i, label='z')
    plt.title("Position of Pendulum")
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    plt.figure(1)
    plt.plot(times, w_x, label='x')
    plt.plot(times, w_y, label='y')
    plt.plot(times, w_z, label='z')
    plt.title("Angular Velocity of Pendulum")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()

    plt.figure(2)
    plt.plot(times, norm_vel_constraint)
    plt.title("Velocity Constraint Violation")
    plt.xlabel('Time (s)')
    plt.ylabel('||PHI_Q*q_dot - nu||')

    plt.figure(3)
    plt.plot(times, tor_x, label='x')
    plt.plot(times, tor_y, label='y')
    plt.plot(times, tor_z, label='z')
    plt.title("Body 1 Torque")
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()

    end = time.time()
    print("Time Elapsed: ", end - start, " seconds")

    plt.show()

def calculateResidual(r_i_ddot,p_i_ddot, lagrange, lagrangeP, i, j, RJ, M_i, J_bar_i, cr, crdot, cp, cpdot, beta_0, h, p_norm_i, Fg):
    # gets the residual aka g(x...)

    # update the body first
    i.r = cr + (beta_0 **2) * (h ** 2) * r_i_ddot
    i.p = cp + (beta_0 **2) * (h ** 2) * p_i_ddot
    i.r_dot = crdot + beta_0*h * r_i_ddot
    i.p_dot = cpdot + beta_0 * h * p_i_ddot

    # update the constraints
    RJ.update(i,j)
    p_norm_i.update(i)

    # form the g matrix (note: no generalized torques)
    g = np.zeros((13,1))
    g[0:3,0]= (M_i @ r_i_ddot + RJ.phi_r().transpose() @ lagrange- Fg).reshape((3,))

    Jp = 4 * G_from_p(i.p).transpose() @ J_bar_i @ G_from_p(i.p)
    g[3:7,0] = (Jp @ p_i_ddot + RJ.phi_p().transpose() @ lagrange + i.p.reshape((4,1)) * lagrangeP).reshape((4,))
    g[7]=(1/(beta_0 **2 * h **2)) * p_norm_i.phi()
    g[8:13,0]=((1/(beta_0 **2 * h **2)) * RJ.phi()).reshape((5,))

    return g

def computeJacobian(r_i_ddot,p_i_ddot, lagrange, lagrangeP, i, j, RJ, M_i, J_bar_i, cr, crdot, cp, cpdot, beta_0, h, p_norm_i):
    # Quasi-Newton - choosing to not compute all h^2 terms
    # update the body first
    i.r = cr + (beta_0 **2) * (h ** 2) * r_i_ddot
    i.p = cp + (beta_0 **2) * (h ** 2) * p_i_ddot
    i.r_dot = crdot + beta_0 * h * r_i_ddot
    i.p_dot = cpdot + beta_0 * h * p_i_ddot

    # update the constraints
    RJ.update(i, j)
    p_norm_i.update(i)

    Jp = 4 * G_from_p(i.p).transpose() @ J_bar_i @ G_from_p(i.p)

    # Build the jacobian
    PSI = np.zeros((13,13))
    PSI[0:3,0:3]=M_i
    PSI[0:3,8:13]=RJ.phi_r().transpose()
    PSI[3:7,3:7]=Jp
    PSI[3:7,7]=i.p.reshape((4,))
    PSI[3:7,8:13]=RJ.phi_p().transpose()
    PSI[7,3:7]=i.p.transpose()
    PSI[8:13,0:3]=RJ.phi_r()
    PSI[8:13,3:7]=RJ.phi_p()


    # return jacobian (quasi-newton)
    return PSI

if __name__ == "__main__":
    dynamicsAnalysis()