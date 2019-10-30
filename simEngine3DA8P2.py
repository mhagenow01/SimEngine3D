""" Peforms a dynamics analysis for a
revolute joint double pendulum
"""

import numpy as np
from Utilities.RigidBody import RigidBody
from Utilities.kinematic_identities import p_from_A, A_from_p, a_dot_from_p_dot, a_ddot, G_from_p, E_from_p
from GCons.Revolute import Revolute
from GCons.DP1 import DP1
from GCons.P_norm import P_norm
import matplotlib.pyplot as plt

def dynamicsAnalysis():
    ###### SIMULATION PARAMETERS #################
    sim_length = 10.
    h = 0.001 # step for solver

    ###### Define the three bodies ################
    # Body j is going to be the ground and as such doesn't have any generalized coordinates
    j = RigidBody()  # defaults are all zero
    s_bar_j_q = np.array([0., 0., 0.]).reshape((3,1))

    ##### Body i - First Pendulum ########################
    # Initial configuration for body i - 90 degree theta
    r_i = np.array([0., 2, 0]).reshape((3,1))
    r_dot_i = np.array([0., 0., 0.]).reshape((3,1))  # no initial velocity

    # Need to convert A TO P for use in this formulation
    A_i_initial = np.array([[0., 0., 1],[1, 0., 0.],[0., 1., 0.]])
    p_i_initial = p_from_A(A_i_initial)
    p_dot_i = np.array([0., 0., 0., 0.]).reshape((4,1)) # orientation is not initially changing

    # mass matrix for body i
    m_i = np.power(0.05, 2) * 4 * 7800
    M_i = m_i * np.eye(3)

    # Inertia matrix
    J_bar_i = np.zeros((3, 3))
    J_bar_i[0, 0] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(0.05, 2))
    J_bar_i[1, 1] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(4, 2))
    J_bar_i[2, 2] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(4, 2))

    i = RigidBody(r_i,p_i_initial,r_dot_i,p_dot_i,m_i)

    ##### Body k - Second Pendulum ########################
    # Initial configuration for body k - hanging down
    r_k = np.array([0., 4, -1]).reshape((3, 1))
    r_dot_k = np.array([0., 0., 0.]).reshape((3, 1))  # no initial velocity

    # Need to convert A TO P for use in this formulation
    A_k_initial = np.array([[0., 0., 1], [0, 1., 0.], [-1., 0., 0.]])
    p_k_initial = p_from_A(A_k_initial)
    p_dot_k = np.array([0., 0., 0., 0.]).reshape((4, 1))  # orientation is not initially changing

    # mass matrix for body i - assuming same cross section
    m_k = np.power(0.05, 2) * 2 * 7800  # half the length of body i
    M_k = m_k * np.eye(3)

    # Inertia matrix
    J_bar_k = np.zeros((3, 3))
    J_bar_k[0, 0] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(0.05, 2))
    J_bar_k[1, 1] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(2, 2))
    J_bar_k[2, 2] = (1 / 12) * m_i * (np.power(0.05, 2) + np.power(2, 2))

    k = RigidBody(r_k, p_k_initial, r_dot_k, p_dot_k, m_k)  # velocities as defaults

    ###### Define the geometric constraints ################
    #
    # 1. Revolute joint that holds body 1 (i) to global frame (j)
    # 2. Revolute joint that holds body 2 (k) to body 1 (i)

    # Revolute joint holding to global frame
    s_bar_i_q = np.array([-2., 0., 0.]).reshape((3,1))

    # Define the local vectors of the revolute joint
    c_bar_j = np.array([1., 0., 0.]).reshape((3,1))
    a_bar_i = np.array([1., 0., 0.]).reshape((3,1))
    b_bar_i = np.array([0., 1., 0.]).reshape((3,1))

    RJ1 = Revolute(i, s_bar_i_q, a_bar_i, b_bar_i, j, s_bar_j_q, c_bar_j, j_ground=True)
    RJ1.update(i,j)

    # Revolute joint holding body 2 to body 1
    s_bar_k_p = np.array([-1., 0., 0.]).reshape((3, 1))

    # Define the local vectors of the revolute joint
    c_bar_i = np.array([0., 0., 1.]).reshape((3, 1))
    a_bar_k = np.array([1., 0., 0.]).reshape((3, 1))
    b_bar_k = np.array([0., 1., 0.]).reshape((3, 1))

    s_bar_i_p = np.array([2., 0., 0.]).reshape((3, 1))

    RJ2 = Revolute(k, s_bar_k_p, a_bar_k, b_bar_k, i, s_bar_i_p, c_bar_i, j_ground=False)
    RJ2.update(k, i)

    # Normalization constraints are needed for body i and body k
    p_norm_i = P_norm(i)
    p_norm_k = P_norm(k)

    # Overall mass matrix has body i and body k
    # and is constant
    M_total = np.zeros((6,6))
    M_total[:3,:3]=M_i
    M_total[3:6,3:6]=M_k


    # Compute the initial conditions for acceleration and the lagrange multipliers
    # by solving a linear system


    LHS = np.zeros((26,26))
    LHS[0:6,0:6]=M_total

    # Calculate the combined PHI_partial terms
    phi_r = np.zeros((10,6))
    phi_r[0:5,0:3]=RJ1.phi_r()
    phi_r[5:10,0:3]=RJ2.phi_r()[1]
    phi_r[5:10,3:6]=RJ2.phi_r()[0]

    phi_p = np.zeros((10,8))
    phi_p[0:5,0:4]=RJ1.phi_p()
    phi_p[5:10,0:4]=RJ2.phi_p()[1]
    phi_p[5:10,4:8]=RJ2.phi_p()[0]

    # Calculate P for P/P^T
    P = np.zeros((2,8))
    P[0:1,0:4]=i.p.transpose()
    P[1:2,4:8]=k.p.transpose()

    LHS[0:6,16:26]=phi_r.transpose()

    # Calculate the combined inertia matrix
    Jpi = 4 * G_from_p(i.p).transpose() @ J_bar_i @ G_from_p(i.p)
    Jpk = 4 * G_from_p(k.p).transpose() @ J_bar_k @ G_from_p(k.p)
    Jp = np.zeros((8,8))
    Jp[0:4,0:4]=Jpi
    Jp[4:8,4:8]=Jpk

    LHS[6:14,6:14] = Jp
    LHS[6:14,14:16] = P.transpose()
    LHS[6:14,16:26] = phi_p.transpose()

    LHS[14:16,6:14]=P

    LHS[16:26,0:6]=phi_r
    LHS[16:26,6:14]=phi_p

    # Calculate the two gravitational forces
    Fgi = np.array([0., 0.,-9.81 * m_i]).reshape((3,1))
    Fgk = np.array([0., 0.,-9.81 * m_k]).reshape((3,1))
    Fg = np.concatenate((Fgi,Fgk),axis=0)

    RHS = np.zeros((26,1))
    RHS[0:6,0] = Fg.reshape((6,))

    RHS[6:14,0] = np.zeros((8,)) # no torques

    RHS[14,0] = p_norm_i.gamma()
    RHS[15,0] = p_norm_k.gamma()
    RHS[16:21,0]=RJ1.gamma().reshape((5,))
    RHS[21:26,0]=RJ2.gamma().reshape((5,))

    initial_conds = np.linalg.solve(LHS,RHS)

    # Lists for plotting
    x_i = []
    y_i = []
    z_i = []
    x_k = []
    y_k = []
    z_k = []
    w_x_i = []
    w_y_i = []
    w_z_i = []
    w_x_k = []
    w_y_k = []
    w_z_k = []
    norm_vel_constraint = []
    times = []

    r_i_ddot = initial_conds[0:3, 0].reshape((3, 1))
    r_k_ddot = initial_conds[3:6, 0].reshape((3, 1))
    p_i_ddot = initial_conds[6:10, 0].reshape((4, 1))
    p_k_ddot = initial_conds[10:14, 0].reshape((4, 1))
    lagrangeP = initial_conds[14:16, 0]
    lagrange = initial_conds[16:26, 0].reshape((10, 1))


    printIter = 0
    for tt in np.arange(h,sim_length,h):
        if (printIter % 100) == 0:
            print(tt)
            # print("p:" , i.p.transpose(), " t:",tt)
        printIter = printIter+1

        # Step 0  - Prime the solver
        # Don't actually need to do anything in this implementation


        # First order BDF needs previous timesteps
        r_i_prev = i.r
        r_k_prev = k.r
        r_dot_i_prev = i.r_dot
        r_dot_k_prev = k.r_dot
        p_i_prev = i.p
        p_k_prev = k.p
        p_dot_i_prev = i.p_dot
        p_dot_k_prev = k.p_dot


        # Use BDF of order 1 - compute static terms
        cr_i = r_i_prev + h*r_dot_i_prev
        crdot_i = r_dot_i_prev
        cp_i = p_i_prev + h * p_dot_i_prev
        cpdot_i = p_dot_i_prev
        cr_k = r_k_prev + h*r_dot_k_prev
        crdot_k = r_dot_k_prev
        cp_k = p_k_prev + h * p_dot_k_prev
        cpdot_k = p_dot_k_prev
        beta_0 = 1


        # Compute non-linear residual
        iterations = 0
        correction = np.ones((26,1)) # seed so first iteration runs

        # Calculate the jacobian for the iterative process (only done once per timestep)
        PSI = computeJacobian(r_i_ddot, r_k_ddot, p_i_ddot, p_k_ddot, lagrange, lagrangeP, i, j, k, RJ1, RJ2, M_total,
                              J_bar_i, J_bar_k, cr_i, crdot_i, cp_i, cpdot_i,
                              cr_k, crdot_k, cp_k, cpdot_k, beta_0, h, p_norm_i, p_norm_k)

        while iterations < 10 and np.linalg.norm(correction) > 0.01:

            # Need to fix
            g = calculateResidual(r_i_ddot,r_k_ddot,p_i_ddot, p_k_ddot, lagrange, lagrangeP, i, j, k, RJ1, RJ2, M_total, J_bar_i,J_bar_k,
                                  cr_i, crdot_i, cp_i, cpdot_i,cr_k, crdot_k, cp_k, cpdot_k,beta_0, h, p_norm_i, p_norm_k, Fg)

            correction = np.linalg.solve(PSI , -g)


            r_i_ddot = r_i_ddot + correction[0:3,0].reshape((3,1))
            r_k_ddot = r_k_ddot + correction[3:6,0].reshape((3,1))
            p_i_ddot = p_i_ddot + correction[6:10,0].reshape((4,1))
            p_k_ddot = p_k_ddot + correction[10:14,0].reshape((4,1))
            lagrangeP = lagrangeP + correction[14:16,0]
            lagrange = lagrange + correction[16:26].reshape((10,1))
            iterations = iterations+1

        i.r = cr_i + (beta_0 ** 2) * (h ** 2) * r_i_ddot
        i.p = cp_i + (beta_0 ** 2) * (h ** 2) * p_i_ddot
        i.r_dot = crdot_i + beta_0 * h * r_i_ddot
        i.p_dot = cpdot_i + beta_0 * h * p_i_ddot
        k.r = cr_k + (beta_0 ** 2) * (h ** 2) * r_k_ddot
        k.p = cp_k + (beta_0 ** 2) * (h ** 2) * p_k_ddot
        k.r_dot = crdot_k + beta_0 * h * r_k_ddot
        k.p_dot = cpdot_k + beta_0 * h * p_k_ddot

        x_i.append(i.r[0])
        y_i.append(i.r[1])
        z_i.append(i.r[2])
        x_k.append(k.r[0])
        y_k.append(k.r[1])
        z_k.append(k.r[2])

        w_global_i = 2 * E_from_p(i.p) @ i.p_dot
        w_x_i.append(w_global_i[0])
        w_y_i.append(w_global_i[1])
        w_z_i.append(w_global_i[2])

        w_global_k = 2 * E_from_p(k.p) @ k.p_dot
        w_x_k.append(w_global_k[0])
        w_y_k.append(w_global_k[1])
        w_z_k.append(w_global_k[2])

        vel_constraint_violation = np.concatenate((RJ2.phi_r()[1],RJ2.phi_r()[0], RJ2.phi_p()[1],RJ2.phi_p()[0]), axis=1) @\
                                   np.concatenate((i.r_dot,k.r_dot,i.p_dot,k.p_dot),axis=0) - RJ2.nu()
        norm_vel_constraint.append(np.linalg.norm(vel_constraint_violation))

        times.append(tt)

    # Do some plotting
    plt.figure(0)
    plt.plot(times, x_i, label='x')
    plt.plot(times, y_i, label='y')
    plt.plot(times, z_i, label='z')
    plt.title("Positions")
    plt.xlabel('Time (s)')
    plt.ylabel('Position of Body 1 (m)')
    plt.legend()

    plt.figure(1)
    plt.plot(times, x_k, label='x')
    plt.plot(times, y_k, label='y')
    plt.plot(times, z_k, label='z')
    plt.title("Positions")
    plt.xlabel('Time (s)')
    plt.ylabel('Position of Body 2 (m)')
    plt.legend()

    plt.figure(2)
    plt.plot(times, w_x_i, label='x')
    plt.plot(times, w_y_i, label='y')
    plt.plot(times, w_z_i, label='z')
    plt.title("Angular Velocity of Body 1")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()

    plt.figure(3)
    plt.plot(times, w_x_k, label='x')
    plt.plot(times, w_y_k, label='y')
    plt.plot(times, w_z_k, label='z')
    plt.title("Angular Velocity of Body 2")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()

    plt.figure(2)
    plt.plot(times, norm_vel_constraint)
    plt.title("Velocity Constraint Violation")
    plt.xlabel('Time (s)')
    plt.ylabel('||PHI_Q*q_dot - nu||')

    plt.show()

def calculateResidual(r_i_ddot,r_k_ddot,p_i_ddot, p_k_ddot, lagrange, lagrangeP, i, j, k, RJ1, RJ2, M_total, J_bar_i,J_bar_k,
                                  cr_i, crdot_i, cp_i, cpdot_i,cr_k, crdot_k, cp_k, cpdot_k,beta_0, h, p_norm_i, p_norm_k, Fg):
    # gets the residual aka g(x...)

    # update the body first
    i.r = cr_i + beta_0 * (h ** 2) * r_i_ddot
    i.p = cp_i + beta_0 * (h ** 2) * p_i_ddot
    i.r_dot = crdot_i + beta_0 * h * r_i_ddot
    i.p_dot = cpdot_i + beta_0 * h * p_i_ddot
    k.r = cr_k + beta_0 * (h ** 2) * r_k_ddot
    k.p = cp_k + beta_0 * (h ** 2) * p_k_ddot
    k.r_dot = crdot_k + beta_0 * h * r_k_ddot
    k.p_dot = cpdot_k + beta_0 * h * p_k_ddot

    # update the constraints
    RJ1.update(i, j)
    RJ2.update(k, i)
    p_norm_i.update(i)
    p_norm_k.update(k)

    # Calculate the combined PHI_partial terms
    phi_r = np.zeros((10, 6))
    phi_r[0:5, 0:3] = RJ1.phi_r()
    phi_r[5:10, 0:3] = RJ2.phi_r()[1]
    phi_r[5:10, 3:6] = RJ2.phi_r()[0]

    phi_p = np.zeros((10, 8))
    phi_p[0:5, 0:4] = RJ1.phi_p()
    phi_p[5:10, 0:4] = RJ2.phi_p()[1]
    phi_p[5:10, 4:8] = RJ2.phi_p()[0]

    # Calculate P for P/P^T
    P = np.zeros((2, 8))
    P[0:1, 0:4] = i.p.transpose()
    P[1:2, 4:8] = k.p.transpose()

    # form the g matrix (note: no generalized torques)
    g = np.zeros((26,1))
    g[0:6,0]= (M_total @ np.concatenate((r_i_ddot,r_k_ddot),axis=0) + phi_r.transpose() @ lagrange-Fg).reshape((6,))

    Jpi = 4 * G_from_p(i.p).transpose() @ J_bar_i @ G_from_p(i.p)
    Jpk = 4 * G_from_p(k.p).transpose() @ J_bar_k @ G_from_p(k.p)
    Jp = np.zeros((8, 8))
    Jp[0:4, 0:4] = Jpi
    Jp[4:8, 4:8] = Jpk

    g[6:14,0] = (Jp @ np.concatenate((p_i_ddot,p_k_ddot),axis=0) + phi_p.transpose() @ lagrange + P.transpose() @ lagrangeP.reshape((2,1))).reshape((8,))
    g[14:16,0]=((1/(beta_0 **2 * h **2)) * np.concatenate((p_norm_i.phi(),p_norm_k.phi()),axis=0)).reshape((2,))

    g[16:26,0]=((1/(beta_0 **2 * h **2)) * np.concatenate((RJ1.phi(),RJ2.phi()),axis=0)).reshape((10,))
    return g

def computeJacobian(r_i_ddot,r_k_ddot, p_i_ddot,p_k_ddot, lagrange, lagrangeP, i, j,k, RJ1,RJ2, M_total, J_bar_i,J_bar_k, cr_i, crdot_i, cp_i, cpdot_i,
                                  cr_k,crdot_k,cp_k,cpdot_k,beta_0, h, p_norm_i,p_norm_k):
    # Quasi-Newton - choosing to not compute all h^2 terms
    # update the body first
    i.r = cr_i + beta_0 * (h ** 2) * r_i_ddot
    i.p = cp_i + beta_0 * (h ** 2) * p_i_ddot
    i.r_dot = crdot_i + beta_0 * h * r_i_ddot
    i.p_dot = cpdot_i + beta_0 * h * p_i_ddot
    k.r = cr_k + beta_0 * (h ** 2) * r_k_ddot
    k.p = cp_k + beta_0 * (h ** 2) * p_k_ddot
    k.r_dot = crdot_k + beta_0 * h * r_k_ddot
    k.p_dot = cpdot_k + beta_0 * h * p_k_ddot

    # update the constraints
    RJ1.update(i, j)
    RJ2.update(k, i)
    p_norm_i.update(i)
    p_norm_k.update(k)

    PSI = np.zeros((26, 26))
    PSI[0:6, 0:6] = M_total

    # Calculate the combined PHI_partial terms
    phi_r = np.zeros((10, 6))
    phi_r[0:5, 0:3] = RJ1.phi_r()
    phi_r[5:10, 0:3] = RJ2.phi_r()[1]
    phi_r[5:10, 3:6] = RJ2.phi_r()[0]

    phi_p = np.zeros((10, 8))
    phi_p[0:5, 0:4] = RJ1.phi_p()
    phi_p[5:10, 0:4] = RJ2.phi_p()[1]
    phi_p[5:10, 4:8] = RJ2.phi_p()[0]

    # Calculate P for P/P^T
    P = np.zeros((2, 8))
    P[0:1, 0:4] = i.p.transpose()
    P[1:2, 4:8] = k.p.transpose()

    PSI[0:6, 16:26] = phi_r.transpose()
    Jpi = 4 * G_from_p(i.p).transpose() @ J_bar_i @ G_from_p(i.p)
    Jpk = 4 * G_from_p(k.p).transpose() @ J_bar_k @ G_from_p(k.p)
    Jp = np.zeros((8, 8))
    Jp[0:4, 0:4] = Jpi
    Jp[4:8, 4:8] = Jpk

    PSI[6:14, 6:14] = Jp
    PSI[6:14, 14:16] = P.transpose()
    PSI[6:14, 16:26] = phi_p.transpose()

    PSI[14:16, 6:14] = P

    PSI[16:26, 0:6] = phi_r
    PSI[16:26, 6:14] = phi_p
    # return jacobian (quasi-newton)
    return PSI

if __name__ == "__main__":
    dynamicsAnalysis()