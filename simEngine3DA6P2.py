""" Function to set up a revolute joint pendulum
 with the parameters for HW #6
"""

import numpy as np
from Utilities.RigidBody import RigidBody
from Utilities.kinematic_identities import p_from_A
from GCons.Revolute import Revolute
from GCons.DP1 import DP1

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
    t = 0.
    f = np.sin(np.pi / 4 * np.cos(2 * t))
    f_dot = -np.cos(np.pi/4 * np.cos(2*t)) * np.pi/2 * np.sin(2*t)
    f_ddot = np.sin(np.pi/4 * np.cos(2*t)) * np.power(np.pi,2)/4 * np.power(np.sin(2*t),2)- \
             -np.cos(np.pi / 4 * np.cos(2 * t)) * np.pi * np.cos(2*t)
    dp1 = DP1(i, a_bar_i, j, d_bar_j, f, f_dot, f_ddot, j_ground=True)

    ###### Print the Desired Initial Properties ########################
    PHI = np.concatenate((RJ.phi(), dp1.phi().reshape((1,1))),axis=0)
    print("PHI:")
    print(PHI)

    print ()
    phi_q_rj = np.concatenate((RJ.phi_r(),RJ.phi_p()),axis=1)
    phi_q_dp1 = np.concatenate((dp1.phi_r().reshape((1,3)),dp1.phi_p()),axis=1)
    PHI_Q = np.concatenate((phi_q_rj,phi_q_dp1),axis=0)
    print("PHI_Q")
    print(PHI_Q)

    NU = np.concatenate((RJ.nu(), dp1.nu().reshape((1,1))), axis=0)
    print("NU:")
    print(NU)

    GAMMA = np.concatenate((RJ.gamma(), dp1.gamma()), axis=0)
    print("GAMMA:")
    print(GAMMA)


if __name__ == "__main__":
    setUpPendulum()