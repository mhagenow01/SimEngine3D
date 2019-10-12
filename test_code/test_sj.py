""" This function allows for the testing of the individual
expressions for the two Gcons (DP2 & D) relating two rigid bodies
for Homework 6
"""

from Utilities.RigidBody import RigidBody
from GCons.CD import CD
from GCons.SJ import SJ

import numpy as np

def main():
    # Define the rigid body location, euler parameters, and any needed vectors
    # for Rigid Body 1
    r1 = np.array([[8], [6], [-3]])
    p1 = np.array([[0.5601], [0.4200], [-0.7001], [0.1400]])
    r1_dot = np.array([[7.], [8.], [9.]])
    p1_dot = np.array([[-0.0139177], [0.090466], [0.2366], [0.9672857]])
    rb1 = RigidBody(r1, p1, r1_dot, p1_dot)
    s_1_p = np.array([[0.3], [0.25], [0.7]])

    # Define the rigid body location, euler parameters, and any needed vectors
    # for Rigid Body 2
    r2 = np.array([[-0.5], [1.6], [-6.3]])
    p2 = np.array([[0.3499959], [-0.4242], [0.5409], [0.6363]])
    r2_dot = np.array([[11.], [12.], [13.]])
    p2_dot = np.array([[0.0629245], [-0.388034], [0.534858], [-0.747927]])
    rb2 = RigidBody(r2, p2, r2_dot, p2_dot)
    s_2_q = np.array([[1.5], [0.4], [0.8]])

    c1 = np.array([1,0,0]).reshape((3,1))
    c2 = np.array([0,1,0]).reshape((3,1))
    c3 = np.array([0,0,1]).reshape((3,1))

    # Define the actual constraints
    # NOTE: The values of the function and its derivatives are all zero for now
    cd_1 = CD(c1, rb1, s_1_p, rb2, s_2_q, 0, 0, 0, j_ground=False)
    cd_2 = CD(c2, rb1, s_1_p, rb2, s_2_q, 0, 0, 0, j_ground=False)
    cd_3 = CD(c3, rb1, s_1_p, rb2, s_2_q, 0, 0, 0, j_ground=False)

    sj = SJ(rb1,s_1_p,rb2,s_2_q,j_ground=False)


    print("####### CD-1 values: #########")
    print("PHI:")
    print(cd_1.phi())
    print("Nu:")
    print(cd_1.nu())
    print("Gamma:")
    print(cd_1.gamma())
    print("PHI_r:")
    print(cd_1.phi_r())
    print("PHI_p:")
    print(cd_1.phi_p())

    print("####### CD-2 values: #########")
    print("PHI:")
    print(cd_2.phi())
    print("Nu:")
    print(cd_2.nu())
    print("Gamma:")
    print(cd_2.gamma())
    print("PHI_r:")
    print(cd_2.phi_r())
    print(np.shape(cd_2.phi_r()[0]))
    print("PHI_p:")
    print(cd_2.phi_p())
    print(np.shape(cd_2.phi_p()[0]))

    print("####### CD-3 values: #########")
    print("PHI:")
    print(cd_3.phi())
    print("Nu:")
    print(cd_3.nu())
    print("Gamma:")
    print(cd_3.gamma())
    print("PHI_r:")
    print(cd_3.phi_r())
    print("PHI_p:")
    print(cd_3.phi_p())

    print("####### SJ values: #########")
    print("PHI:")
    print(sj.phi())
    print("Nu:")
    print(sj.nu())
    print("Gamma:")
    print(sj.gamma())
    print("PHI_r:")
    print(sj.phi_r())
    print(np.shape(sj.phi_r()[0]))
    print("PHI_p:")
    print(sj.phi_p())
    print(np.shape(sj.phi_p()[0]))

