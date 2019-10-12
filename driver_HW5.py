""" This function allows for the testing of the individual
expressions for the two Gcons relating two rigid bodies
"""

from Utilities.RigidBody import RigidBody
from GCons.DP1 import DP1
from GCons.CD import CD
import numpy as np

def main():
    # Define the rigid body location, euler parameters, and any needed vectors
    # for Rigid Body 1
    r1=np.array([[0.2], [0.3], [0.4]])
    p1=np.array([[1.], [0.], [0.], [0.]])
    r1_dot = np.array([[0.], [0.], [0.]])
    p1_dot = np.array([[1.], [0.], [0.], [0.]])
    rb1 = RigidBody(r1, p1, r1_dot, p1_dot)
    a_rb1 = np.array([[1.0], [0], [0.0]])
    s_1_p = np.array([[0.0], [0], [0.0]])

    # Define the rigid body location, euler parameters, and any needed vectors
    # for Rigid Body 2
    r2 = np.array([[0.], [0.], [0.]])
    p2 = np.array([[0.975104], [0.0991687], [0.1983375], [0.]])
    r2_dot = np.array([[0.], [0.], [0.]])
    p2_dot = np.array([[1.], [0.], [0.], [0.]])
    rb2 = RigidBody(r2, p2, r2_dot, p2_dot)
    a_rb2 = np.array([[1.], [0.0], [0.0]])
    s_2_q = np.array([[0.0], [0], [0.0]])

    # Define the driving functions
    c = np.array([[0], [1], [0]])

    # Define the actual constraints
    # NOTE: The values of the function and its derivatives are all zero for now
    dp1 = DP1(rb1, a_rb1, rb2, a_rb2, 0, 0, 0, j_ground=False)
    cd = CD(c, rb1, s_1_p, rb2, s_2_q, 0, 0, 0, j_ground=False)

    print("####### DP1 values: #########")
    print("PHI:")
    print(dp1.phi())
    print("Nu:")
    print(dp1.nu())
    print("Gamma:")
    print(dp1.gamma())
    print("PHI_r:")
    print(dp1.phi_r())
    print("PHI_p:")
    print(dp1.phi_p())

    print("####### CD values: #########")
    print("PHI:")
    print(cd.phi())
    print("Nu:")
    print(cd.nu())
    print("Gamma:")
    print(cd.gamma())
    print("PHI_r:")
    print(cd.phi_r())
    print("PHI_p:")
    print(cd.phi_p())

