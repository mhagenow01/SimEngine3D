""" This function allows for the testing of the individual
expressions for the two Gcons (DP2 & D) relating two rigid bodies
for Homework 6
"""

from Utilities.RigidBody import RigidBody
from GCons.DP2 import DP2
from GCons.D import D
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

    # Define the actual constraints
    # NOTE: The values of the function and its derivatives are all zero for now
    dp2 = DP2(rb1, a_rb1, s_1_p, rb2, s_2_q, 0, 0, 0, j_ground=False)
    d = D(rb1, s_1_p, rb2, s_2_q, 0, 0, 0, j_ground=False)

    print("####### DP2 values: #########")
    print("PHI:")
    print(dp2.phi())
    print("Nu:")
    print(dp2.nu())
    print("Gamma:")
    print(dp2.gamma())
    print("PHI_r:")
    print(dp2.phi_r())
    print("PHI_p:")
    print(dp2.phi_p())

    print("####### D values: #########")
    print("PHI:")
    print(d.phi())
    print("Nu:")
    print(d.nu())
    print("Gamma:")
    print(d.gamma())
    print("PHI_r:")
    print(d.phi_r())
    print("PHI_p:")
    print(d.phi_p())

