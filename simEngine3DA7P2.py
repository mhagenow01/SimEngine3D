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

def solveIVP():
    ###### Backwards Euler ################

    # solve a linear system for delta
    # update, until updates are small enough

    # need function g and the jacobian in order to do this...





if __name__ == "__main__":
    solveIVP()