""" Defines a rigid body
"""

import numpy as np


class RigidBody:
    def __init__(self, r=np.array([[0.], [0.], [0.]]), p=np.array([0., 0., 0., 1.]),r_dot=np.array([[0.], [0.], [0.]]),
                 p_dot=np.array([0., 0., 0., 0.]), m=0, J_bar=np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]).reshape((3,3))):
        self.r = r
        self.p = p
        self.p_dot = p_dot
        self.r_dot = r_dot

        # Rigid Body properties for dynamics
        self.m = m  # mass
        self.M = self.m * np.eye(3)  # mass matrix
        self.J_bar = J_bar  # inertia matrix
