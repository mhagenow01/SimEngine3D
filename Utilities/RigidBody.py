""" Defines a rigid body
"""

import numpy as np


class RigidBody:
    def __init__(self, r=np.array([[0.], [0.], [0.]]), p=np.array([0., 0., 0., 1.]),r_dot=np.array([[0.], [0.], [0.]]), p_dot=np.array([0., 0., 0., 0.])):
        self.r = r
        self.p = p
        self.p_dot = p_dot
        self.r_dot = r_dot

        # Room to import needed properties for dynamics
