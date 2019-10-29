""" Defines the class of related functions for the coordinate difference constraint
"""

import numpy as np
from Utilities.kinematic_identities import A_from_p,B_from_p,a_dot_from_p_dot

class P_norm:
    def __init__(self, i):
        # Constructor initializes all constraint values for the first timestep
        self.i = i

    def update(self, i):
        # This function will called be at each iteration to update the relevant changing information of the constraint
        self.i = i

    def phi(self):
        # Returns the value of the actual constraint expression (scalar)
        return (0.5*self.i.p.reshape((1, 4)) @ self.i.p.reshape((4, 1)) - 0.5).reshape((1, 1))

    def nu(self):
        # Returns the RHS of the velocity expression (scalar)
        return np.array(0.).reshape((1,1))

    def gamma(self):
        # Returns the RHS of the acceleration expression (scalar)
        return (-2*self.i.p_dot.reshape((1, 4)) @ self.i.p_dot.reshape((4, 1))).reshape((1, 1))

    def phi_r(self):
        # Returns the Jacobian of the constraint equation with respect to position. Tuple with
        # 1x3 vector for ri and 1x3 vector for rj
        return np.zeros((3,)).reshape((1,3))

    def phi_p(self):
        # Returns the Jacobian of the constraint equation with respect to position. Tuple with
        # 1x4 vector for pi and 1x4 vector for pj
        return self.i.p.reshape((1,4))

