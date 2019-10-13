""" Defines the class of related functions for the perpendicular one constraint
"""

import numpy as np
from Utilities.kinematic_identities import A_from_p,B_from_p,a_dot_from_p_dot
from GCons.DP1 import DP1


class Perp1:
    def __init__(self, i, a_bar_i, b_bar_i, j, c_bar_j,j_ground=False):
        # Constructor initializes all constraint values for the first timestep
        self.i = i
        self.a_bar_i = a_bar_i
        self.b_bar_i = b_bar_i
        self.j = j
        self.c_bar_j = c_bar_j
        self.j_ground=j_ground #tells whether j it the ground and doesn't bring gen. coordinates

        self.dp1_1 = DP1(self.i, self.a_bar_i, self.j, self.c_bar_j, 0, 0, 0, j_ground=self.j_ground)
        self.dp1_2 = DP1(self.i, self.b_bar_i, self.j, self.c_bar_j, 0, 0, 0, j_ground=self.j_ground)
        #Note: It is assumed that this doesn't have an associated driving function

    def update(self, i, j):
        # This function will called be at each iteration to update the relevant changing information of the constraint
        self.i = i
        self.j = j
        self.dp1_1.update(self.i,self.j, 0, 0, 0)
        self.dp1_2.update(self.i,self.j, 0, 0, 0)

    def phi(self):
        # Returns the value of the two DP1 constraint expressions (2x1 vector)
        phi_one = self.dp1_1.phi()
        phi_two = self.dp1_2.phi()
        return np.array([phi_one, phi_two]).reshape((2, 1))

    def nu(self):
        # Returns the RHS of the velocity expression (2x1) always zero
        # since a perpendicular doesn't have an associated function
        nu_one = self.dp1_1.nu()
        nu_two = self.dp1_2.nu()
        return np.array([nu_one, nu_two]).reshape((2, 1))

    def gamma(self):
        # Returns the RHS of the acceleration expression (scalar)
        gamma_one = self.dp1_1.gamma()
        gamma_two = self.dp1_2.gamma()
        return np.array([gamma_one, gamma_two]).reshape((2, 1))

    def phi_r(self):
        # Returns the Jacobian of the constraint equations with respect to position. Tuple with
        # 2x3 vector for ri and 2x3 vector for rj

        phi_r_one = self.dp1_1.phi_r()
        phi_r_two = self.dp1_2.phi_r()

        if self.j_ground is True:
            return np.stack((phi_r_one, phi_r_two)).reshape((2,3))
        else:
            #There are two entries (i and j). stack each to create two new entries
            return np.stack((phi_r_one[0], phi_r_two[0])).reshape((2,3)), np.stack((phi_r_one[1], phi_r_two[1])).reshape((2,3))

    def phi_p(self):
        # Returns the Jacobian of the constraint equation with respect to position. Tuple with
        # 2x4 vector for pi and 2x4 vector for pj
        phi_p_one = self.dp1_1.phi_p()
        phi_p_two = self.dp1_2.phi_p()

        if self.j_ground is True:
            return np.stack((phi_p_one, phi_p_two)).reshape((2,4))
        else:
            # There are two entries (i and j). stack each to create two new entries
            return np.stack((phi_p_one[0], phi_p_two[0])).reshape((2,4)), np.stack((phi_p_one[1], phi_p_two[1])).reshape((2,4))

