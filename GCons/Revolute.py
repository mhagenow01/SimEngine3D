""" Defines the class of related functions for the revolute joint constraint
"""

import numpy as np
from GCons.Perp1 import Perp1
from GCons.SJ import SJ


class Revolute:
    def __init__(self, i, s_bar_ip, a_bar_i, b_bar_i, j, s_bar_jq, c_bar_j, j_ground=False):
        # Constructor initializes all constraint values for the first timestep
        self.i = i
        self.s_bar_ip = s_bar_ip
        self.a_bar_i = a_bar_i
        self.b_bar_i = b_bar_i
        self.j = j
        self.s_bar_jq = s_bar_jq
        self.c_bar_j = c_bar_j
        self.j_ground = j_ground #tells whether j it the ground and doesn't bring gen. coordinates

        self.sj = SJ(self.i, self.s_bar_ip, self.j, self.s_bar_jq, j_ground=self.j_ground)
        self.perp1 = Perp1(self.i, self.a_bar_i, self.b_bar_i, self.j, self.c_bar_j, j_ground=self.j_ground)

        #Note: It is assumed that this doesn't have an associated driving function

    def update(self, i, j):
        # This function will called be at each iteration to update the relevant changing information of the constraint
        self.i = i
        self.j = j
        self.sj.update(i, j)
        self.perp1.update(i, j)

    def phi(self):
        # Returns the value of the SJ and Perp1 constraint expressions (5x1 vector)
        phi_one = self.sj.phi()
        phi_two = self.perp1.phi()
        return np.concatenate((phi_one, phi_two), axis=0).reshape((5, 1))

    def nu(self):
        # Returns the RHS of the velocity expression (5x1) always zero
        # since a perpendicular doesn't have an associated function
        nu_one = self.sj.nu()
        nu_two = self.perp1.nu()
        return np.concatenate((nu_one, nu_two), axis=0).reshape((5, 1))

    def gamma(self):
        # Returns the RHS of the acceleration expression (scalar)
        gamma_one = self.sj.gamma()
        gamma_two = self.perp1.gamma()
        return np.concatenate((gamma_one, gamma_two), axis=0).reshape((5, 1))

    def phi_r(self):
        # Returns the Jacobian of the constraint equations with respect to position. Tuple with
        # 5x3 vector for ri and 5x3 vector for rj

        phi_r_one = self.sj.phi_r()
        phi_r_two = self.perp1.phi_r()

        if self.j_ground is True:
            return np.concatenate((phi_r_one, phi_r_two), axis=0)
        else:
            #There are two entries (i and j). stack each to create two new entries
            return np.concatenate((phi_r_one[0], phi_r_two[0]), axis=0), np.concatenate((phi_r_one[1], phi_r_two[1]), axis=0)

    def phi_p(self):
        # Returns the Jacobian of the constraint equation with respect to position. Tuple with
        # 5x4 vector for pi and 5x4 vector for pj
        phi_p_one = self.sj.phi_p()
        phi_p_two = self.perp1.phi_p()

        if self.j_ground is True:
            return np.concatenate((phi_p_one, phi_p_two), axis=0).reshape((5,4))
        else:
            # There are two entries (i and j). stack each to create two new entries
            return np.concatenate((phi_p_one[0], phi_p_two[0]), axis=0).reshape((5,4)),\
                   np.concatenate((phi_p_one[1], phi_p_two[1]), axis=0).reshape((5,4))

