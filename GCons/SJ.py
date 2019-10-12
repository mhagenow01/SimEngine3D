""" Defines the class of related functions for the spherical joint constraint
"""

import numpy as np
from Utilities.kinematic_identities import A_from_p,B_from_p,a_dot_from_p_dot
from GCons.CD import CD


class SJ:
    def __init__(self, i, s_bar_ip, j, s_bar_jq,j_ground=False):
        # Constructor initializes all constraint values for the first timestep
        self.i = i
        self.s_bar_ip = s_bar_ip
        self.j = j
        self.s_bar_jq = s_bar_jq
        self.j_ground = j_ground #tells whether j it the ground and doesn't bring gen. coordinates

        self.cd_1 = CD(np.array([1., 0., 0.]).reshape((3, 1)), self.i, self.s_bar_ip,self.j,self.s_bar_jq, 0, 0, 0, j_ground=self.j_ground)
        self.cd_2 = CD(np.array([0., 1., 0.]).reshape((3, 1)), self.i, self.s_bar_ip,self.j,self.s_bar_jq, 0, 0, 0, j_ground=self.j_ground)
        self.cd_3 = CD(np.array([0., 0., 1.]).reshape((3, 1)), self.i, self.s_bar_ip,self.j,self.s_bar_jq, 0, 0, 0, j_ground=self.j_ground)

        #Note: It is assumed that this doesn't have an associated driving function

    def update(self, i, j):
        # This function will called be at each iteration to update the relevant changing information of the constraint
        self.i = i
        self.j = j
        self.cd_1.update(self.i, self.j, 0, 0, 0)
        self.cd_2.update(self.i, self.j, 0, 0, 0)
        self.cd_3.update(self.i, self.j, 0, 0, 0)

    def phi(self):
        # Returns the value of the two DP1 constraint expressions (2x1 vector)
        phi_one = self.cd_1.phi()
        phi_two = self.cd_2.phi()
        phi_three = self.cd_3.phi()
        return np.array([phi_one, phi_two, phi_three]).reshape((3, 1))

    def nu(self):
        # Returns the RHS of the velocity expression (2x1) always zero
        # since a perpendicular doesn't have an associated function
        nu_one = self.cd_1.nu()
        nu_two = self.cd_2.nu()
        nu_three = self.cd_3.nu()
        return np.array([nu_one, nu_two, nu_three]).reshape((3, 1))

    def gamma(self):
        # Returns the RHS of the acceleration expression (scalar)
        gamma_one = self.cd_1.gamma()
        gamma_two = self.cd_2.gamma()
        gamma_three = self.cd_3.gamma()
        return np.array([gamma_one, gamma_two, gamma_three]).reshape((3, 1))

    def phi_r(self):
        # Returns the Jacobian of the constraint equations with respect to position. Tuple with
        # 2x3 vector for ri and 2x3 vector for rj

        phi_r_one = self.cd_1.phi_r()
        phi_r_two = self.cd_2.phi_r()
        phi_r_three = self.cd_3.phi_r()

        if self.j_ground is True:
            return np.stack((phi_r_one, phi_r_two, phi_r_three))
        else:
            #There are two entries (i and j). stack each to create two new entries
            return np.stack((phi_r_one[0], phi_r_two[0], phi_r_three[0])), np.stack((phi_r_one[1], phi_r_two[1], phi_r_three[1]))

    def phi_p(self):
        # Returns the Jacobian of the constraint equation with respect to position. Tuple with
        # 2x4 vector for pi and 2x4 vector for pj
        phi_p_one = self.cd_1.phi_p()
        phi_p_two = self.cd_2.phi_p()
        phi_p_three = self.cd_3.phi_p()

        if self.j_ground is True:
            return np.stack((phi_p_one, phi_p_two, phi_p_three)).reshape((3,4))
        else:
            # There are two entries (i and j). stack each to create two new entries
            return np.stack((phi_p_one[0], phi_p_two[0], phi_p_three[0])).reshape((3,4)),\
                   np.stack((phi_p_one[1], phi_p_two[1], phi_p_three[1])).reshape((3,4))

