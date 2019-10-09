""" Defines the class of related functions for the coordinate difference constraint
"""

import numpy as np
from Utilities.kinematic_identities import A_from_p,B_from_p,a_dot_from_p_dot

class CD:
    def __init__(self, c, i, s_bar_ip, j, s_bar_jq, f_t, f_t_dot, f_t_ddot):
        # Constructor initializes all constraint values for the first timestep
        self.c = c
        self.i = i
        self.s_bar_ip = s_bar_ip
        self.j = j
        self.s_bar_jq = s_bar_jq
        self.f_t = f_t
        self.f_t_dot = f_t_dot
        self.f_t_ddot = f_t_ddot

    def update(self, i, j, f_t, f_t_dot, f_t_ddot):
        # This function will called be at each iteration to update the relevant changing information of the constraint
        self.i = i
        self.j = j
        self.f_t = f_t
        self.f_t_dot = f_t_dot
        self.f_t_ddot = f_t_ddot
        # Note: I will likely move the functions to be lambda expressions, but for now, i just have values for the 0,1,2
        # order derivatives

    def phi(self):
        # Returns the value of the actual constraint expression (scalar)
        return (self.c.transpose() @ (self.j.r + A_from_p(self.j.p) @ self.s_bar_jq - self.i.r -
                                     A_from_p(self.i.p) @ self.s_bar_ip) - self.f_t)[0][0]

    def nu(self):
        # Returns the RHS of the velocity expression (scalar)
        return self.f_t_dot

    def gamma(self):
        # Returns the RHS of the acceleration expression (scalar)
        return self.c.transpose() @ B_from_p(self.i.p_dot, self.s_bar_ip) @ self.i.p_dot - self.c.transpose() \
               @ B_from_p(self.j.p_dot, self.s_bar_jq) @ self.j.p_dot + self.f_t_ddot

    def phi_r(self):
        # Returns the Jacobian of the constraint equation with respect to position. Tuple with
        # 1x3 vector for ri and 1x3 vector for rj
        return (-self.c.transpose()), self.c.transpose()

    def phi_p(self):
        # Returns the Jacobian of the constraint equation with respect to position. Tuple with
        # 1x4 vector for pi and 1x4 vector for pj
        return -self.c.transpose() @ B_from_p(self.i.p, self.s_bar_ip), self.c.transpose() @ B_from_p(self.j.p, self.s_bar_jq)

