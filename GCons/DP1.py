""" Defines the class of related functions for the dot product one constraint
"""

import numpy as np
from Utilities.kinematic_identities import A_from_p

class DP1:
    def __init__(self,i,a_bar_i,j,a_bar_j,f_t):
        self.i = i
        self.a_bar_i=a_bar_i
        self.j=j
        self.a_bar_j = a_bar_j
        self.f_t=f_t

    def phi(self,i,a_bar_i,j,a_bar_j,f_t):
        return a_bar_i.transpose() @ A_from_p(i.p).transpose() @ A_from_p(j.p) @ a_bar_j -f_t
    def nu:

    def gamma:

    def phi_r:

    def phi_p:
