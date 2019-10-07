""" Kinematic identities
"""

import numpy as np

def A_from_p(p):
    e0=p[0]
    e=np.array(p[1:]).reshape((3, 1))
    return (2*e0-1)*np.eye(3)+2*(np.matmul(e, e.reshape((1, 3)))+e0*a_tilde(e))

def a_tilde(a):
    return np.array([0.,-a[2], a[1]],
                    [a[2], 0., -a[0]],
                    [-a[1], a[0], 0.]);

def B_from_p(p,s_bar):
    e0 = p[0]
    e = np.array(p[1:]).reshape((3, 1))
    B=np.zeros((3,4))
    B[:,0]=2*(e0*np.eye(3)+a_tilde(e)) @ s_bar
    B[:,1:]= e @ s_bar.transpose() - (e0*np.eye(3)+a_tilde(e)) @ a_tilde(s_bar)
    return B