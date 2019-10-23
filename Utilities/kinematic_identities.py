""" Kinematic identities
"""

import numpy as np

def A_from_p(p):
    e0 = p[0]
    e = np.array(p[1:]).reshape((3, 1))
    return (np.power(e0, 2) - (e.reshape((1, 3)) @ e))*np.eye(3)+2*(e @ e.reshape((1, 3)))+2*e0*a_tilde(e)

def p_from_A(A):
    # if the trace of A doesn't equal -1 (FIX!!!!!)
    A.reshape((3, 3))
    trA = np.trace(A)
    e0 = np.sqrt((trA+1)/4) # positive e0

    # Now get the signs
    e1 = np.sqrt((1+2*A[0][0]-trA)/4)*np.sign(A[2][1]-A[1][2])
    e2 = np.sqrt((1+2*A[1][1]-trA)/4)*np.sign(A[0][2]-A[2][0])
    e3 = np.sqrt((1+2*A[2][2]-trA)/4)*np.sign(A[1][0]-A[0][1])
    return np.array([e0, e1, e2, e3]).reshape((4, 1))


def a_tilde(a):
    a = a.reshape((3, ))
    return np.array([[0., -a[2], a[1]],
                    [a[2], 0., -a[0]],
                    [-a[1], a[0], 0.]])


def B_from_p(p,s_bar):
    e0 = p[0]
    e = np.array(p[1:]).reshape((3, 1))
    B = np.zeros((3, 4))
    B[:, 0] = (2*(e0*np.eye(3)+a_tilde(e)) @ s_bar).reshape((3,))
    B[:, 1:] = 2*(e @ s_bar.transpose() - (e0*np.eye(3)+a_tilde(e)) @ a_tilde(s_bar))
    return B


def a_dot_from_p_dot(p, s_bar, p_dot):
    return B_from_p(p, s_bar) @ p_dot

def a_ddot(p,s_bar, p_dot, p_ddot):
    return B_from_p(p_dot,s_bar) @ p_dot + B_from_p(p,s_bar) @ p_ddot

def d_from_vecs(i,s_bar_ip,j,s_bar_jq):
    return j.r + A_from_p(j.p) @ s_bar_jq - i.r - A_from_p(i.p) @ s_bar_ip

def d_dot_from_vecs(i,s_bar_ip,j,s_bar_jq):
    return j.r_dot + B_from_p(j.p, s_bar_jq) @ j.p_dot - i.r_dot - B_from_p(i.p, s_bar_ip) @ i.p_dot

def G_from_p(p):
    p.reshape((4,1))
    e0 = p[0]
    e = np.array(p[1:]).reshape((3, 1))

    G = np.zeros((3, 4))
    G[:, 0] = -e.reshape((3,))
    G[:, 1:] = -a_tilde(e)+e0*np.eye(3)

    return G

