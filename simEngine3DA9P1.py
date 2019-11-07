""" Peforms a steady-state finite difference
method for a 1D problem
"""

import numpy as np
import matplotlib.pyplot as plt

def finiteDifference():

    # Diffusion, Velocity Field, and Rate of Generation/Dissipation are constant
    ux = 1
    D = 1
    s = 1
    dphidx = 0

    # Define the discretized points, x
    n = 20  # number of equidistant elements
    domain_start = 0
    domain_end = 1
    x = np.linspace(domain_start,domain_end,n+1)
    h = 1/n

    print("X:", x)

    A = np.zeros((n+1,n+1))

    x_plus_1_coeff = ux/(2*h)-D/(h ** 2)
    x_coeff = -ux/(2*h)+2*D/(h ** 2)
    x_minus_1_coeff = - D / (h ** 2)

    A[0,0]=1

    for ii in range(1,n):
        #Each loop covers a row of coefficients
        A[0+ii,1+ii]=x_plus_1_coeff
        A[0+ii,0+ii]=x_coeff
        A[0+ii,-1+ii]=x_minus_1_coeff

    # Last iteration uses boundary condition
    A[n, n] = 3/(2*h)
    A[n, n-1] = -4/(2*h)
    A[n, n-2] = 1/(2*h)

    b = np.zeros((n+1,1))
    b[0,0]=0 # boundary condition
    b[1:n,0]=s
    b[n,0]=dphidx

    phi_x = np.linalg.solve(A,b)

    # Plot Results
    plt.figure(0)
    plt.plot(x,phi_x)
    plt.xlabel('x')
    plt.ylabel('phi(x)')
    plt.title('Finite Difference Method')

    plt.show()

    return 0

if __name__ == "__main__":
    finiteDifference()