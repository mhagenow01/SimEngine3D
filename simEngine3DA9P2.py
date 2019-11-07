""" Peforms a steady-state finite difference
method for a 1D problem
"""

import numpy as np
import matplotlib.pyplot as plt

def finiteVolume():

    # Diffusion, Velocity Field, and Rate of Generation/Dissipation are constant
    ux = 1
    D = 1
    s = 1
    fx = 0.5
    dphidx = 0

    # Define the discretized points, x
    n = 10  # number of equidistant elements
    domain_start = 0
    domain_end = 1
    h = 1 / n

    # Discretize into the n equidistant volumes
    x = np.linspace(domain_start+h/2,domain_end-h/2,n)

    print("X:", x)


    A = np.zeros((n,n))

    x_plus_1_coeff = -ux*fx-D/h
    x_coeff = -2*ux*fx+2*D/h
    x_minus_1_coeff = -ux*fx-D/h

    # Dirichlet boundary condition
    A[0,0]=-ux*fx+3*D/h
    A[0,1]=-ux*fx-D/h

    for ii in range(1,n-1):
        #Each loop covers a row of coefficients
        A[0+ii,1+ii]=x_plus_1_coeff
        A[0+ii,0+ii]=x_coeff
        A[0+ii,-1+ii]=x_minus_1_coeff

    # Neumann boundary condition
    A[n-1, n-1] = -3*ux*fx+D/h
    A[n-1, n-2] =  -ux*fx-D/h

    b = np.zeros((n,1))
    b[0:n,0]=-s

    phi_x = np.linalg.solve(A,b)

    # Plot Results
    plt.figure(0)
    plt.plot(x,phi_x)
    plt.xlabel('x')
    plt.ylabel('phi(x)')
    plt.title('Finite Volume Method')

    plt.show()

    return 0

if __name__ == "__main__":
    finiteVolume()