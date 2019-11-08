# SimEngine3D
### Created by Mike Hagenow as part of ME751

## Dependecies
* Python >3.5
* Numpy
* Matplotlib

## System Overview

The intent of the system will be to read parameters from the included *.adm file and called through simEngine3D.py. Currently, this functionality is in progress, but will likely be fully implemented when we move to the kinematic/dynamics engine.

## HW 5 - Testing Gcons
I have provided a file called test_code/driver_HW5.py which allows for testing of the DP1 and CD constraints.
Within this file, you provide the rigid body parameters, vectors, and necessary points on the bodies. It will then create the constraint objects (these are classes with methods for each expression) and then print out all of the results. Since I am using numpy, the format of the arrays sometimes looks a bit weird, but they are functionally correct and of the correct size.
For the Jacobians (position and euler parameters), the output is a tuple of the the two (1x3) or (1x4) arrays. If j is
the ground, there is a constraint flag to not return the jacobians.

## HW 6 - Pendulum Kinematics
I have provided a file called driver.py which allows for testing of the DP2 and D GCons. There also various other
testing files for the other GCons (e.g. SJ, Revolute, etc.) in the test_code folder.
For #2, there is a file simEngine3DA6P2.py (note: no hyphen) that sets up the pendulum problem and prints out the initial values
of the constraint-related quantities. Note: the expressions include geometric, driving, and normalization constraints. 
For #3, there is a file simEngine3DA6P3.y (note: no hyphen) that runs the kinematic analysis for the pendulum for 10 seconds. The Newton-Raphson does 10 iterations per timestep. Once completed, the program generates 6 plots for the position/velocity/acceleration information for O' and Q.

## HW 7 - Inverse Dynamics
I have provided a file simEngine3DA7P1.py which performs the kinematic analysis and plots the torques from the driving constraints
as well as the overall reaction forces for body i (not required). I have provided a file simEngine3DA7P2.py which implements
backwards Euler for the given IVP (note the system does not evolve in time from the given starting configuration). I have provided a file simEngine3DA7P3.py
which performs the Backwards Euler vs BDF (4th order) Convergence Analysis. It will plot the linear and log-log convergence plots
as well as calculate the slopes on the log-log plot.

## HW 8 - Dynamics Analysis
I have provided a file simEngine3DA8P1.py which performs a dynamic analysis and creates plots for a single revolute pendulum. You can set
the timestep and simulation length at the beginning of the file. I have provided a file simEngine3DA8P2.py which performs a dynamic analysis and creates plots for a double revolute pendulum. You can set
the timestep and simulation length at the beginning of the file.

## HW 9 - CFD Track Finite Difference/Finite Volume
I have provided a file simEngine3DA9P1.py which implements the finite difference method for the 1D problem.
I have provided a file simEngine3DA9P2.py which implements the finite volume method for the 1D problem.
Both methods are set up for n=1000 points (finite difference is n+1). Both methods produce the same result
and the first method was validated against the suggested analytical solutions.