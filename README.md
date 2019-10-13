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
of the constraint-related quantities. Note: the expressions are for the geometric and driving constraints and DO NOT include the euler parameter normalization constraints. 
For #3, there is a file simEngine3DA6P3.y (note: no hyphen) that runs the kinematic analysis for the pendulum for 10 seconds. The Newton-Raphson does 10 iterations per timestep. Once completed, the program generates 6 plots for the position/velocity/acceleration information for O' and Q.

 
