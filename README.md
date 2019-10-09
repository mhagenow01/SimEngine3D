# SimEngine3D
### Created by Mike Hagenow as part of ME751

## Dependecies
* Python >3.5
* Numpy

## System Overview

The intent of the system will be to read parameters from the included *.adm file and called through simEngine3D.py. Currently, this functionality is in progress, but will likely be fully implemented when we move to the kinematic/dynamics engine.

## HW 5 - Testing Gcons
I have provided a file called driver.py which allows for testing of the DP1 and CD constraints.
Within this file, you provide the rigid body parameters, vectors, and necessary points on the bodies. It will then create the constraint objects (these are classes with methods for each expression) and then print out all of the results. Since I am using numpy, the format of the arrays sometimes looks a bit weird, but they are functionally correct and of the correct size.
For the Jacobians (position and euler parameters), the output is a tuple of the the two (1x3) or (1x4) arrays.

 
