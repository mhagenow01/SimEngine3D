""" Main function to run simEngine
"""

import numpy as np
from GCons.P_norm import P_norm
from Utilities.kinematic_identities import G_from_p

class SimEngine:

    def __init__(self):
        self.bodies = []
        self.groundBodies = []
        self.constraints = []
        self.norm_constraints=[]

        # Return stuff
        self.Phi = np.array([0])

    ######### Functions for building up the Simulation #######################
    def addRigidBody(self, i, ground=False):
        if ground is False:
            self.bodies.append(i)
            self.norm_constraints.append(P_norm(i))
        else:
            self.groundBodies.append(i)
    def addConstraint(self,constraint):
        self.constraints.append(constraint)

    def updateRigidBody(self,new_values,level=0):

        new_values.reshape((7*len(self.bodies),1))
        for ii in range(0,len(self.bodies)):
            new_r_level = new_values[7*ii:7*ii+3,1].reshape((3,1))
            new_p_level = new_values[7*ii+3:7*ii+7,1].reshape((4,1))

            if level==0: # positions
                self.bodies[ii].r = new_r_level
                self.bodies[ii].p = new_p_level
            if level == 1: # velocities
                self.bodies[ii].r_dot = new_r_level
                self.bodies[ii].p_dot = new_p_level
            if level == 2: # accelerations
                self.bodies[ii].r_ddot = new_r_level
                self.bodies[ii].p_ddot = new_p_level




    ######### Functions that return matrices and quantities of interest ############################

    def getPhi(self):


    def getM(self):
        # Gets the overall mass matrix for all of the rigid bodies
        # size is 3nb x 3nb
        num_bodies = len(self.bodies)

        M = np.zeros((3*num_bodies, 3*num_bodies))

        # Mass matrix is block diagonal and constant in time
        for ii in range(0,num_bodies):
            M[3*ii:3(ii+1),3+ii:3(ii+1)] = self.bodies[ii].M
        return M

    def getJp(self):
        # Gets the overall inertia matrix for all of the rigid bodies
        # size is 4nb x 4nb
        num_bodies = len(self.bodies)
        Jp = np.zeros((4 * num_bodies, 4 * num_bodies))

        # Calculate the inertia matrix
        for ii in range(0, num_bodies):
            # this changes based on configuration - not constant in time
            Jpblock = 4 * G_from_p(self.bodies[ii].p).transpose() @ self.bodies[ii].J_bar @ G_from_p(self.bodies[ii].p)
            Jp[4 * ii:4(ii + 4), 4 + ii:4(ii + 1)] = Jpblock

        return Jp

