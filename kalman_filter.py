'''
kalman filter
'''
import numpy as np


class KalmanFilter(object):

    def __init__(self):
        """
        Initialize variable used by Kalman Filter class
        """
        #self.dt = 0.005  # delta time

        self.H = np.eye(4)  # matrix in observation equations
        self.x_e = np.zeros([4,1])  # previous state vector
        # (x,y) tracking object center

        self.C_e = np.eye(4)  # covariance matrix
        # the higher initial C_e, the less dependancy of prediction
        self.A = np.eye(4) # state transition mat


        self.W = np.eye(self.x_e.shape[0])  # process noise matrix
        self.V = 10*np.eye(4)  # observation noise matrix
        #the higher initial V, the less dependancy of measurement
        self.lastResult = np.zeros([4,1])

    def predict(self,factor = 1.0):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            x_e: previous state vector
            C_e: previous covariance matrix
            A: state transition matrix
            W: process noise matrix

        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.W *= factor
        self.x_p = np.dot(self.A, self.x_e)
        # Predicted estimate covariance
        self.C_p = np.dot(self.A, np.dot(self.C_e, self.A.T)) + self.W
        self.lastResult = self.x_p  # same last predicted result
        return self.x_p

    def correct(self, z, flag = True):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        x: predicted state vector u
        H: matrix in observation equations
        z: vector of observations
        C_p: predicted covariance matrix
        W: process noise matrix
        V: observation noise matrix

        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.z = self.lastResult
        else:  # update using detection
            self.z = z
        self.z = np.reshape(self.z,[4,1])
        self.K = np.dot(self.C_p, np.dot(self.H.T, np.linalg.
                                         inv(np.dot(self.H, np.dot(self.C_p,self.H.T)) + self.V)))

        self.x_e = self.x_p + np.dot(self.K, (self.z - np.dot(self.H, self.x_p)))
        self.C_e = self.C_p - np.dot(self.K, np.dot(self.H, self.C_p))
        self.lastResult = self.x_e
        return self.x_e

