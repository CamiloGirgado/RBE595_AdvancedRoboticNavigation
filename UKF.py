import numpy as np
import cv2
import scipy.io
import os
import matplotlib.pyplot as plt
import filterpy
import filterpy.kalman
from scipy.spatial.transform import Rotation as R
from observationModel_1 import observationModel_1
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import JulierSigmaPoints


class UnscentedKalmanFilter:
    def __init__(self, observationmodel_1):
        self.observationModel_1 = observationModel_1
        points = JulierSigmaPoints(n=15, kappa=0.1)
        self.Q = np.eye(15)*0.0015
        self.Q[np.arange(6), np.arange(6)] = [0.015, 0.015, 0.015, 0.001, 0.001, 0.001]
        self.ukf = UnscentedKalmanFilter(dim_x=15, dim_z=6, dt=0.001, hx=self.hx, fx=self.fx, points=points)
        self.R = self.P_Matrix()
        self.ukf.R = self.R
        self.check_covariance_matrix(self.R)
        self.ukf.Q = self.Q
        points = MerweScaledSigmaPoints(n=15, alpha=.1, beta=2., kappa=0)
        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

    def P_Matrix(self):
        P = np.array(
        [[ 0.01248985 , 0.00179274 , 0.01191035 , 0.00812441,  0.00853663, -0.00074059],
        [ 0.00179274 , 0.00494662 , 0.00222319 , 0.00453181, -0.00188542, -0.00014287],
        [ 0.01191035 , 0.00222319 , 0.01989463,  0.00623472,  0.00840728, -0.00132054],
        [ 0.00812441 , 0.00453181 , 0.00623472 , 0.00973619,  0.00250991, -0.00037419],
        [ 0.00853663 ,-0.00188542 , 0.00840728 , 0.00250991,  0.00830289, -0.00050637],
        [-0.00074059 ,-0.00014287 ,-0.00132054, -0.00037419, -0.00050637,  0.00012994]]
        )
        P2 = np.diag([0.008]*6)
        return P2

    def G_Matrix(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        return np.matrix([
            [np.cos(pitch), 0, -np.sin(roll)*np.cos(pitch)],
            [0, 1, np.sin(roll)],
            [np.sin(pitch), 0, np.cos(roll)*np.cos(pitch)],
            ])
    
    def check_covariance_matrix(self, matrix):
        if np.allclose(matrix, matrix.T):
            if self.debug:
                print("Covariance matrix is symmetric.")
        else:
            print("Covariance matrix is not symmetric.")
    
    def hx(self, x):
        hx = self.H @ x
        return hx.T

    def fx(self, x, dt, data):
        xout = x.copy()
        xout[0] = x[6]*dt + x[0]
        xout[1] = x[7]*dt + x[1]
        xout[2] = x[8]*dt + x[2]
        g = np.array([0, 0 -9.81]) # Gravity Vector
        G_matrix = self.G_Matrix(x[3:6])
        U_w = (np.array([data['omg']])).T
        q_dot = np.linalg.inv(G_matrix) @ U_w
        xout[3:6] = q_dot.squeeze()
        Rq_matrix = self.Rq_matrix(data[['rpy']])
        U_a = (np.array([data['acc']])).T
        xout[6:9] = (Rq_matrix @ U_a + g).squeeze()

        return xout

    def Rq_matrix(self, data):
        rpy = data['rpy']
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        R = rotation_z @ rotation_y @ rotation_x
        check = R.from_matrix(self.R).as_euler('xyz', degrees=False)
        
        return R

    def hx(self, x):
        hx=self.H @ x
        return hx.T

# -------------------- UKF Prediction and Update --------------------

    def predict(self, dt, data):
        self.ukf.predict(dt,fx=self.fx,data=data)
        return self.ukf.x

    def update(self, z):
        self.ukf.update(z.squeeze()) 
        return self.ukf.x