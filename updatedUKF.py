from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.interpolate import LinearNDInterpolator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from observationModel_1 import observationModel
from numpy.random import randn


class UkfFilter2:
    def __init__(self, measurement_data):
        self.debug = False
        self.measurement_data = measurement_data
        self.R = None
        self.n_states = 15
        self.n_measurements = 6
        # points = MerweScaledSigmaPoints(n=15, alpha=.1, beta=2., kappa=0)
        
        self.Q = np.eye(15)*0.001
        self.Q[0,0]=0.2
        self.Q[1,1]=0.2
        self.Q[2,2]=0.2
        self.Q[3,3]=0.02
        self.Q[4,4]=0.02
        self.Q[5,5]=0.02
        self.check_covariance_matrix(self.Q)
        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ])
        
        #  The measurement mean and noise covariance
        self.R = self.P_Matrix()

        self.check_covariance_matrix(self.R)
        
        # # Define the covariance matrices for gyroscope and accelerometer bias noise
        sigma_bg_x = 0.0025
        sigma_bg_y = 0.01
        sigma_bg_z = 0.0015
        
        sigma_ba_x = 0.015
        sigma_ba_y = 0.015
        sigma_ba_z = 0.015
        
        self.Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])  # Gyroscope bias noise covariance
        self.Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])  # Accelerometer bias noise covariance
        # Generate random noise for biases (Nbg and Nba)

        alpha = 0.1
        beta = 2
        kappa = 0
        self.n = self.n_states
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n

        # Weights
        self.Wm = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambda_))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)


        mu = np.zeros(self.n_states)
        Sigma = np.eye(self.n_states) * 0.1
        points,_,_ = self.julier_sigma_points(mu, Sigma)
        # print("Sigma points: ", points)
        # State
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

    def julier_sigma_points(self,mu, Sigma, alpha=1e-3, beta=2, kappa=0):
        n = mu.shape[0]
        lambda_ = alpha**2 * (n + kappa) - n
        sigma_points = np.zeros((2 * n + 1, n))
        weights_mean = np.zeros(2 * n + 1)
        weights_cov = np.zeros(2 * n + 1)

        # Cholesky decomposition
        sqrt_matrix = np.linalg.cholesky((n + lambda_) * Sigma)

        # Sigma points
        sigma_points[0] = mu
        for i in range(n):
            sigma_points[i + 1] = mu + sqrt_matrix[:, i]
            sigma_points[n + i + 1] = mu - sqrt_matrix[:, i]

        # Weights
        weights_mean[0] = lambda_ / (n + lambda_)
        weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        weights_mean[1:] = weights_cov[1:] = 1 / (2 * (n + lambda_))

        return sigma_points, weights_mean, weights_cov
    
    def check_covariance_matrix(self, matrix):
        if np.allclose(matrix, matrix.T):
            if self.debug:
                print("Covariance matrix is symmetric.")
        else:
            print("Covariance matrix is not symmetric.")

        # Check if positive definite
        eigenvalues = np.linalg.eigvals(matrix)
        if np.all(eigenvalues > 0):
            if self.debug:
                print("Covariance matrix is positive definite.")
        else:
            print("Covariance matrix is not positive definite.")
    
    def generate_sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        self.x = self.x.reshape(-1,1)
        sigma_points[0] = self.x.squeeze()
        sqrt_P = np.linalg.cholesky((self.n + self.lambda_) * self.P)
        for i in range(self.n):
            sigma_points[i + 1] = (self.x + sqrt_P[:, i].reshape(-1,1)).squeeze()
            sigma_points[i + 1 + self.n] = (self.x - sqrt_P[:, i].reshape(-1,1)).squeeze()
            
        return sigma_points
    
    def predict(self, dt, data):
        sigma_pts = self.generate_sigma_points()
        propagated = np.array([self.fx(pt, dt,data) for pt in sigma_pts])
        self.x = np.sum(self.Wm[:, None] * propagated, axis=0)
        self.P = self.Q.copy()
        for i in range(2 * self.n + 1):
            diff = (propagated[i] - self.x).reshape(-1, 1)
            self.P += self.Wc[i] * diff @ diff.T
        self._sigma_pts_pred = propagated
        return self.x.squeeze()

    def update(self, z):
        Z_sigma = np.array([self.hx(pt) for pt in self._sigma_pts_pred])
        z_pred = np.sum(self.Wm[:, None] * Z_sigma, axis=0)

        S = self.R.copy()
        for i in range(2 * self.n + 1):
            dz = (Z_sigma[i] - z_pred).reshape(-1, 1)
            S += self.Wc[i] * dz @ dz.T

        Pxz = np.zeros((self.n, self.n_measurements))
        for i in range(2 * self.n + 1):
            dx = (self._sigma_pts_pred[i] - self.x).reshape(-1, 1)
            dz = (Z_sigma[i] - z_pred).reshape(-1, 1)
            Pxz += self.Wc[i] * dx @ dz.T

        K = Pxz @ np.linalg.inv(S)
        self.x = self.x.reshape(-1, 1) + K @ (z - z_pred.reshape(-1, 1))
        self.P = self.P - K @ S @ K.T

        return self.x.squeeze()
        
    def G_Matrix(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        return np.matrix([
            [np.cos(pitch), 0, -np.sin(roll)*np.cos(pitch)],
            [0, 1, np.sin(roll)],
            [np.sin(pitch), 0, np.cos(roll)*np.cos(pitch)],
            ])
    
    # def G_Matrix(self, rpy):
    #     # rpy = data['rpy']
    #     roll= rpy[0]   # phi
    #     pitch = rpy[1] # theta
    #     yaw = rpy[2]   # psi
    #     c_pitch = np.cos(pitch)
    #     sc_02 = -np.sin(roll)*np.cos(pitch)
    #     s_roll = np.sin(roll)
    #     s_pitch = np.sin(pitch)
    #     c_roll_pitch = np.cos(roll)*np.cos(pitch)
    #     return np.array([
    #         [c_pitch, 0, sc_02],
    #         [0,       1, s_roll],
    #         [s_pitch, 0, c_roll_pitch],
            
    #     ])
    

    def fx(self,x, dt,data):
        xout = x.copy()
        if data.get('omg') is None or data.get('acc') is None:
            return xout
        
        gyro_bias_prev = x[9:12]
        accel_bias_prev = x[12:15]
        # gyro_bias_next = gyro_bias_prev + np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)*dt
        gyro_bias_next = (np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg))*dt
        accel_bias_next = (np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa))*dt
        xout[9:12] = gyro_bias_next
        xout[12:15] = accel_bias_next

        G = self.G_Matrix(x[3:6])
        U_w = (np.array([data['omg']]) + gyro_bias_prev).T
        q_dot = np.linalg.inv(G) @ U_w
        xout[3:6] = q_dot.squeeze()
        
        U_a = (np.array([data['acc']]) + accel_bias_prev ).T
        Rq_matrix = self.Rq_matrix(x[3:6])
        g = np.array([[0, 0, 9.81]]).T
        xout[6:9] = (Rq_matrix.T @ U_a - g).squeeze()
        
        xout[0] = (x[6] * dt + x[0]) + x[9]
        xout[1] = (x[7] * dt + x[1]) + x[10]
        xout[2] = (x[8] * dt + x[2]) + x[11]
        
        return xout
    
    
    def hx(self, x):
        hx=self.H @ x
        return hx.T
        
    
    def P_Matrix(self):
        P = np.array(
[[ 0.01248985 , 0.00179274 , 0.01191035 , 0.00812441,  0.00853663, -0.00074059],
 [ 0.00179274 , 0.00494662 , 0.00222319 , 0.00453181, -0.00188542, -0.00014287],
 [ 0.01191035 , 0.00222319 , 0.01989463,  0.00623472,  0.00840728, -0.00132054],
 [ 0.00812441 , 0.00453181 , 0.00623472 , 0.00973619,  0.00250991, -0.00037419],
 [ 0.00853663 ,-0.00188542 , 0.00840728 , 0.00250991,  0.00830289, -0.00050637],
 [-0.00074059 ,-0.00014287 ,-0.00132054, -0.00037419, -0.00050637,  0.00012994]]
        )
        P2 = np.diag([.008]*6)
        P2[0,0] = 0.008
        P2[1,1] = 0.008
        P2[2,2] = 0.008
        P2[3,3] = 0.008
        P2[4,4] = 0.008
        P2[5,5] = 0.001
        # return P2
        return P2
        
    def Q_Matrix(self, data):
        pass

    def Rq_matrix(self, rpy):
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        r= rotation_z @ rotation_y @ rotation_x
        
        return r