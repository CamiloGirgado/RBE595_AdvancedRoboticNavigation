import numpy as np
import cv2
import scipy.io
import os
import matplotlib.pyplot as plt
import filterpy
import filterpy.kalman
from scipy.spatial.transform import Rotation as R
from observationModel_1 import observationModel_1
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Q_continuous_white_noise
from filterpy.kalman import MerweScaledSigmaPoints


class UnscentedKalmanFilter:
    def __init__(self, observationmodel_1):
        self.data = process_data
        self.Q = np.eye(15)*0.0015
        self.Q[np.arange(6), np.arange(6)] = [0.015, 0.015, 0.015, 0.001, 0.001, 0.001]
        R_meas = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        g = np.array([0, 0 -9.81]) # Gravity Vector
        self.ukf = UnscentedKalmanFilter(dim_x=15, dim_z=6, dt=0.001, hx=self.hx, fx=self.fx, points=points)
        points = MerweScaledSigmaPoints(n=15, alpha=.1, beta=2., kappa=0)
        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

# -------------------- Process Model --------------------
    def process_model(state, u_ω, u_a, g, dt):
        """Defines the process model for the quadcopter's state."""
        p = state[:3]   # Position
        q = state[3:6]  # Orientation (Euler angles)
        v = state[6:9]  # Velocity
        ba = state[12:]  # Accelerometer bias
    
    # Derivatives
        p_dot = v  # Velocity is the derivative of position
        v_dot = R_matrix @ (u_a - ba) - g  # Corrected acceleration
        state_dot = np.concatenate([p_dot, np.zeros(3), v_dot, np.zeros(6)])  # Assume biases are constant
        return state_dot

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
        self.rpy = data['rpy']
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        return np.matrix([
            [np.cos(pitch), 0, -np.sin(roll)*np.cos(pitch)],
            [0, 1, np.sin(roll)],
            [np.sin(pitch), 0, np.cos(roll)*np.cos(pitch)],
            ])

    def fx(self, x, dt, data):
        xout[0] = x[6]*dt + x[0]
        xout[1] = x[7]*dt + x[1]
        xout[2] = x[8]*dt + x[2]
        G_matrix = self.G_Matrix(x[3:6])
        U_w = (np.array([data['omg']])).T
        q_dot = np.linalg.inv(G_matrix) @ U_w
        xout[3:6] = q_dot.squeeze()
        Rq_matrix = self.Rq_matrix(data[['rpy']])
        U_a = (np.array([data['acc']])).T
        xout[6:9] = (Rq_matrix @ U_a + g).squeeze()

        # Define the covariance matrices for gyroscope and accelerometer bias noise
        #sigma_bg_x = 0.2
        #sigma_bg_y = 0.2
        #sigma_bg_z = 5.5
        #sigma_ba_x = 0.2
        #sigma_ba_y = 0.2
        #sigma_ba_z = 5.5

        Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])  # Gyroscope bias noise covariance
        Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])  # Accelerometer bias noise covariance
    
        # Generate noise for gyroscope and accelerometer biases
        Nbg = np.random.multivariate_normal(mean=np.zeros(3), cov=Qg)
        Nba = np.random.multivariate_normal(mean=np.zeros(3), cov=Qa)
        xout[9:12] = x[9:12]
        xout[12:15] = x[12:15]

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


# -------------------- Main Execution --------------------
camera_matrix = np.array([[314.1779, 0, 199.4848], [0, 314.2218, 113.7838], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float32)
data_folder = "/home/camilo/dev/RBE_595_ARN/data"

estimated_positions, true_positions, true_orientations = process_data(data_folder, camera_matrix, dist_coeffs, tag_corners_world)

for t in range(len(estimated_positions)):
    dt = 0.01  # Assuming a fixed timestep or based on actual timestamps
    u_ω = np.array([data['omg']]).T
    u_a = np.array([data['acc']]).T

    state_pred, P_pred = ukf_predict(state, P, u_ω, u_a, g, dt)
    state, P = ukf_update(state_pred, P_pred, estimated_positions[t], R_meas)
    particles = particle_filter_predict(particles, u_ω, u_a, g, dt, Q)
    weights = particle_filter_update(particles, estimated_positions[t], R_meas, weights)
    particles = low_variance_resampling(particles, weights)

    if t % 10 == 0:
        plot_trajectory(estimated_positions, true_positions)
        plot_euler_angles(estimated_positions, true_orientations)