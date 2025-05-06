import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, tan, atan2

# Vehicle geometry
L = 2.83  # wheelbase in meters

# Load and resample data
file_path = r"C:\Users\camil\Documents\WPI\RBE595-ARN\Assignment 4\victoria_park.csv"
data_in = pd.read_csv(file_path, index_col=0)
data_in.index = pd.to_timedelta(data_in.index, unit='s')
data_in = data_in.resample('1s').mean()

# Extract input measurements
steering = np.deg2rad(data_in['steering'].values)  # convert degrees to radians
speed = data_in['speed'].values
laser_scans = data_in.filter(like='laser').values

# Initial state: robot pose only
x = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
P = np.eye(3) * 0.1  # initial covariance for robot pose

# Full SLAM state and covariance
state = x.copy()
cov = P.copy()

# Landmark bookkeeping
landmark_ids = {}  # maps landmark indices to their positions in the state vector

# Parameters
Q = np.diag([0.2, 0.2, np.deg2rad(1)]) ** 2  # process noise
R = np.diag([1.0, np.deg2rad(10)]) ** 2  # measurement noise

# Motion model
def motion_model(x, v, delta, dt):
    x_pos, y_pos, yaw = x
    dx = v * cos(yaw) * dt
    dy = v * sin(yaw) * dt
    dtheta = (v / L) * tan(delta) * dt
    return np.array([x_pos + dx, y_pos + dy, yaw + dtheta])

# Jacobian of motion model
def jacobian_motion(x, v, delta, dt):
    theta = x[2]
    dx_dtheta = -v * sin(theta) * dt
    dy_dtheta = v * cos(theta) * dt
    G = np.eye(len(x))
    G[0, 2] = dx_dtheta
    G[1, 2] = dy_dtheta
    return G

# Extract landmarks from laser scan
def extract_landmarks(scan_row):
    angles = np.linspace(-np.pi/2, np.pi/2, len(scan_row))
    landmarks = []
    for r, theta in zip(scan_row, angles):
        if 0.5 < r < 30.0:
            lx = r * np.cos(theta)
            ly = r * np.sin(theta)
            landmarks.append([lx, ly])
    return np.array(landmarks)

# EKF SLAM main loop
dt = 1.0
trajectory = []

for i in range(len(speed)):
    v = speed[i]
    delta = steering[i]
    
    ### Prediction step
    robot_state = state[:3]
    robot_state = motion_model(robot_state, v, delta, dt)
    G = jacobian_motion(state[:3], v, delta, dt)

    Fx = np.hstack((np.eye(3), np.zeros((3, len(state) - 3))))
    state[:3] = robot_state
    cov = Fx.T @ G @ Fx @ cov @ Fx.T @ G.T @ Fx + Fx.T @ Q @ Fx

    ### Update step
    observed_landmarks = extract_landmarks(laser_scans[i])
    robot_x, robot_y, robot_theta = state[0:3]

    for lm in observed_landmarks:
        # Transform to world frame
        R_theta = np.array([[cos(robot_theta), -sin(robot_theta)],
                            [sin(robot_theta),  cos(robot_theta)]])
        lm_world = R_theta @ lm + state[:2]
        lm_key = tuple(np.round(lm_world, 1))  # hashable approx key

        if lm_key not in landmark_ids:
            # New landmark: augment state and covariance
            landmark_ids[lm_key] = len(state)
            state = np.hstack((state, lm_world))
            Lm = len(state)
            cov = np.pad(cov, ((0, 2), (0, 2)), 'constant')
            cov[Lm-2:Lm, Lm-2:Lm] = np.eye(2) * 1e2  # high initial uncertainty
            continue

        lm_index = landmark_ids[lm_key]
        lx, ly = state[lm_index:lm_index+2]

        # Expected measurement
        dx = lx - robot_x
        dy = ly - robot_y
        r = np.sqrt(dx**2 + dy**2)
        phi = atan2(dy, dx) - robot_theta
        z_hat = np.array([r, phi])

        # Measurement Jacobian H
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)
        H = np.zeros((2, len(state)))
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, lm_index] = dx / sqrt_q
        H[0, lm_index+1] = dy / sqrt_q
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        H[1, lm_index] = -dy / q
        H[1, lm_index+1] = dx / q

        # Innovation
        z = np.array([np.linalg.norm(lm), atan2(lm[1], lm[0])])
        y = z - z_hat
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi  # wrap angle

        S = H @ cov @ H.T + R
        K = cov @ H.T @ np.linalg.inv(S)

        # Update
        state += K @ y
        cov = (np.eye(len(state)) - K @ H) @ cov

    trajectory.append(state[:3].copy())

trajectory = np.array(trajectory)

# Plot
plt.figure(figsize=(12, 12))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Vehicle Path')
for key, idx in landmark_ids.items():
    plt.plot(state[idx], state[idx + 1], 'r*', markersize=5)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('EKF SLAM - Victoria Park')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
