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

# Initial state
x = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
trajectory = [x.copy()]

# Motion model

def motion_model(x, v, delta, dt):
    x_pos, y_pos, yaw = x
    dx = v * cos(yaw) * dt
    dy = v * sin(yaw) * dt
    dtheta = (v / L) * tan(delta) * dt
    return np.array([x_pos + dx, y_pos + dy, yaw + dtheta])

# Process all data
dt = 1.0  # 1Hz after resampling
for i in range(len(speed)):
    v = speed[i]
    delta = steering[i]
    x = motion_model(x, v, delta, dt)
    trajectory.append(x.copy())

trajectory = np.array(trajectory)

# Simple landmark extraction from laser scans (peaks within range)
def extract_landmarks(scan_row):
    angles = np.linspace(-np.pi/2, np.pi/2, len(scan_row))
    landmarks = []
    for r, theta in zip(scan_row, angles):
        if 0.5 < r < 80.0:
            lx = r * np.cos(theta)
            ly = r * np.sin(theta)
            landmarks.append([lx, ly])
    return np.array(landmarks)

# Example: Extract landmarks from one scan
sample_landmarks = extract_landmarks(laser_scans[100])

# Transform sample landmarks to world frame
x_pos, y_pos, yaw = trajectory[100]
R = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])
transformed_landmarks = (R @ sample_landmarks.T).T + np.array([x_pos, y_pos])

# Plot
plt.figure(figsize=(10, 10))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Vehicle Path')
plt.scatter(transformed_landmarks[:, 0], transformed_landmarks[:, 1], c='red', s=10, label='Landmarks')
plt.axis('equal')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Victoria Park SLAM (Odometry + Landmark Extraction)')
plt.legend()
plt.grid()
plt.show()
