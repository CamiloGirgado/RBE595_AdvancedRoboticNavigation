import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_in = pd.read_csv('victoria_park.csv', index_col=0)
data_in.index = pd.to_timedelta(data_in.index, unit='s')
data_in = data_in.resample('1s').mean()
data_in = data_in.dropna()

# Constants
L = 2.83 
dt = 1.0
a = 0.5 
b = 3.78 

laser_cols = [col for col in data_in.columns if 'laser' in col]
laser_angles = np.linspace(-np.pi / 2, np.pi / 2, len(laser_cols))

# Initialize state
mu = np.array([0.0, 0.0, 0.0])  # x, y, yaw
sigma = np.diag([1.0, 1.0, 0.1])  # initial covariance
trajectory = [mu[:2].copy()]
landmarks = []

# Motion model
def motion_model(mu, v, delta, dt=1.0):
    x, y, theta = mu
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(delta) * dt
    return np.array([x, y, theta])

# Jacobian
def jacobian_F(mu, v, delta, dt=1.0):
    _, _, theta = mu
    F = np.eye(3)
    F[0, 2] = -v * np.sin(theta) * dt
    F[1, 2] =  v * np.cos(theta) * dt
    return F

# Motion noise
Q = np.diag([0.5, 0.5, 0.05])**2

for _, row in data_in.iterrows():
    v = row['speed']
    delta = row['steering']

    # Predict
    mu = motion_model(mu, v, delta, dt)
    F = jacobian_F(mu, v, delta, dt)
    sigma = F @ sigma @ F.T + Q

    # Extract valid laser measurements
    ranges = row[laser_cols].values
    valid = (ranges > 1.0) & (ranges < 80.0)
    ranges = ranges[valid]
    angles = laser_angles[valid]

    # Compute camera position in world frame
    x, y, theta = mu
    cam_x = x + a * np.cos(theta) - b * np.sin(theta)
    cam_y = y + a * np.sin(theta) + b * np.cos(theta)

    for r, angle in zip(ranges, angles):
        lx = cam_x + r * np.cos(theta + angle)
        ly = cam_y + r * np.sin(theta + angle)
        landmarks.append([lx, ly])

    trajectory.append(mu[:2].copy())

# Convert to arrays
trajectory = np.array(trajectory)
landmarks = np.array(landmarks)

# Plot results
plt.figure(figsize=(10, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Estimated Path')

if landmarks.shape[0] > 0 and landmarks.ndim == 2:
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='green', alpha=0.5, label='Landmarks')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('EKF SLAM - Victoria Park (Camera Offset Included)')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
