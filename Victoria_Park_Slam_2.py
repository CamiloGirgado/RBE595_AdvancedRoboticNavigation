import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 2.83  # wheelbase
MAX_RANGE = 80.0  # max lidar range in meters
ANGLE_MIN = -np.pi / 2
ANGLE_MAX = np.pi / 2
NUM_BEAMS = 361
beam_angles = np.linspace(ANGLE_MIN, ANGLE_MAX, NUM_BEAMS)

# Load data
csv_path = r'C:/Users/camil/Documents/WPI/RBE595-ARN/Assignment 4/victoria_park.csv'
data = pd.read_csv(csv_path, index_col=0)
data.index = pd.to_timedelta(data.index, unit='s')
data = data.resample('1s').mean().interpolate()

# Helpers
def deg2meters(lat, lon):
    lat_scale = 111320
    lon_scale = 111320 * np.cos(np.radians(lat))
    return lat * lat_scale, lon * lon_scale

def motion_model(mu, u, dt):
    x, y, theta = mu[0:3]
    v, delta = u
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(delta) * dt
    mu[0:3] = [x, y, theta]
    return mu

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Initialization
dt = 1.0
mu = np.zeros(3)  # initial state: x, y, theta
Sigma = np.eye(3) * 0.1
R = np.diag([0.5, 0.5, np.deg2rad(1.0)])**2
Q_gps = np.diag([15.0, 15.0])**2
Q_lidar = np.diag([1.0, np.deg2rad(2.0)])**2

trajectory = []
gps_trace = []

# Map storage
landmarks = {}  # landmark_id: index in state
next_landmark_id = 0

# SLAM loop
for _, row in data.iterrows():
    v = row['speed']
    delta = row['steering']
    u = [v, delta]

    # Predict
    mu = motion_model(mu, u, dt)
    theta = mu[2]
    Fx = np.eye(len(mu))
    Fx[0:3, 0:3] = np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1,  v * np.cos(theta) * dt],
        [0, 0, 1]
    ])
    Sigma = Fx @ Sigma @ Fx.T + np.pad(R, ((0, Sigma.shape[0] - 3), (0, Sigma.shape[1] - 3)))

    # GPS update
    if not np.isnan(row['latitude']) and not np.isnan(row['longitude']):
        z_gps = np.array(deg2meters(row['latitude'], row['longitude']))
        H = np.zeros((2, len(mu)))
        H[:, 0:2] = np.eye(2)
        z_hat = mu[0:2]
        y = z_gps - z_hat
        S = H @ Sigma @ H.T + Q_gps
        K = Sigma @ H.T @ np.linalg.inv(S)
        mu += K @ y
        Sigma = (np.eye(len(mu)) - K @ H) @ Sigma
        gps_trace.append(z_gps)

    # Lidar updates
    for i in range(NUM_BEAMS):
        beam_col = f'laser_{i}'
        if beam_col not in row or np.isnan(row[beam_col]):
            continue
        r = row[beam_col]
        if r <= 1.0 or r >= MAX_RANGE:
            continue

        angle = beam_angles[i] + mu[2]
        lx = mu[0] + r * np.cos(angle)
        ly = mu[1] + r * np.sin(angle)
        landmark_pos = np.array([lx, ly])

        # Nearest neighbor association
        associated = False
        for lid, idx in landmarks.items():
            lm = mu[idx:idx+2]
            dist = np.linalg.norm(lm - landmark_pos)
            if dist < 2.0:  # association threshold
                dx, dy = lm - mu[0:2]
                r_hat = np.sqrt(dx**2 + dy**2)
                b_hat = normalize_angle(np.arctan2(dy, dx) - mu[2])
                z_hat = np.array([r_hat, b_hat])

                H = np.zeros((2, len(mu)))
                q = dx**2 + dy**2
                H[0, 0] = -dx / np.sqrt(q)
                H[0, 1] = -dy / np.sqrt(q)
                H[0, idx] = dx / np.sqrt(q)
                H[0, idx+1] = dy / np.sqrt(q)
                H[1, 0] = dy / q
                H[1, 1] = -dx / q
                H[1, 2] = -1
                H[1, idx] = -dy / q
                H[1, idx+1] = dx / q

                z = np.array([r, normalize_angle(angle - mu[2])])
                y = z - z_hat
                y[1] = normalize_angle(y[1])
                S = H @ Sigma @ H.T + Q_lidar
                K = Sigma @ H.T @ np.linalg.inv(S)
                mu += K @ y
                Sigma = (np.eye(len(mu)) - K @ H) @ Sigma
                associated = True
                break

        if not associated:
            # Initialize new landmark
            mu = np.concatenate([mu, landmark_pos])
            idx = len(mu) - 2
            landmarks[next_landmark_id] = idx
            next_landmark_id += 1

            # Expand Sigma
            Sigma = np.pad(Sigma, ((0, 2), (0, 2)), 'constant')
            Sigma[idx:idx+2, idx:idx+2] = np.eye(2) * 5.0

    trajectory.append(mu[0:2])

# --- Plot Results ---
trajectory = np.array(trajectory)
gps_trace = np.array(gps_trace)

plt.figure(figsize=(12, 9))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='EKF SLAM Trajectory')
plt.plot(gps_trace[:, 0], gps_trace[:, 1], 'r.', alpha=0.4, label='GPS')
for lid, idx in landmarks.items():
    plt.plot(mu[idx], mu[idx+1], 'go', markersize=3)
plt.title("EKF SLAM with Lidar Landmarks")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
