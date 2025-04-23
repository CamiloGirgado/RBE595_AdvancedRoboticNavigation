import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
from observationModel import process_data
from observationModel import estimate_pose

R_meas = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
g = np.array([0, 0 -9.81]) # Gravity Vector
Q = np.diag([0.01 * 9 + [0.001] * 6])

# -------------------- Process Model --------------------
def process_model(state, u_ω, u_a, g, dt):
    """Defines the process model for the quadcopter's state."""
    p = state[:3]   # Position
    q = state[3:6]  # Orientation (Euler angles)
    v = state[6:9]  # Velocity
    ba = state[12:]  # Accelerometer bias

    # Rotation matrix from Euler angles
    R_matrix = np.array([
        [np.cos(q[2]) * np.cos(q[1]), -np.sin(q[2]), -np.cos(q[2]) * np.sin(q[1])],
        [np.sin(q[2]) * np.cos(q[1]), np.cos(q[2]), -np.sin(q[2]) * np.sin(q[1])],
        [-np.sin(q[1]), 0, np.cos(q[1])]
    ])

    # Derivatives
    p_dot = v  # Velocity is the derivative of position
    v_dot = R_matrix @ (u_a - ba) - g  # Corrected acceleration
    state_dot = np.concatenate([p_dot, np.zeros(3), v_dot, np.zeros(6)])  # Assume biases are constant
    return state_dot

# -------------------- Measurement Model --------------------
def measurement_model(state):
    """Maps the state to the measurement space."""
    p = state[:3]   # Position
    q = state[3:6]  # Orientation (Euler angles)
    return np.concatenate([p, q])

# -------------------- Compute Angular Velocity --------------------
def compute_angular_velocity(orientations, dt):
    """Computes angular velocity from Euler angles using finite differences."""
    omega = np.zeros_like(orientations)
    omega[1:] = (orientations[1:] - orientations[:-1]) / dt
    return omega

# -------------------- Compute Linear Acceleration --------------------
def compute_linear_acceleration(positions, dt):
    """Computes linear acceleration from position using finite differences."""
    velocity = np.zeros_like(positions)
    acceleration = np.zeros_like(positions)
    
    velocity[1:] = (positions[1:] - positions[:-1]) / dt  # First derivative
    acceleration[1:] = (velocity[1:] - velocity[:-1]) / dt  # Second derivative
    
    return acceleration

# -------------------- Process Data --------------------
def process_data(directory, camera_matrix, dist_coeffs, tag_corners_world):
    """Processes all .mat files in the given directory and estimates pose."""
    true_positions = []
    true_orientations = []

    # List all MAT files in the folder
    mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]

    if not mat_files:
        print("No .mat files found in the directory!")
        return None, None, None, None
    mat_files = ['studentdata6.mat']
    for file_name in mat_files:
        file_path = os.path.join(directory, file_name)
        print(f"Loading file: {file_name}")
        data = scipy.io.loadmat(file_path, simplify_cells=True)
        # Debugging: Print the structure of the .mat file
        print(f"Structure of {file_name}:", data.keys())
        if 'data' not in data or 'time' not in data or 'vicon' not in data:

            print(f"Skipping {file_name} due to missing keys!")

            continue
        dataset = data['data']
        time_stamps = data['time']
        vicon = data['vicon']
        true_positions.extend(vicon[0:3, :])
        estimated_all = None
        # true_orientations.extend(vicon[3:5, :])
        true_orientations.extend(np.vstack((vicon[3:6, :],np.array([time_stamps]))))
        for entry in dataset:
            position, orientation = estimate_pose(entry, camera_matrix, dist_coeffs, tag_corners_world)
            if position is not None:
                estimated = np.hstack((position, orientation))
                estimated = np.hstack((estimated, entry['t']))
                if estimated_all is None:
                    estimated_all = estimated
                else:
                    estimated_all = np.vstack((estimated_all, estimated))
                #estimated_positions.append(position)
                #estimated_orientations.append(orientation)
        break
    return estimated_all, np.array(true_positions), np.array(true_orientations)


# -------------------- EKF Prediction and Update --------------------
def ekf_predict(state, P, u_ω, u_a, g, dt):
    F = np.eye(15)
    F[0:3, 6:9] = np.eye(3) * dt
    P_pred = F @ P @ F.T + Q
    return state + process_model(state, u_ω, u_a, g, dt) * dt, P_pred

def ekf_update(state_pred, P_pred, z, R_meas):
    H = np.eye(6, 15)
    z_pred = measurement_model(state_pred)
    S = H @ P_pred @ H.T + R_meas
    K = P_pred @ H.T @ np.linalg.inv(S)
    state_updated = state_pred + K @ (z - z_pred)
    P_updated = (np.eye(15) - K @ H) @ P_pred
    return state_updated, P_updated

# -------------------- Particle Filter --------------------

def particle_filter_predict(particles, u_ω, u_a, g, dt, Q):
    noise = np.random.multivariate_normal(np.zeros(15), Q, size=particles.shape[0])
    for i in range(particles.shape[0]):
        particles[i] += process_model(particles[i], u_ω, u_a, g, dt) * dt + noise[i]
    return particles

def particle_filter_update(particles, z, R_meas, weights):
    predicted_measurements = np.array([measurement_model(p) for p in particles])
    likelihood = np.exp(-0.5 * np.sum((predicted_measurements - z)**2 / np.diag(R_meas), axis=1))
    weights *= likelihood
    weights /= np.sum(weights)
    return weights

def low_variance_resampling(particles, weights):
    cumulative_weights = np.cumsum(weights)
    random_values = np.random.rand(weights.size)
    indices = np.searchsorted(cumulative_weights, random_values)
    return particles[indices]

# -------------------- Visualization --------------------
def plot_trajectory(estimated_positions, true_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(true_positions[0], true_positions[1], true_positions[2], label='Ground Truth')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimate')
    ax.legend()
    plt.show()

def plot_euler_angles(estimated_orientations, true_orientations):
    angles = ['Roll', 'Pitch', 'Yaw']
    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(true_orientations[-1], true_orientations[i], label='Ground Truth')
        plt.plot(estimated_orientations[:, i + 3], label='Estimate')
        plt.ylabel(angles[i])
        plt.legend()
    plt.xlabel('Time')
    plt.show()

# -------------------- Main Execution --------------------
camera_matrix = np.array([[314.1779, 0, 199.4848], [0, 314.2218, 113.7838], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float32)
tag_corners_world = generate_tag_corners()
data_folder = "/home/camilo/dev/RBE_595_ARN/data"

estimated_positions, true_positions, true_orientations = process_data(data_folder, camera_matrix, dist_coeffs, tag_corners_world)

for t in range(len(estimated_positions)):
    dt = 0.01  # Assuming a fixed timestep or based on actual timestamps
    u_ω = compute_angular_velocity(estimated_all, dt)[t]
    u_a = compute_linear_acceleration(estimated_positions, dt)[t]

    state_pred, P_pred = ekf_predict(state, P, u_ω, u_a, g, dt)
    state, P = ekf_update(state_pred, P_pred, estimated_positions[t], R_meas)
    particles = particle_filter_predict(particles, u_ω, u_a, g, dt, Q)
    weights = particle_filter_update(particles, estimated_positions[t], R_meas, weights)
    particles = low_variance_resampling(particles, weights)

    if t % 10 == 0:
        plot_trajectory(estimated_positions, true_positions)
        plot_euler_angles(estimated_positions, true_orientations)