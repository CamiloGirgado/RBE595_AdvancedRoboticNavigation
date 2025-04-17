import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Initialization ----------------------------

# State vector: [position, orientation, velocity, gyroscope bias, accelerometer bias]
# 15-state: [px, py, pz, φ, θ, ψ, ṗx, ṗy, ṗz, bgx, bgy, bgz, bax, bay, baz]
state = np.zeros(15)

# Process noise covariance (initial guess)
Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

# Measurement noise covariance (based on previous assignment)
R = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # Assuming position (3) and orientation (3) measurements

# Initial uncertainty (covariance matrix for state vector)
P = np.eye(15) * 0.1  # Start with some uncertainty in the initial state estimate

# Particle filter setup
num_particles = 1000
particles = np.random.randn(num_particles, 15)  # Initial particle distribution
weights = np.ones(num_particles) / num_particles  # Equal initial weights

# ---------------------- Process Model (State Transition) ----------------------

def process_model(state, u_ω, u_a, g, dt):
    p = state[:3]  # position
    q = state[3:6]  # orientation (Euler angles: φ, θ, ψ)
    v = state[6:9]  # velocity
    bg = state[9:12]  # gyroscope bias
    ba = state[12:]  # accelerometer bias

    # Rotation matrix from Euler angles
    R = np.array([
        [np.cos(q[2]) * np.cos(q[1]), np.cos(q[2]) * np.sin(q[1]) * np.sin(q[0]) - np.sin(q[2]) * np.cos(q[0]), np.cos(q[2]) * np.sin(q[1]) * np.cos(q[0]) + np.sin(q[2]) * np.sin(q[0])],
        [np.sin(q[2]) * np.cos(q[1]), np.sin(q[2]) * np.sin(q[1]) * np.sin(q[0]) + np.cos(q[2]) * np.cos(q[0]), np.sin(q[2]) * np.sin(q[1]) * np.cos(q[0]) - np.cos(q[2]) * np.sin(q[0])],
        [-np.sin(q[1]), np.cos(q[1]) * np.sin(q[0]), np.cos(q[1]) * np.cos(q[0])]
    ])

    # Process model
    p_dot = v  # velocity is the derivative of position
    v_dot = R.dot(u_a - ba) - g  # acceleration model (subtract gravity, apply accelerometer bias)
    bg_dot = np.zeros(3)  # Gyroscope bias drift (assumed to be small noise)
    ba_dot = np.zeros(3)  # Accelerometer bias drift (assumed to be small noise)

    # Return state derivative
    state_dot = np.concatenate([p_dot, v_dot, bg_dot, ba_dot])

    return state_dot

# -------------------------- Measurement Model ----------------------------

def measurement_model(state):
    p = state[:3]  # position
    q = state[3:6]  # orientation
    z = np.concatenate([p, q])  # measurement vector (position and orientation)
    return z

# ------------------------- Nonlinear Kalman Filter (EKF) ---------------------

def ekf_predict(state, P, u_ω, u_a, g, dt):
    # Predict state based on the process model
    state_dot = process_model(state, u_ω, u_a, g, dt)
    state_pred = state + state_dot * dt  # Update state with Euler integration

    # Calculate the Jacobian of the process model (for EKF)
    F = np.eye(15)  # Jacobian of the process model (to be computed based on state transition equations)

    # Update covariance
    P_pred = F @ P @ F.T + Q  # Predict covariance

    return state_pred, P_pred

def ekf_update(state_pred, P_pred, z, R):
    # Update state using the measurement model
    H = np.eye(6, 15)  # Measurement Jacobian (3 position, 3 orientation)
    z_pred = measurement_model(state_pred)  # Predicted measurement

    # Compute Kalman Gain
    S = H @ P_pred @ H.T + R  # Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

    # Update state
    y = z - z_pred  # Innovation (measurement residual)
    state_updated = state_pred + K @ y
    P_updated = (np.eye(15) - K @ H) @ P_pred

    return state_updated, P_updated

# -------------------------- Particle Filter -------------------------------

def particle_filter_predict(particles, u_ω, u_a, g, dt, Q):
    # Predict the new state for each particle
    num_particles = particles.shape[0]
    predicted_particles = particles.copy()

    for i in range(num_particles):
        predicted_particles[i, :] += process_model(particles[i, :], u_ω, u_a, g, dt) * dt
        noise = np.random.multivariate_normal(np.zeros(15), Q)
        predicted_particles[i, :] += noise

    return predicted_particles

def particle_filter_update(particles, z, R, weights):
    # Update particle weights based on the measurement model
    num_particles = particles.shape[0]
    predicted_measurements = np.array([measurement_model(p) for p in particles])

    # Calculate likelihood of each particle based on the measurement
    likelihood = np.exp(-0.5 * np.sum((predicted_measurements - z)**2 / np.diag(R), axis=1))
    weights = weights * likelihood
    weights /= np.sum(weights)  # Normalize weights

    return weights

def low_variance_resampling(particles, weights):
    # Resample particles based on their weights
    cumulative_weights = np.cumsum(weights)
    random_values = np.random.rand(particles.shape[0])
    resampled_particles = particles[np.searchsorted(cumulative_weights, random_values)]

    return resampled_particles

# ---------------------------- RMSE Calculation ---------------------------

def rmse(estimated_positions, true_positions):
    return np.sqrt(np.mean(np.sum((estimated_positions - true_positions)**2, axis=1)))

# ---------------------------- Visualization -----------------------------

def plot_trajectory(true_positions, observations, state_estimates, particles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], color='g', label='Ground Truth')
    ax.scatter(observations[:, 0], observations[:, 1], observations[:, 2], color='r', label='Observations')
    ax.scatter(state_estimates[0], state_estimates[1], state_estimates[2], color='b', label='EKF Estimate')
    ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], color='y', label='Particle Filter')
    ax.legend()
    plt.show()

# ------------------------ Main Loop for Filtering -----------------------

# Assuming you have some form of input data
measurements = np.load('measurements.npy')  # Replace with actual data
true_positions = np.load('true_positions.npy')  # Replace with actual data

for t in range(len(measurements)):
    # Predict step for both Kalman filter and Particle filter
    state_pred, P_pred = ekf_predict(state, P, u_ω, u_a, g, dt)
    state_updated, P_updated = ekf_update(state_pred, P_pred, measurements[t], R)
    
    # Particle Filter prediction and update
    particles = particle_filter_predict(particles, u_ω, u_a, g, dt, Q)
    weights = particle_filter_update(particles, measurements[t], R, weights)
    particles = low_variance_resampling(particles, weights)

    # RMSE calculation for position
    rmse_val = rmse(estimated_positions, true_positions)

    # Visualization (3D plot)
    if t % plot_interval == 0:
        plot_trajectory(true_positions, observations, state_updated[:3], particles)
