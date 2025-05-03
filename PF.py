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
