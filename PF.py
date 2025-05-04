import numpy as np
from scipy.stats import multivariate_normal

# -------------------- Particle Filter --------------------
class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_model, measurement_model, init_state_sampler):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = init_state_sampler(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
        self.process_model = self.fx
        self.measurement_model = self.measurement_model
        sigma_bg_x = 0.2
        sigma_bg_y = 0.2
        sigma_bg_z = 5.5
        sigma_ba_x = 0.2
        sigma_ba_y = 0.2
        sigma_ba_z = 5.5
        self.Q = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2, sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])
        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

    
    def measurement_model(self, x, measurement):
        mm = x[0:6]
        prob = multivariate_normal.pdf(measurement, mean=mm, cov=self.R_meas)
        return prob
    
    def particle_filter_predict(particles, u_ω, u_a, g, dt, Q):
        noise = np.random.multivariate_normal(np.zeros(15), Q, size=particles.shape[0])
        for i in range(particles.shape[0]):
            particles[i] += process_model(particles[i], u_ω, u_a, g, dt) * dt + noise[i]
        return particles

    def particle_filter_update(self, measurement):
        measured_position = measurement[0:3]
        measured_orientation = measurement[3:6]
        position_difference = measured_position - self.particles[:, 0:3]
        orientation_difference = measured_orientation - self.particles[:, 3:6]

        # Noise Standard Deviation
        position_std_deviation = 0.1
        orientation_std_deviation = 0.1

        # Normalized Errors
        position_error = np.linalg.norm(position_difference, axis=1) / position_std_deviation
        orientation_error = np.linalg.norm(orientation_difference, axis=1) / orientation_std_deviation

        # Total Error
        total_error = position_error + orientation_error

    
    def sampling(self):
        self.particles = np.random.multivariate_normal(self.particles, self.Q, size=self.num_particles)
        return self.particles
    
    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    def fx(self, x, dt, data):
        xout = x.copy()
        U_w = (np.array([data['omg']])).T
        U_a = (np.array([data['acc']])).T

        xout[0] = x[6]*dt + x[0]
        xout[1] = x[7]*dt + x[1]
        xout[2] = x[8]*dt + x[2]
        G_matrix = self.G_Matrix(x[3:6])
        q_dot = np.linalg.inv(G_matrix) @ U_w
        xout[3:6] = q_dot.squeeze()
        Rq_matrix = self.Rq_matrix(data[['rpy']])
        xout[6:9] = (Rq_matrix @ U_a + g).squeeze()
        
        omg_bias = x[9:12]
        acc_bias = x[12:15]
        velocity = x[6:9]
        angular_velocity = x[3:6]

        # Orientation Update
        updated_angular_velocity = U_w + omg_bias
        Rotation_vector = updated_angular_velocity * dt
        Rotation = R.from_rotvec(Rotation_vector)
        quaternion = R.from_euler('xyz', x[3:6], degrees=False).as_quat()
        quaternion = np.concatenate((quaternion, [0]))  # Convert to quaternion format
        updated_quaternion = updated_quaternion / np.linalg.norm(updated_quaternion)
        xout[3:6] = R.from_quat(quaternion).as_euler('xyz', degrees=False) + Rotation_vector

        # Update Velocity
        updated_acceleration = linear_acceleration - acc_bias

        g = [0, 0, -9.807]
        updated_velocity = velocity + acc_bias * dt
        corrected_acceleration = np.dot(Rq_matrix, updated_velocity) + g

        # Update Position
        updated_position = xout[0:3] + updated_velocity * dt

        # Bais Update
        xout[9:12] = updated_omg_bias
        xout[12:15] = updated_acc_bias

        return xout
    


    def low_variance_resampling(self):
        indices = np.searchsorted(self.num_particles, size = self.num_particles, p = np.cumsum(self.weights))
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        return particles[indices]
