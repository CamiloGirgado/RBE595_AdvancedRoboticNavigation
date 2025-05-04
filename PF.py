import numpy as np
from scipy.stats import multivariate_normal

# -------------------- Particle Filter --------------------
class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_model, measurement_model, init_state_sampler):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = init_state_sampler(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
        self.process_model = process_model
        self.measurement_model = measurement_model
    
    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    def measurement_model(self, x, measurement):
        mm = x[0:6]
        prob = multivariate_normal.pdf(measurement, mean=mm, cov=self.R_meas)
        return prob

    def fx(self, x, dt, data):
        U_w = (np.array([data['omg']])).T
        U_a = (np.array([data['acc']])).T
        xout = x.copy()
        xout[0] = x[6]*dt + x[0]
        xout[1] = x[7]*dt + x[1]
        xout[2] = x[8]*dt + x[2]
        G_matrix = self.G_Matrix(x[3:6])
        q_dot = np.linalg.inv(G_matrix) @ U_w
        xout[3:6] = q_dot.squeeze()
        Rq_matrix = self.Rq_matrix(data[['rpy']])
        xout[6:9] = (Rq_matrix @ U_a + g).squeeze()
        
        return xout
    
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

    def low_variance_resampling(self):
        indices = np.searchsorted(self.num_particles, size = self.num_particles p =np.cumsum(self.weights))
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        return particles[indices]
