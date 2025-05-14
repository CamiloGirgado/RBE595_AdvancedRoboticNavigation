import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm, multivariate_normal
from scipy.spatial.transform import Rotation as R

# -------------------- Particle Filter --------------------
class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        # self.R = self.P_Matrix()
        self.mean_init = np.zeros(15)
        # self.cov_init = np.eye(15)*0.0015
        # self.particles = self.init_state_sampler(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
        self.process_model = self.fx

        sigma_bg_x = 0.01
        sigma_bg_y = 0.01
        sigma_bg_z = 0.01
        sigma_ba_x = 0.01
        sigma_ba_y = 0.01
        sigma_ba_z = 0.01
        self.Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])    # OMG bias noise covariance
        self.Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])    # ACC bias noise covariance
        # self.Nbg = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)
        # self.Nba = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa)

        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

        self.R = np.diag([
            .001, .001, .001,           # position (large if starting with vague initial pose)
            np.deg2rad(75.0)**2,np.deg2rad(75.0)**2, np.deg2rad(75.0)**2           # orientation (low confidence)
        ])
        self.P = np.diag([
            .15, .15, .15,           # position (large if starting with vague initial pose)
            np.deg2rad(90.0)**2, np.deg2rad(90.0)**2, np.deg2rad(90.0)**2,           # orientation (in radians; small if initial orientation is known)
            0.1, 0.1, 0.1,           # velocity (moderate confidence)
            
            0.01, 0.01, 0.01,      # gyro bias (smaller drift uncertainty if high-quality IMU)
            0.1, 0.1, 0.1        # accel bias (usually start near zero bias with modest uncertainty)
        ])
        self.particles = self.init_state_sampler(num_particles, state_dim)

        # self.measurement_cov = np.eye(6)*0.001
        # self.measurement_cov[0, 0] = 0.001
        # self.measurement_cov[1, 1] = 0.001
        # self.measurement_cov[2, 2] = 0.001
        # self.measurement_cov[3, 3] = 0.01
        # self.measurement_cov[4, 4] = 0.01
        # self.measurement_cov[5, 5] = 0.01
    
    def particle_filter_predict(self, dt=1, control_input=None):
        for i in range(self.num_particles):
            self.particles[i] = self.process_model(self.particles[i], dt, control_input)

    def particle_filter_update(self, measurement):
        # mvn = multivariate_normal( cov=self.measurement_cov )
        mvn = multivariate_normal(mean=np.zeros(6), cov=self.R )
        self.particles[:,3:6]=np.arctan2( np.sin(self.particles[:,3:6]),np.cos(self.particles[:,3:6]) )
        measurement[3:6] = np.arctan2( np.sin(measurement[3:6]),np.cos(measurement[3:6]) )
        pos_and_orientation_diff = self.particles[:, 0:6] - measurement[0:6].T # position/orientation error for each particle
        self.weights = mvn.pdf(pos_and_orientation_diff)
        self.weights += 1e-300  # prevent divide by zero
        self.weights /= np.sum(self.weights)

        # measured_position = measurement[0:3].T
        # measured_orientation = measurement[3:6].T 
        # position_difference = self.particles[:, 0:3] - measured_position
        # orientation_difference = self.particles[:, 3:6] - measured_orientation

        # # Noise Standard Deviation
        # position_std_deviation = np.array([0.001, 0.001, 0.001])
        # orientation_std_deviation = np.array([0.001, 0.001, 0.001])

        # # Normalized Errors
        # position_error = (position_difference / position_std_deviation) ** 2
        # orientation_error = (orientation_difference / orientation_std_deviation) **2

        # # Total Error
        # total_error = np.sum(position_error, axis=1) + np.sum(orientation_error, axis=1)

        # probability = np.exp(-0.5 * total_error)
        # new_weights = probability * self.weights

        # # Avoid all weights becoming zero (in case of extreme outlier measurement)
        # if np.all(new_weights == 0):
        #     new_weights = np.ones_like(new_weights) * 1e-12
        
        # # Normalize weights to sum to 1
        # new_weights = new_weights / np.sum(new_weights)
        # self.weights = new_weights

    def sampling(self):
        # indices = np.random.choice(
        #     self.num_particles, size=self.num_particles, p=self.weights
        # )
        N_eff = 1. / np.sum(self.weights**2)
        if N_eff < self.num_particles / 2:
            indices = np.random.choice(
                self.num_particles, size=self.num_particles, p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)
            # self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        all = np.average(self.particles, weights=self.weights, axis=0)

        sin_roll  = np.average(np.sin(self.particles[:, 3]), weights=self.weights)
        cos_roll  = np.average(np.cos(self.particles[:, 3]), weights=self.weights)
        sin_pitch = np.average(np.sin(self.particles[:, 4]), weights=self.weights)
        cos_pitch = np.average(np.cos(self.particles[:, 4]), weights=self.weights)
        sin_yaw   = np.average(np.sin(self.particles[:, 5]), weights=self.weights)
        cos_yaw   = np.average(np.cos(self.particles[:, 5]), weights=self.weights)
        avg_roll  = np.arctan2(sin_roll, cos_roll)
        avg_pitch = np.arctan2(sin_pitch, cos_pitch)
        avg_yaw   = np.arctan2(sin_yaw, cos_yaw)
        all[3:6] = np.array([avg_roll, avg_pitch, avg_yaw])

        return all
    
    def init_state_sampler(self, num_particles, state_dim):
        # return np.random.uniform(-2, 2, (num_particles, state_dim))
        # particles = np.random.multivariate_normal(self.mean_init, self.cov_init, size=num_particles)
        particles = np.random.multivariate_normal(self.mean_init, self.P, size=num_particles)
        return particles

    def fx(self, x, dt, data):
        xout = x.copy()
        if data.get('omg') is None or data.get('acc') is None:
            return xout
        omg_bias = x[9:12]
        acc_bias = x[12:15]
        xout[9:12] += np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)*dt
        xout[12:15] += np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa)*dt
        # updated_omg_bias = omg_bias + np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)*dt
        # updated_acc_bias = acc_bias + np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa)*dt
        
        # xout[9:12] = updated_omg_bias
        # xout[12:15] = updated_acc_bias

        G_matrix = self.G_Matrix(x[3:6])
        U_w = (np.array([data['omg']]) + omg_bias).T
        q_dot = np.linalg.inv(G_matrix) @ U_w
        xout[3:6] = q_dot.squeeze()*dt
        xout[3:6] = np.arctan2(np.sin(xout[3:6]), np.cos(xout[3:6]))

        U_a = (np.array([data['acc']]) + acc_bias).T
        Rq_matrix = self.Rq_matrix(x[3:6])
        g = np.array([[0, 0, 9.81]]).T
        xout[6:9] = (Rq_matrix.T @ U_a - g).squeeze()*dt

        v = np.random.multivariate_normal(mean=np.zeros(3), cov=np.diag([.75,.75,.75]))*dt
        # xout[0] = (x[6] * dt + x[0]) + x[9]
        # xout[1] = (x[7] * dt + x[1]) + x[10]
        # xout[2] = (x[8] * dt + x[2]) + x[11]

        xout[0] = (x[6] * dt + x[0]) +v[0]
        xout[1] = (x[7] * dt + x[1]) +v[1]
        xout[2] = (x[8] * dt + x[2]) +v[2]

        # xout[0] = x[6]*dt + x[0]
        # xout[1] = x[7]*dt + x[1]
        # xout[2] = x[8]*dt + x[2]
        
        return xout
    
    def Rq_matrix(self, rpy):
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        # self.r = rotation_z @ rotation_y @ rotation_x
        r = rotation_x @ rotation_y @ rotation_z
        return r
        
    def G_Matrix(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        return np.matrix([
            [np.cos(pitch), 0, -np.sin(roll)*np.cos(pitch)],
            [0, 1, np.sin(roll)],
            [np.sin(pitch), 0, np.cos(roll)*np.cos(pitch)],
            ])