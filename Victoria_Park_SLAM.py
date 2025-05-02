import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from haversine import haversine, Unit 

# Load Dataset
data = pd.read_csv("victoria_park.csv", index_col=0)

# Convert index to timedelta and resample to 1-second intervals
data.index = pd.to_timedelta(data.index, unit='s')
data = data.resample('1s').mean()

# Convert GPS coordinates to Cartesian frame
def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    return haversine((lat, lon), (ref_lat, ref_lon), unit=Unit.METERS)

ref_lat, ref_lon = data.iloc[0]['latitude'], data.iloc[0]['longitude']
data['x'] = data.apply(lambda row: latlon_to_xy(row['latitude'], ref_lon, ref_lat, ref_lon), axis=1)
data['y'] = data.apply(lambda row: latlon_to_xy(ref_lat, row['longitude'], ref_lat, ref_lon), axis=1)

# Define Motion Model (Ackermann Steering)
def motion_model(state, control, dt):
    x, y, psi = state
    v, delta = control
    L = 2.83  # Vehicle wheelbase
    
    x_next = x + v * np.cos(psi) * dt
    y_next = y + v * np.sin(psi) * dt
    psi_next = psi + (v / L) * np.tan(delta) * dt
    return np.array([x_next, y_next, psi_next])

# EKF SLAM Setup
class EKF_SLAM:
    def __init__(self, num_landmarks):
        self.num_landmarks = num_landmarks
        self.dim_x = 3 + 2 * num_landmarks
        self.dim_z = 2 * num_landmarks
        self.filter = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.filter.Q = Q_discrete_white_noise(dim=self.dim_x, dt=0.1, var=0.1)
        self.filter.P *= 10  # Initial uncertainty

    def predict(self, control, dt):
        self.filter.x[:3] = motion_model(self.filter.x[:3], control, dt)
        self.filter.predict()
    
    def update(self, measurements):
        self.filter.update(measurements)

# Initialize SLAM
slam = EKF_SLAM(num_landmarks=20)
vehicle_positions = []

# Run SLAM
for i in range(len(data)):
    control = [data.iloc[i]['speed'], data.iloc[i]['steering']]
    slam.predict(control, dt=0.1)
    measurements = data.iloc[i][['laser_x', 'laser_y']].values
    slam.update(measurements)
    vehicle_positions.append(slam.filter.x[:3])

# Plot Results
vehicle_positions = np.array(vehicle_positions)
plt.plot(vehicle_positions[:, 0], vehicle_positions[:, 1], label="Estimated Path")
plt.scatter(data['x'], data['y'], color='red', label="GPS Path")
plt.legend()
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.title("SLAM Vehicle Path Estimation")
plt.show()