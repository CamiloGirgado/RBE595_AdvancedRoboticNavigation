import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Assuming EKF SLAM setup, motion model, and measurements already implemented as discussed

# Read and preprocess the data
data_in = pd.read_csv('victoria_park.csv', index_col=0)
data_in.index = pd.to_timedelta(data_in.index, unit='s')
data_in = data_in.resample('1s').mean()

# Extract relevant columns
time = data_in.index.total_seconds()
speed = data_in['speed']
steering = data_in['steering_angle']
lat = data_in['latitude']
lon = data_in['longitude']
lidar_x = data_in.filter(regex='laser_x')

# Assuming ground truth is in local Cartesian coordinates as x and y
x_gt = data_in['x_position']  # Ground truth x-position
y_gt = data_in['y_position']  # Ground truth y-position

# Initialize EKF SLAM parameters, state, covariance, etc.

# Vehicle State (x, y, yaw)
state = np.zeros(3)  # [x, y, yaw]
P = np.eye(3)  # Initial state covariance

# For demonstration, assume landmarks and lidar data processed elsewhere
# You would include prediction and update steps as per EKF SLAM process

# Initialize lists for storing estimated positions
x_est = []
y_est = []

# Run the EKF loop (simplified version here)
for t in range(1, len(time)):
    # Predict step
    v = speed[t]  # linear velocity
    delta = steering[t]  # steering angle
    dt = time[t] - time[t-1]  # time step
    
    # Apply motion model (simple example)
    x = state[0] + v * np.cos(state[2]) * dt
    y = state[1] + v * np.sin(state[2]) * dt
    yaw = state[2] + (v / 2.83) * np.tan(delta) * dt  # L=2.83m for vehicle model
    
    state = np.array([x, y, yaw])  # Updated state
    
    # Store estimated positions
    x_est.append(x)
    y_est.append(y)
    
    # Update step (would use lidar and GPS measurements here)

# Convert to numpy arrays for easier processing
x_est = np.array(x_est)
y_est = np.array(y_est)

# Calculate RMSE (Root Mean Square Error) in both x and y directions
rmse_x = np.sqrt(mean_squared_error(x_gt, x_est))
rmse_y = np.sqrt(mean_squared_error(y_gt, y_est))

print(f"RMSE in X: {rmse_x:.2f} meters")
print(f"RMSE in Y: {rmse_y:.2f} meters")

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(x_gt, y_gt, label="Ground Truth", color='blue')
plt.plot(x_est, y_est, label="EKF Estimated", color='red')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Vehicle Trajectory: Ground Truth vs EKF Estimated')
plt.legend()
plt.grid(True)
plt.show()

# Plot landmarks (if applicable)
# Assuming landmarks are stored in a list or array as (x, y) positions
landmarks = np.array([[-20, 30], [50, 100], [100, -50]])  # Example landmarks

plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='green', alpha=0.5, label='Landmarks')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Map of Landmarks')
plt.legend()
plt.grid(True)
plt.show()
