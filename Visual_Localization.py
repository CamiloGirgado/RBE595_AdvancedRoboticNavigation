import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from matlab_data import matlab_data
from observationModel_1 import observationModel
from UKF import UKF
from updatedUKF import UkfFilter2
from PF import ParticleFilter

class VisualLocalization:
    def __init__(self, file):
        self.data = matlab_data(file).get_data()
        # self.mat_contents = None
        self.actual_vicon_np = None
        self.results_np = None
        self.actual_vicon_aligned_np = None
        self.diff_matrix = None
        self.cov_matrix = None
        # self.loadMatlabData(file)
        self.file = file
        # self.ukf_or_particle_filter = "UKF"
        self.particle_count = 0
        self.pf = None
        self.results_filtered_np = None
        self.x = None
        self.time = None
        self.ax = None
        self.fig = None
        self.axs = None
        # self.true_positions = None
        self.actual_vicon_np = np.vstack((self.data['vicon'], np.array([self.data['time']])))
    
    def run_UKF(self):
        observationModel_1 = observationModel()
        position = None
        ukf = UkfFilter2(self.data)
        self.time = []
        self.results_np = None
        self.results_filtered_np = None
        self.x = np.zeros((15,1))
        is_initialized = False
        dt = 0.
        time_last = 0.
        for data in self.data['data']:
            if isinstance(data['id'],np.ndarray):
                # This has no April tags found in the image
                if len(data['id']) == 0:
                    continue
            # Estimate the pose for each item in the data
            position,orientation = observationModel_1.estimate_pose(data)  # Estimate the pose for each item in the data   

            if position is None or orientation is None:
                print("Warning: Pose estimation failed for the current data item. Skipping this item.")
                continue  # Skip this item if pose estimation failed
            dt = data['t'] - time_last
            time_last = data['t']
            # if data.get('omg') is None:
            #     continue
            filtered_state_x = ukf.predict(dt,data)
            
            z = np.hstack((np.array(position).T,orientation))
            filtered_state_x = ukf.update(z.T)
            result = np.hstack((np.array(position).T,orientation))
            result = np.hstack((result, np.array([data['t']])))
            filtered_state_x = np.hstack((filtered_state_x, np.array([data['t']])))
            # filtered_state_x = np.hstack((filtered_state_x, np.full((filtered_state_x.shape[0], 1), data['t'])))
            # filtered_state_x = np.hstack((filtered_state_x, np.array([data['t']]).reshape(-1, 1)))
            # filtered_state_x = np.hstack((filtered_state_x, np.array([data['t']])))
            # result= np.hstack((np.array(position).squeeze(),orientation,data['t']))
            self.results_np = result if self.results_np is None else np.vstack((self.results_np, result))
            self.results_filtered_np = filtered_state_x if self.results_filtered_np is None else np.vstack((self.results_filtered_np, filtered_state_x))
        return self.results_np

    def run_PF(self,particle_count=0):
        self.ukf_or_particle_filter = "ParticleFilter"
        self.particle_count = particle_count
        self.pf = ParticleFilter(
            num_particles = particle_count,
            state_dim=15
        )
        observationModel_1 = observationModel()
        position = None
        self.time = []
        self.results_np = None
        self.results_filtered_np = None
        self.x = np.zeros((15,1))
        is_initialized = False
        dt = 0.
        time_last = 0.
        for data in self.data['data']:
            if isinstance(data['id'],np.ndarray):
                # This has no April tags found in the image
                if len(data['id']) == 0:
                    continue
            # Estimate the pose for each item in the data
            position,orientation = observationModel_1.estimate_pose(data)  # Estimate the pose for each item in the data   
           
            if position is None or orientation is None:
                print("Warning: Pose estimation failed for the current data item. Skipping this item.")
                continue  # Skip this item if pose estimation failed
            dt = data['t'] - time_last
            if not is_initialized:
                dt = 0.001
                self.pf.particles[:,0:3] = np.tile(position.reshape(3,1), self.pf.num_particles).T
                self.pf.particles[:,3:6] = np.tile(orientation.reshape(3,1), self.pf.num_particles).T
                is_initialized = True
                # self.pf.particles[:,9:12] = np.array([[0.0001,0.0001,0.0001]]).T
                # self.pf.particles[:,12:15] = np.array([[0.0001,0.0001,0.0001]]).T    
            time_last = data['t']

            self.pf.particle_filter_predict(dt,data)
            z = np.hstack((np.array(position).T,orientation))
            self.pf.particle_filter_update(z.T)
            self.pf.sampling()
            filtered_state_x = self.pf.estimate()
            result = np.hstack((np.array(position).T,orientation))
            result = np.hstack((result, np.array([data['t']])))
            filtered_state_x = np.hstack((filtered_state_x, np.array([data['t']])))
            # result= np.hstack((np.array(position).squeeze(),orientation,data['t']))
            self.results_np = result if self.results_np is None else np.vstack((self.results_np, result))
            self.results_filtered_np= filtered_state_x if self.results_filtered_np is None else np.vstack((self.results_filtered_np, filtered_state_x))
        

    def process_data(self, directory, camera_matrix, dist_coeffs, tag_corners_world):
        estimated_positions = []
        estimated_orientations = []
        true_positions = []
        true_orientations = []

        # List all MAT files in the folder
        mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]

        if not mat_files:
            print("No .mat files found in the directory!")
            return None, None, None, None
        mat_files = ['studentdata7.mat']
        for file_name in mat_files:
            file_path = os.path.join(directory, file_name)
            print(f"Loading file: {file_name}")
            data = sio.io.loadmat(file_path, simplify_cells=True)
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
                position, orientation = VisualLocalization.estimate_pose(entry, camera_matrix, dist_coeffs, tag_corners_world)
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
        return estimated_all

    def plot_trajectory(self):
        self.__plot_trajectory_vicon__()
        self.__plot_trajectory_estimated__()
        self.__plot_trajectory_estimated_filtered__()

    def __plot_trajectory_vicon__(self):
        x = self.actual_vicon_np[0, :]
        y = self.actual_vicon_np[1, :]
        z = self.actual_vicon_np[2, :]

        # Plot the trajectory
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        self.ax.plot(x, y, z, label='Actual', color='c', linewidth=2)  # Set color and linewidth for better visibility

        # ax.plot(true_positions[0, :], true_positions[1, :], true_positions[2, :], label='Ground Truth')
        # ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated')
        self.ax.legend()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.title('Trajectory Comparison')

    
    def __plot_trajectory_estimated__(self):
        x = self.results_np.T.squeeze()[0,:]
        y = self.results_np.T.squeeze()[1,:]
        z = self.results_np.T.squeeze()[2,:]
        
        # Plot the trajectory
        self.ax.plot(x, y, z, label='Estimated', color='m', linewidth=1, linestyle='-' ) 
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.set_title('3D Trajectory Plot')
        self.ax.legend()

    def __plot_trajectory_estimated_filtered__(self):

        x = self.results_filtered_np.T.squeeze()[0,:]
        y = self.results_filtered_np.T.squeeze()[1,:]
        z = self.results_filtered_np.T.squeeze()[2,:]
        
        # Plot the trajectory
        self.ax.plot(x, y, z, label='Filtered Estimate', color='y', linewidth=1, linestyle='-' )
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.set_title('3D Trajectory Plot')
        self.ax.legend()
        plt.show()

    def plot_orientation(self):
        
        roll  = self.actual_vicon_np[3, :]
        pitch = self.actual_vicon_np[4, :]
        yaw   = self.actual_vicon_np[5, :]
        x = self.actual_vicon_np[-1, :]
        
        
        # Plot the trajectory
        self.fig, self.axs = plt.subplots(3, 1, figsize=(16, 16))
        self.fig.suptitle('Roll / Pitch / Yaw Plot')

        
        # Plot the trajectory
        self.axs[0].plot(x, roll, label='Actual', color='c', linewidth=1)  # Set color and linewidth for better visibility
        self.axs[0].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[3,:], label='Estimated', color='m', linewidth=1)  # Set color and linewidth for better visibility
        self.axs[0].plot(self.results_filtered_np.T.squeeze()[15,:], self.results_filtered_np.T.squeeze()[3,:], label='Filtered', color='y', linewidth=1)  # Set color and linewidth for better visibility

        self.axs[1].plot(x, pitch, label='Actual', color='c', linewidth=1)  # Set color and linewidth for better visibility
        self.axs[1].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[4,:], label='Estimated', color='m', linewidth=1)  # Set color and linewidth for better visibility
        self.axs[1].plot(self.results_filtered_np.T.squeeze()[15,:], self.results_filtered_np.T.squeeze()[4,:], label='Filtered', color='y', linewidth=1)  # Set color and linewidth for better visibility

        self.axs[2].plot(x, yaw, label='Actual results_filtered_', color='c', linewidth=1)  # Set color and linewidth for better visibility
        self.axs[2].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[5,:], label='Estimated', color='m', linewidth=1)  # Set color and linewidth for better visibility
        self.axs[2].plot(self.results_filtered_np.T.squeeze()[15,:], self.results_filtered_np.T.squeeze()[5,:], label='Filtered', color='y', linewidth=1)  # Set color and linewidth for better visibility
        
        # Set labels and title
        self.axs[0].set_xlabel('Time')
        self.axs[0].set_ylabel('Roll (rad)')
        self.axs[0].set_title('Roll Plot')
        self.axs[0].legend()
        self.axs[1].set_xlabel('Time')
        self.axs[1].set_ylabel('Pitch (rad)')
        self.axs[1].set_title('Pitch Plot')
        self.axs[1].legend()
        self.axs[2].set_xlabel('Time')
        self.axs[2].set_ylabel('Yaw (rad)')
        self.axs[2].set_title('Yaw Plot')    
        self.axs[2].legend()
        plt.subplots_adjust(wspace=0.4, hspace=0.6) # Adjust values as needed
        plt.show()


    def interpolate(self, time_target, t1, t2,y1, y2):
        # Interpolate between two points.
        interpolated_data = y1 + ((time_target - t1) * (y2 - y1) / (t2 - t1))
        return interpolated_data
    
    def interpolate_data(estimated_all, true_positions):
        # Interpolate estimated data to align with true positions."""
        if estimated_all is None or len(estimated_all) == 0:
            return None

        
        interpolated_data = []
        
        for estimated_data in estimated_all:
            # Find the closest estimated timestamp
            estimated_time = float(estimated_data[-1])
            closest_idx = np.argmin(true_positions[-1, :]< estimated_time)
            interpolated_x = VisualLocalization.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[0,closest_idx-1], true_positions[0,closest_idx])
            interpolated_y = VisualLocalization.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[1,closest_idx-1], true_positions[1,closest_idx])
            interpolated_z = VisualLocalization.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[2,closest_idx-1], true_positions[2,closest_idx])    
            interpolated_roll = VisualLocalization.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[3,closest_idx-1], true_positions[3,closest_idx])    
            interpolated_pitch = VisualLocalization.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[4,closest_idx-1], true_positions[4,closest_idx])    
            interpolated_yaw = VisualLocalization.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[5,closest_idx-1], true_positions[5,closest_idx])    
            interpolated_data.append(np.array([interpolated_x, interpolated_y, interpolated_z, interpolated_roll, interpolated_pitch, interpolated_yaw, estimated_time]))

        return np.array(interpolated_data) if interpolated_data else None


    def compute_covariance(estimated_all, true_positions, true_orientations):
        # Computes the covariance matrix of the observation noise.

        true_data = np.vstack((true_positions, true_orientations))

        interpolated_data = VisualLocalization.interpolate_data(estimated_all, true_data)
        if interpolated_data is None:
            return None
        print("Interpolated Data Shape:", interpolated_data.shape)
        print("Estimated Data Shape:", estimated_all.shape)
        
        diff_matrix = interpolated_data[:, :-1] - estimated_all[:, :-1]
        R_matrix = np.cov(diff_matrix.T)
        # Check if covariance matrix is symetric
        if not np.allclose(R_matrix, R_matrix.T):
            print("Covariance matrix is not symmetric!")
        else:
            print("Covariance matrix is symmetric!")
        # Check if covariance matrix is positive definite
        if np.any(np.linalg.eigvals(R_matrix) > 0):
            print("Covariance matrix is positive definite!")
        else:
            print("Covariance matrix is not positive definite!")

        return R_matrix

    def rmse(self):
            if self.results_filtered_np is None:
                print("No results available to calculate RSME.")
                return None
            
            self.actual_vicon_aligned_np = None
            for idx,x_measurement_model in enumerate(self.results_filtered_np[:, -1]):
                x = float(x_measurement_model)
                min_idx = np.argmin(self.actual_vicon_np[-1,:] < x)
                if min_idx == 0:
                    continue
                if min_idx == self.actual_vicon_np[-1,:].shape[0]-1:
                    min_idx = min_idx-1
                x_interpolated = self.interpolate(x,
                        self.actual_vicon_np[-1,min_idx],
                        self.actual_vicon_np[-1,min_idx+1],
                        self.actual_vicon_np[0,min_idx],
                        self.actual_vicon_np[0,min_idx+1])
                y_interpolated = self.interpolate(x,
                    self.actual_vicon_np[-1,min_idx],
                    self.actual_vicon_np[-1,min_idx+1],
                    self.actual_vicon_np[1,min_idx],
                    self.actual_vicon_np[1,min_idx+1])
                z_interpolated = self.interpolate(x,
                    self.actual_vicon_np[-1,min_idx],
                    self.actual_vicon_np[-1,min_idx+1],
                    self.actual_vicon_np[2,min_idx],
                    self.actual_vicon_np[2,min_idx+1])
                roll_interpolated = self.interpolate(x,
                    self.actual_vicon_np[-1,min_idx],
                    self.actual_vicon_np[-1,min_idx+1],
                    self.actual_vicon_np[3,min_idx],
                    self.actual_vicon_np[3,min_idx+1])
                pitch_interpolated = self.interpolate(x,
                    self.actual_vicon_np[-1,min_idx],
                    self.actual_vicon_np[-1,min_idx+1],
                    self.actual_vicon_np[4,min_idx],
                    self.actual_vicon_np[4,min_idx+1])
                yaw_interpolated = self.interpolate(x,
                    self.actual_vicon_np[-1,min_idx],
                    self.actual_vicon_np[-1,min_idx+1],
                    self.actual_vicon_np[5,min_idx],
                    self.actual_vicon_np[5,min_idx+1])
                new_row = [x_interpolated,y_interpolated,z_interpolated,
                    roll_interpolated,
                    pitch_interpolated,
                    yaw_interpolated,x]
                
                self.actual_vicon_aligned_np = new_row if self.actual_vicon_aligned_np is None \
                    else np.vstack((self.actual_vicon_aligned_np,new_row))
            max_idx = min(self.actual_vicon_aligned_np.shape[0], self.results_np.shape[0])
            
            self.diff_matrix_estimated = self.actual_vicon_aligned_np.T[0:3,:max_idx] - self.results_np.T.squeeze()[0:3,:max_idx] 
            distance_sum = 0
            for x in range(self.diff_matrix_estimated.T.shape[0]):
                distance_sum += (self.diff_matrix_estimated.T[x,0]**2 + 
                                self.diff_matrix_estimated.T[x,1]**2 + 
                                self.diff_matrix_estimated.T[x,2]**2)**0.5
            
            rmse_measurement_model = np.sqrt(distance_sum/self.diff_matrix_estimated.T.shape[0])
            # print("RMSE of measurement model: ", rmse_measurement_model)


            self.diff_matrix_estimated = self.actual_vicon_aligned_np.T[0:3,:max_idx] - self.results_filtered_np.T.squeeze()[0:3,:max_idx] 
            distance_sum = 0
            for x in range(self.diff_matrix_estimated.T.shape[0]):
                distance_sum += (self.diff_matrix_estimated.T[x,0]**2 + 
                                self.diff_matrix_estimated.T[x,1]**2 + 
                                self.diff_matrix_estimated.T[x,2]**2)**0.5
            
            rmse_filtered = np.sqrt(distance_sum/self.diff_matrix_estimated.T.shape[0])
            # print("RMSE of Filtered: ", rmse_filtered)
            rmse_difference = rmse_measurement_model - rmse_filtered
            print("RMSE difference: ", rmse_difference)
            return rmse_difference, rmse_measurement_model

