import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

class observationModel:
    def __init__(self,ax=None, fig=None):
        self.ax = ax
        self.fig = fig
        self.camera_matrix = np.array([
            [314.1779, 0.0, 199.4848],
            [0.0, 314.2218, 113.7838],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float64)
        self.tag_corners_world = observationModel.generate_tag_corners()
        self.actual_vicon_np = None

    def generate_tag_corners():
        """Generates the world coordinates for AprilTag corners in a 12x9 grid."""
        tag_size = 0.152  # Size of each AprilTag
        spacing = 0.152   # Default spacing between tags
    
        # Extra spacing applies between columns 3-4 and 6-7 (0-indexed)
        extra_spacing_cols = {3: 0.178, 6: 0.178}
    
        tag_corners_world = {}
    
        for row in range(12):  # Rows go down (x-direction)
            x = row * (tag_size + spacing)
    
            for col in range(9):  # Columns go right (y-direction)
                y = 0
                for c in range(1, col+1):
                    y += tag_size
                    #if c > 0:
                    if c in extra_spacing_cols.keys():
                        y += extra_spacing_cols[c]
                    else:
                        y += spacing
    
                # Define the four corners in world frame: P1 to P4
                P1 = np.array([x + tag_size, y, 0])            # Bottom-left
                P2 = np.array([x + tag_size, y + tag_size, 0]) # Bottom-right
                P3 = np.array([x, y + tag_size, 0])            # Top-right
                P4 = np.array([x, y, 0])                       # Top-left
    
                tag_id = col * 12 + row  # Row-major order
                tag_corners_world[tag_id] = np.array([P1, P2, P3, P4])
    
        return tag_corners_world

    def estimate_pose(self, data):
        # Debugging: Check structure
        print("Data keys:", data.keys())
        print("ID Type:", type(data['id']))

        obj_points = []
        img_points = []

        if isinstance(data['id'], int): # or len(data['id']) == 0:
            obj_points.append(self.tag_corners_world[data['id']])
            img_points.append(
                    np.array([data['p1'], data['p2'], data['p3'], data['p4']]))
        elif len(data['id']) == 0:
            return None, None
        else:
            for i, tag_id in enumerate(data['id']):
                if tag_id in self.tag_corners_world:
                    obj_points.append(self.tag_corners_world[tag_id])
                    img_points.append(
                        np.array([data['p1'][:, i], data['p2'][:, i], data['p3'][:, i], data['p4'][:, i]]))

        obj_points = np.array(obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.array(img_points, dtype=np.float32).reshape(-1, 2)
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            return None, None
        
        #Convert rotation vector to rotation matrix
        R_cam_to_world, _ = cv2.Rodrigues(rvec)

        # Apply translation vector
        t_camera_to_robot = np.array([-0.04, 0, -0.03]).reshape(3, 1)
        R_x = R.from_euler('x', np.pi).as_matrix()
        R_z = R.from_euler('z', np.pi / 4).as_matrix()

        # Combine rotations
        R_cam_to_robot = R_x @ R_z
        #R_cam_to_robot = R_z @ R_x

        #Convert the camera pose to the drone pose
        R_world_to_robot = (R_cam_to_world) @ R_cam_to_robot 
        #R_world_to_robot =  (R_cam_to_world) @ R_cam_to_robot
        # t_robot = R_cam_to_robot @ tvec + t_camera_to_robot
        t_robot = (-(R_cam_to_world).T @ tvec) + t_camera_to_robot

        # Convert rotation matrix to Euler Angles
        euler_angles = R.from_matrix(R_world_to_robot).as_euler('xyz', degrees = False)

        return t_robot.flatten(), euler_angles

    def process_data(directory, camera_matrix, dist_coeffs, tag_corners_world):
        #Processes all .mat files in the given directory and estimates pose.
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
                position, orientation = observationModel.estimate_pose(entry, camera_matrix, dist_coeffs, tag_corners_world)
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

    def plot_trajectory(estimated_positions, true_positions):
        """Plots the estimated trajectory against the ground truth."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(true_positions[0, :], true_positions[1, :], true_positions[2, :], label='Ground Truth')
        ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Trajectory Comparison')
        plt.show()

    def plot_euler_angles(estimated_orientations, true_orientations):
        """Plots estimated and ground truth Euler angles."""
        angles = ['Roll', 'Pitch', 'Yaw']
        plt.figure()
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            # plt.plot(true_orientations[:, i].T, label='Ground Truth')
            plt.plot(true_orientations[-1, :], true_orientations[i, :], label='Ground Truth')
            plt.plot(estimated_orientations.T[-1, :], estimated_orientations.T[i+3, :], label='Estimated')
            plt.ylabel(angles[i])
            plt.legend()
        plt.xlabel('Time')
        plt.suptitle('Euler Angles Comparison')
        plt.show()

    def interpolate(time_target,t1, t2,y1, y2):
            interpolated_data = y1 + ((time_target - t1) * (y2 - y1) / (t2 - t1))
            return interpolated_data
    
    def interpolate_data(estimated_all, true_positions):
        if estimated_all is None or len(estimated_all) == 0:
            return None

        
        interpolated_data = []
        
        for estimated_data in estimated_all:
            # Find the closest estimated timestamp
            estimated_time = float(estimated_data[-1])
            closest_idx = np.argmin(true_positions[-1, :]< estimated_time)
            interpolated_x = observationModel.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[0,closest_idx-1], true_positions[0,closest_idx])
            interpolated_y = observationModel.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[1,closest_idx-1], true_positions[1,closest_idx])
            interpolated_z = observationModel.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[2,closest_idx-1], true_positions[2,closest_idx])    
            interpolated_roll = observationModel.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[3,closest_idx-1], true_positions[3,closest_idx])    
            interpolated_pitch = observationModel.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[4,closest_idx-1], true_positions[4,closest_idx])    
            interpolated_yaw = observationModel.interpolate(estimated_time, true_positions[-1, closest_idx-1], true_positions[-1, closest_idx], true_positions[5,closest_idx-1], true_positions[5,closest_idx])    
            interpolated_data.append(np.array([interpolated_x, interpolated_y, interpolated_z, interpolated_roll, interpolated_pitch, interpolated_yaw, estimated_time]))

        return np.array(interpolated_data) if interpolated_data else None

    def compute_covariance(estimated_all, true_positions, true_orientations):
        """Computes the covariance matrix of the observation noise."""

        true_data = np.vstack((true_positions, true_orientations))

        interpolated_data = observationModel.interpolate_data(estimated_all, true_data)
        if observationModel.interpolate_data is None:
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

    # Main execution
    #camera_matrix = np.array([
    #       [314.1779, 0, 199.4848],
    #       [0, 314.2218, 113.7838],
    #       [0, 0, 1]
    #   ], dtype=np.float32)

    #dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float32)
    #tag_corners_world = generate_tag_corners()
    #data_folder = "/home/camilo/dev/RBE_595_ARN/data"  # Updated to correct folder path
    #estimated_data, true_positions, true_orientations = process_data(data_folder, camera_matrix, dist_coeffs, tag_corners_world)

    #if estimated_data is not None and estimated_data.size > 0:
    #    plot_trajectory(estimated_data, true_positions)
    #    plot_euler_angles(estimated_data, true_orientations)
    #    R_matrix = compute_covariance(estimated_data, true_positions, true_orientations)
    #    print("Covariance Matrix:\n", R_matrix)
    #else:
    #    print("No valid estimated positions found.")

    #def test_april_tags(tag_ids, tag_corners_world):
    #    """Prints the world coordinates of multiple AprilTags."""
    #    for tag_id in tag_ids:
    #        if tag_id in tag_corners_world:
    #           print(f"AprilTag {tag_id} Coordinates:\n", tag_corners_world[tag_id], "\n")
    #        else:
    #            print(f"AprilTag {tag_id} not found!\n")

    #test_april_tags([0, 12, 24, 36, 48, 60, 72, 84, 96], tag_corners_world)  # Change IDs as needed