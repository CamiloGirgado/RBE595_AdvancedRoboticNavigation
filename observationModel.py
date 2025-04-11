import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

def generate_tag_corners():
    """Generates the world coordinates for AprilTag corners in a 12x9 grid."""
    tag_size = 0.152  # Size of each AprilTag
    spacing = 0.152   # Default spacing between tags
 
    # Extra spacing applies between columns 3-4 and 6-7 (0-indexed)
    extra_spacing_cols = {3: 0.178 - spacing, 6: 0.178 - spacing}
 
    tag_corners_world = {}
 
    for row in range(12):  # Rows go down (x-direction)
        x = row * (tag_size + spacing)
 
        for col in range(9):  # Columns go right (y-direction)
            y = 0
            for c in range(col):
                y += tag_size
                if c in extra_spacing_cols:
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

def estimate_pose(data, camera_matrix, dist_coeffs, tag_corners_world):
    """Estimates the position and orientation of the quadrotor."""
    # Debugging: Check structure
    print("Data keys:", data.keys())
    print("ID Type:", type(data['id']))

    if isinstance(data['id'], int) or len(data['id']) == 0:

        return None, None  # No valid tags detected
    
    obj_points = []
    img_points = []

    for i, tag_id in enumerate(data['id']):
        if tag_id in tag_corners_world:
            obj_points.append(tag_corners_world[tag_id])
            img_points.append(
                np.array([data['p1'][:, i], data['p2'][:, i], data['p3'][:, i], data['p4'][:, i]]))

    obj_points = np.array(obj_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.array(img_points, dtype=np.float32).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

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
    euler_angles = R.from_matrix(R_world_to_robot).as_euler('xyz')

    return t_robot.flatten(), euler_angles

def process_data(directory, camera_matrix, dist_coeffs, tag_corners_world):
    """Processes all .mat files in the given directory and estimates pose."""
    estimated_positions = []
    estimated_orientations = []
    true_positions = []
    true_orientations = []

    # List all MAT files in the folder
    mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]

    if not mat_files:
        print("No .mat files found in the directory!")
        return None, None, None, None
    mat_files = ['studentdata5.mat']
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
        plt.plot(estimated_orientations[-1, :], estimated_orientations[i+3, :], label='Estimated')
        plt.ylabel(angles[i])
        plt.legend()
    plt.xlabel('Time')
    plt.suptitle('Euler Angles Comparison')
    plt.show()

#def compute_covariance(estimated_positions, true_positions, estimated_orientations, true_orientations):
#    """Computes covariance matrix of observation noise."""
#    residuals = np.hstack((true_positions - estimated_positions, true_orientations - estimated_orientations))
##    R_matrix = np.cov(residuals.T)
#    return R_matrix

#def compute_covariance(estimated_positions, true_positions, estimated_orientations, true_orientations):
#    """Computes covariance matrix of observation noise."""
#    min_length = min(len(estimated_positions), len(true_positions))
#    estimated_positions = estimated_positions[:min_length]
#    true_positions = true_positions[:min_length]
#    estimated_orientations = estimated_orientations[:min_length]
#    true_orientations = true_orientations[:min_length]
#    residuals = np.hstack((true_positions - estimated_positions, true_orientations - estimated_orientations))
#    R_matrix = np.cov(residuals.T)
#    return R_matrix

def compute_covariance(estimated_positions, true_positions, estimated_orientations, true_orientations):
    """Computes the covariance matrix of the observation noise."""

    # Make sure all data is the same length
    min_length = min(
        estimated_positions.shape[0],
        true_positions.shape[1],  # ground truth shape is (3, T)
        estimated_orientations.shape[0],
        true_orientations.shape[1]  # ground truth shape is (3, T)
    )

    # Align the data
    est_pos = estimated_positions[:min_length]
    true_pos = true_positions[:, :min_length].T  # Transpose to shape (T, 3)

    est_ori = estimated_orientations[:min_length]
    true_ori = true_orientations[:3, :min_length].T  # Only roll, pitch, yaw, shape (T, 3)

    # Residuals = true - estimated
    pos_residuals = true_pos - est_pos
    ori_residuals = true_ori - est_ori

    # Wrap angles to [-pi, pi] to avoid wraparound issues
    ori_residuals = (ori_residuals + np.pi) % (2 * np.pi) - np.pi

    # Combine residuals: shape (T, 6)
    residuals = np.hstack((pos_residuals, ori_residuals))

    # Covariance matrix: shape (6, 6)
    R_matrix = np.cov(residuals.T)

    return R_matrix

# Main execution
camera_matrix = np.array([
        [314.1779, 0, 199.4848],
        [0, 314.2218, 113.7838],
        [0, 0, 1]
    ], dtype=np.float32)

dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float32)
tag_corners_world = generate_tag_corners()
data_folder = "/home/camilo/dev/RBE_595_ARN/data"  # Updated to correct folder path
estimated_data, true_positions, true_orientations = process_data(data_folder, camera_matrix, dist_coeffs, tag_corners_world)

if estimated_data is not None and estimated_data.size > 0:
    plot_trajectory(estimated_data, true_positions)
    plot_euler_angles(estimated_data, true_orientations)
    #R_matrix = compute_covariance(estimated_positions, true_positions, estimated_orientations, true_orientations)
    #print("Covariance Matrix:\n", R_matrix)
else:
    print("No valid estimated positions found.")

def test_april_tags(tag_ids, tag_corners_world):
    """Prints the world coordinates of multiple AprilTags."""
    for tag_id in tag_ids:
        if tag_id in tag_corners_world:
            print(f"AprilTag {tag_id} Coordinates:\n", tag_corners_world[tag_id], "\n")
        else:
            print(f"AprilTag {tag_id} not found!\n")

test_april_tags([0, 1, 61, 73], tag_corners_world)  # Change IDs as needed