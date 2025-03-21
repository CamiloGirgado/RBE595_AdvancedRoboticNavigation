import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

def generate_tag_corners():
    """Generates the world coordinates for AprilTag corners in a 12x9 grid."""
    tag_size = 0.152
    spacing = 0.152
    extra_spacing_cols = {3: 0.178, 6: 0.178}  # Extra spacing after columns 3 and 6

    tag_corners_world = {}
    y_offset = 0

    for col in range(12):
        x_offset = 0
        for row in range(9):
            tag_id = col * 9 + row
            x0, y0 = x_offset, y_offset
            x1, y1 = x0 + tag_size, y0
            x2, y2 = x1, y1 + tag_size
            x3, y3 = x0, y0 + tag_size
            tag_corners_world[tag_id] = np.array([[x0, y0, 0], [x1, y1, 0], [x2, y2, 0], [x3, y3, 0]])

            x_offset += spacing
        
        if col in extra_spacing_cols:
            y_offset += extra_spacing_cols[col]
        else:
            y_offset += spacing

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
                np.array([data['p1'][:, i], data['p2'][:, i], data['p3'][:, i], data['p4'][:, i]])
            )

    obj_points = np.array(obj_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.array(img_points, dtype=np.float32).reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, None

    R_cam_to_world, _ = cv2.Rodrigues(rvec)
    t_camera_to_robot = np.array([-0.04, 0, -0.03]).reshape(3, 1)
    R_x = R.from_euler('x', np.pi).as_matrix()
    R_z = R.from_euler('z', np.pi / 4).as_matrix()
    R_cam_to_robot = R_x @ R_z
    R_world_to_robot = R_cam_to_robot @ R_cam_to_world

    t_robot = R_cam_to_robot @ tvec + t_camera_to_robot
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

        true_positions.extend(vicon[:, :3])
        true_orientations.extend(vicon[:, 3:6])

        for entry in dataset:
            position, orientation = estimate_pose(entry, camera_matrix, dist_coeffs, tag_corners_world)
            if position is not None:
                estimated_positions.append(position)
                estimated_orientations.append(orientation)

    return np.array(estimated_positions), np.array(estimated_orientations), np.array(true_positions), np.array(true_orientations)

def plot_trajectory(estimated_positions, true_positions):
    """Plots the estimated trajectory against the ground truth."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label='Ground Truth')
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
        plt.plot(true_orientations[:, i], label='Ground Truth')
        plt.plot(estimated_orientations[:, i], label='Estimated')
        plt.ylabel(angles[i])
        plt.legend()
    plt.xlabel('Time')
    plt.suptitle('Euler Angles Comparison')
    plt.show()

def compute_covariance(estimated_positions, true_positions, estimated_orientations, true_orientations):
    """Computes covariance matrix of observation noise."""
    residuals = np.hstack((true_positions - estimated_positions, true_orientations - estimated_orientations))
    R_matrix = np.cov(residuals.T)
    return R_matrix

# Main execution
camera_matrix = np.array([
    [314.1779, 0, 199.4848],
    [0, 314.2218, 113.7838],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([-0.438607, 0.248625, -0.0911, 0.00072, -0.000476], dtype=np.float32)

tag_corners_world = generate_tag_corners()

data_folder = "/home/camilo/dev/RBE_595_ARN/data"  # Updated to correct folder path

estimated_positions, estimated_orientations, true_positions, true_orientations = process_data(
    data_folder, camera_matrix, dist_coeffs, tag_corners_world
)

if estimated_positions is not None and estimated_positions.size > 0:
    plot_trajectory(estimated_positions, true_positions)
    plot_euler_angles(estimated_orientations, true_orientations)
    R_matrix = compute_covariance(estimated_positions, true_positions, estimated_orientations, true_orientations)
    print("Covariance Matrix:\n", R_matrix)
else:
    print("No valid estimated positions found.")
