import scipy.io as sio
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float_kind': "{: .3f}".format})

# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
print(f"Parent directory of the script: {parent_dir_script}")

class MeasurementData:
    def __init__(self, ax=None, fig=None):
        self.ax = ax
        self.fig = fig
        # Update to use the specific directory you provided
        self.data_dir = os.path.expanduser('~/dev/RBE_595_ARN/data')  
        self.camera_matrix = np.array([
            [314.1779, 0.0, 199.4848],
            [0.0, 314.2218, 113.7838],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float64)
        self.camera_to_world_transform = np.array([
            [1.0, 0.0, 0.0, -0.04],  
            [0.0, 1.0, 0.0, 0.0],  
            [0.0, 0.0, 1.0, -0.03],  
            [0.0, 0.0, 0.0, 1.0]  
        ], dtype=np.float64)

        self.transform_from_camera_to_drone = self.homogeneous_matrix(-0.04, 0.0, -0.03, np.pi, 0, np.pi/4)  
        print("Camera to World Transform Matrix:")
        print(self.transform_from_camera_to_drone)

    def homogeneous_matrix(self, tx, ty, tz, rx, ry, rz):
        translation = self.translation_matrix(tx, ty, tz)
        rotation_x = self.rotation_matrix_x(rx)
        rotation_y = self.rotation_matrix_y(ry)
        rotation_z = self.rotation_matrix_z(rz)
        
        rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        transform = np.dot(translation, rotation)
        return transform

    def translation_matrix(self, tx, ty, tz):
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_x(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_y(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_z(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def loadMatlabData(self, file_name):
        mat_fname = os.path.join(self.data_dir, file_name)
        mat_contents = sio.loadmat(mat_fname, simplify_cells=True)
        return mat_contents

    def get_corners_world_frame(self, april_tag_index):
        # Ensure `april_tag_index` is a scalar
        if isinstance(april_tag_index, np.ndarray):
            april_tag_index = april_tag_index.item()  # Get the scalar value from the array
    
        square_size = 0.152  
        space_size = 0.152
        columns_3_4_and_6_7 = 0.178
        difference = columns_3_4_and_6_7 - square_size
    
        col = april_tag_index // 9  # Row-major ordering; ensure `col` is a scalar
        row = april_tag_index % 9  # Ensure `row` is scalar

        p1_x = (row) * (square_size + space_size) + square_size
        p2_x = p1_x
        p3_x = p1_x - square_size
        p4_x = p3_x

        p1_y = (col) * (square_size + space_size)
        if col >= 3:
            p1_y += difference
        if col >= 5:
            p1_y += difference
        p2_y = p1_y + square_size
        p3_y = p2_y
        p4_y = p1_y

        p1 = np.array([p1_x, p1_y, 0.0])
        p2 = np.array([p2_x, p2_y, 0.0])
        p3 = np.array([p3_x, p3_y, 0.0])
        p4 = np.array([p4_x, p4_y, 0.0])
    
        return np.array([p1, p2, p3, p4], dtype=np.float64)


    def process_measurement_data(self, mat_contents):
        position_data = []
        orientation_data = []
        for data in mat_contents['data']:
            if isinstance(data['id'], np.ndarray) and len(data['id']) == 0:
                continue

            position, orientation = self.estimate_pose(data)
            if position is None or orientation is None:
                print("Warning: Pose estimation failed for the current data item. Skipping this item.")
                continue

            position_data.append(position)
            orientation_data.append(orientation)

        return position_data, orientation_data

    def estimate_pose(self, data):
        image_points = np.array([data['p1'], data['p2'], data['p3'], data['p4']], dtype=np.float64)
        object_points = self.get_corners_world_frame(data['id'])

        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)
        if retval is None:
            print("Error: solvePnP failed to estimate pose.")
            return None, None

        rotM, _ = cv2.Rodrigues(rvec)
        camera_position = -np.matrix(rotM).T * np.matrix(tvec)
        
        return camera_position, rotM

    def plot_trajectory_estimated(self, data):
        data_np = np.array(data).squeeze().T
        x = data_np[0, :]
        y = data_np[1, :]
        z = data_np[2, :]

        self.ax.plot(x, y, z, label='Trajectory', color='r', linewidth=1)
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.set_title('3D Trajectory Plot')
        self.ax.legend()

        plt.show()

def loadMatlabData(file_name):
    measurement_data = MeasurementData()
    mat_contents = measurement_data.loadMatlabData(file_name)
    return mat_contents, measurement_data

def plot_trajectory_vicon(data):
    x = data['vicon'][0, :]
    y = data['vicon'][1, :]
    z = data['vicon'][2, :]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Actual', color='b', linewidth=2)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Trajectory Plot')
    ax.legend()

    plt.show()

def check_data():
    checkMatlabData('studentdata0.mat')
    checkMatlabData('studentdata1.mat')
    checkMatlabData('studentdata2.mat')
    checkMatlabData('studentdata3.mat')
    checkMatlabData('studentdata4.mat')
    checkMatlabData('studentdata5.mat')
    checkMatlabData('studentdata6.mat')
    checkMatlabData('studentdata7.mat')

def plot_trajectory_0():
    data, _ = loadMatlabData('studentdata0.mat')
    ax, fig = plot_trajectory_vicon(data)
    return ax, fig

# Run the plot for the trajectory estimation
def main():
    mat_contents, measurement_data = loadMatlabData('studentdata0.mat')
    position_data, orientation_data = measurement_data.process_measurement_data(mat_contents)
    measurement_data.plot_trajectory_estimated(position_data)

if __name__ == "__main__":
    main()
