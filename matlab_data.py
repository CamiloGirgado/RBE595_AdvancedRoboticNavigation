import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

class matlab_data:
    def __init__(self, file_name):
        self.data = scipy.io.loadmat(file_name, simplify_cells=True)

    def get_data(self):
        return self.data