from os.path import dirname, join as pjoin
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from observationModel_1 import observationModel
from Visual_Localization import VisualLocalization

def execute_UKF(file_name):
    rmse = .0
    vl = VisualLocalization(file_name)
    vl.run_UKF()
    rmse = vl.rmse()
    vl.plot_trajectory()  # Plot the trajectory
    vl.plot_orientation()  # Plot the roll trajectory

    return rmse


def execute_PF(file_name, particle_count = 500):
    rmse = .0    
    vl = VisualLocalization(file_name)
    vl.run_PF(particle_count=particle_count)
    rmse = vl.rmse()
    vl.plot_trajectory()  # Plot the trajectory
    vl.plot_orientation()  # Plot the roll trajectory

    return rmse

def run_UKF():
    results = []
    # results.append(execute_UKF('data/studentdata0.mat'))
    # results.append(execute_UKF('data/studentdata1.mat'))
    # results.append(execute_UKF('data/studentdata2.mat'))
    # results.append(execute_UKF('data/studentdata3.mat'))
    # results.append(execute_UKF('data/studentdata4.mat'))
    # results.append(execute_UKF('data/studentdata5.mat'))
    # results.append(execute_UKF('data/studentdata6.mat'))
    # results.append(execute_UKF('data/studentdata7.mat'))
    # print("UKF RMSE:", sum(results)/len(results))

def run_PF(particle_count):
    results = []
    # results.append(execute_PF('data/studentdata0.mat'))
    # results.append(execute_PF('data/studentdata1.mat'))
    results.append(execute_PF('data/studentdata2.mat'))
    # results.append(execute_PF('data/studentdata3.mat'))
    # results.append(execute_PF('data/studentdata4.mat'))
    # results.append(execute_PF('data/studentdata5.mat'))
    # results.append(execute_PF('data/studentdata6.mat'))
    # results.append(execute_PF('data/studentdata7.mat'))
    print("PF RMSE with Particles - {particle_count}:", sum(results)/len(results))

if __name__ == "__main__":
    # run_UKF()
    # run_PF(250)
    run_PF(500)
    # run_PF(750)
    # run_PF(1000)
    # run_PF(2000)
    # run_PF(3000)
    # run_PF(4000)
    # run_PF(5000)