from os.path import dirname, join as pjoin
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from observationModel_1 import observationModel
from Visual_Localization import VisualLocalization

def execute_UKF(file_name):
    rmse = .0
    Visual_Localization = VisualLocalization(file_name)
    Visual_Localization.run_ukf()
    Visual_Localization.calculate_covariance()
    rmse = VisualLocalization.rmse()
    VisualLocalization.plot_trajectory()  # Plot the trajectory
    VisualLocalization.plot_orientation()  # Plot the roll trajectory

    return rmse


def execute_PF(file_name, particle_count=250):
    rmse = .0    
    Visual_Localization = VisualLocalization(file_name)
    Visual_Localization.process_particle_filter(particle_count=particle_count)
    rmse = Visual_Localization.rmse()
    Visual_Localization.plot_trajectory()  # Plot the trajectory
    Visual_Localization.plot_orientation()  # Plot the roll trajectory

    return rmse

def run_UKF():
    results = []
    results.append(run_UKF('studentdata0.mat'))
    results.append(run_UKF('studentdata1.mat'))
    results.append(run_UKF('studentdata2.mat'))
    results.append(run_UKF('studentdata3.mat'))
    results.append(run_UKF('studentdata4.mat'))
    results.append(run_UKF('studentdata5.mat'))
    results.append(run_UKF('studentdata6.mat'))
    results.append(run_UKF('studentdata7.mat'))
    print("UKF RMSE:", sum(results/len(results)))

def run_PF(particle_count):
    results = []
    particles = 250
    results.append(run_PF('studentdata0.mat'))
    results.append(run_PF('studentdata1.mat'))
    results.append(run_PF('studentdata2.mat'))
    results.append(run_PF('studentdata3.mat'))
    results.append(run_PF('studentdata4.mat'))
    results.append(run_PF('studentdata5.mat'))
    results.append(run_PF('studentdata6.mat'))
    results.append(run_PF('studentdata7.mat'))
    print("PF RMSE with Particles - {particle_count}:", sum(results)/len(results))

if __name__ == "__main__":
    run_UKF()
    run_PF(250)
    run_PF(500)
    run_PF(750)
    run_PF(1000)
    run_PF(2000)
    run_PF(3000)
    run_PF(4000)
    run_PF(5000)

