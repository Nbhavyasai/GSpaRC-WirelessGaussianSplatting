###Initialization of the pointcloud#########
####(Uniform , Random, sfm), covariance, signal, opacity

import numpy as np
import pandas as pd
import torch
from gaussian import GaussianModel

class UniformInitializer:
    """
    Performs uniform initialization of the gaussians in the scene
    """
    def __init__(self, xyz_min, xyz_max, cube_size, normalize=False):
        self.xyz_min = np.array(xyz_min)
        self.xyz_max = np.array(xyz_max)
        self.cube_size = cube_size
        self.normalize = normalize

    def get_initial_points(self):
        # Determine the number of points per axis
        num_points_per_axis = ((self.xyz_max - self.xyz_min) / self.cube_size).astype(int)
        x_vals = np.linspace(self.xyz_min[0], self.xyz_max[0], num_points_per_axis[0])
        y_vals = np.linspace(self.xyz_min[1], self.xyz_max[1], num_points_per_axis[1])
        z_vals = np.linspace(self.xyz_min[2], self.xyz_max[2], num_points_per_axis[2])

        # Create a meshgrid
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

        # Reshape and stack into (N, 3)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        return points

initializer_dict = {"uniform": UniformInitializer}