###Based on the type of dataset, preprocess data step and prepare tarining/test data###

import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_dataset(datadir, ratio=0.8, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)
    elif dataset_type == "ble":
        rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        index = pd.read_csv(rssi_dir).index.values
        random.shuffle(index)
    elif dataset_type == "mimo":
        csi_dir = os.path.join(datadir, 'csidata.npy')
        index = [i for i in range(np.load(csi_dir).shape[0])]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')


def get_view_matrix(ray_origin, ray_direction):
    # Caclulate basis for Tx coordinate system
    device = ray_direction.device

    forward_vector = ray_direction      # already a unit vector centered at Rx

    # Temporary up vector (fixed)
    temp_up_vector = torch.tensor([0, 0, -1], dtype=torch.float32, device=device)

    # Calculate right vector (normalized)
    right_vector = torch.linalg.cross(temp_up_vector, forward_vector)
    right_vector = right_vector / torch.norm(right_vector)

    # Calculate up vector (normalized)
    up_vector = torch.linalg.cross(forward_vector, right_vector)
    up_vector = up_vector / torch.norm(up_vector)

    # Construct rotation matrix
    rotation = torch.stack([right_vector, up_vector, forward_vector], dim=0)

    # Compute translation
    translation = -torch.matmul(rotation, ray_origin)

    # Construct 4x4 view matrix
    view_matrix = torch.eye(4, dtype=torch.float32, device=device)
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = translation

    return view_matrix    

class WirelessData:
    def __init__(self):
        self.spectrum_real = torch.empty(0)
        self.spectrum_imag = torch.empty(0)
        self.tx_pos = torch.empty(0)
        self.rx_pos = torch.empty(0)
        self.spectrum_elevation = 0
        self.spectrum_azimuth = 0
        self.view_matrix = torch.eye(4)

    def to(self, device):
        """Move all tensors in the instance to the specified device."""
        self.spectrum_real = self.spectrum_real.to(device)
        self.spectrum_imag = self.spectrum_imag.to(device)
        self.tx_pos = self.tx_pos.to(device)
        self.rx_pos = self.rx_pos.to(device)
        self.view_matrix = self.view_matrix.to(device)
        return self

class Spectrum_dataset(Dataset):
    """spectrum dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1):
        super().__init__()
        self.datadir = datadir
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_info.yml')
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])
        example_spt = imageio.imread(os.path.join(self.spectrum_dir, self.spt_names[0]))
        self.n_elevation, self.n_azimuth = example_spt.shape
        self.rays_per_spectrum = self.n_elevation * self.n_azimuth
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        
        # data to be used
        self.rx_pos, self.ray_directions, self.view_matrix = self.gen_rays_spectrum()
        self.wireless_data = self.load_data()
        
    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        return self.wireless_data[index]
    
    def get_data(self):
        return self.wireless_data

    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        loaded_data: list where each element is dictionary of a single data point
        """
        wireless_data_list = []

        ## Load transmitter position
        # tx_pos = pd.read_csv(self.tx_pos_dir).values
        # tx_pos = torch.tensor(tx_pos, dtype=torch.float32)
        # rx_pos = self.rx_pos.clone().detach().repeat(tx_pos.shape[0], 1)
        tx_pos = pd.read_csv(self.tx_pos_dir).values
        tx_pos = torch.tensor(tx_pos, dtype=torch.float32)
        rx_pos_base = self.rx_pos.clone().detach()
        rx_pos = rx_pos_base.unsqueeze(0).repeat(tx_pos.shape[0],1)

        ## Load data, each spectrum contains 90x360 pixels(rays)
        for i, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            # Read spectrum data saved as images
            spectrum = imageio.imread(os.path.join(self.spectrum_dir, idx + '.png')) / 255.0    # [90, 360]
            spectrum = torch.tensor(spectrum, dtype=torch.float32)

            data_sample = WirelessData()
            data_sample.spectrum_real = spectrum
            data_sample.spectrum_elevation = spectrum.shape[0]
            data_sample.spectrum_azimuth = spectrum.shape[1]
            data_sample.tx_pos = tx_pos[int(idx)-1]
            data_sample.rx_pos = rx_pos[int(idx)-1]
            data_sample.view_matrix = self.view_matrix
            data_sample.to(device)
            wireless_data_list.append(data_sample)

        return wireless_data_list

    def gen_rays_spectrum(self):
        """generate sample rays origin at gateway with resolution given by spectrum

        Returns
        -------
        gateway_pos : tensor. [3]. The origin of all ray (gateway/rx position)
        r_d : tensor. [n_rays, 2]. The direction of rays (unit vector)
        """
        ## Load gateway position and orientation
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_info = yaml.safe_load(f)
            gateway_pos = gateway_info['gateway1']['position']
            gateway_orientation = gateway_info['gateway1']['orientation']

        azimuth = torch.linspace(1, 360, self.n_azimuth) / 180 * np.pi
        azimuth = azimuth % (2 * np.pi)
        azimuth = torch.where(azimuth > np.pi, azimuth - 2 * np.pi, azimuth)
        elevation = torch.linspace(1, 90, self.n_elevation) / 180 * np.pi
        azimuth = torch.tile(azimuth, (self.n_elevation,))  # [1,2,3...360,1,2,3...360,...] pytorch 2.0
        elevation = torch.repeat_interleave(elevation, self.n_azimuth)  # [1,1,1,...,2,2,2,...,90,90,90,...]

        x = 1 * torch.cos(elevation) * torch.cos(azimuth) # [n_azi * n_ele], i.e., [n_rays]
        y = 1 * torch.cos(elevation) * torch.sin(azimuth)
        z = 1 * torch.sin(elevation)

        r_d = torch.stack([y, z, x], dim=0)  # [3, n_rays] 3D direction of rays in gateway coordinate

        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        r_d = R @ r_d  # [3, n_rays] 3D direction of rays in world coordinate
        gateway_pos = torch.tensor(gateway_pos, dtype=torch.float32) #.reshape(3, 1)  # [3, 1] gateway position in world coordinate
        view_matrix = torch.eye(4, device='cuda')
        gateway_pos = gateway_pos.to(view_matrix.device)
        R = R.to(view_matrix.device)
        view_matrix[:3, :3] = R.transpose(0,1)  # invert rotation
        view_matrix[:3, 3] = gateway_pos.squeeze()
        view_matrix = view_matrix.transpose(0,1)

        r_o = torch.tile(gateway_pos, (self.rays_per_spectrum,)).reshape(-1, 3)  # [n_rays, 3]

        return gateway_pos, r_d.T, view_matrix
    
dataset_dict = {"rfid": Spectrum_dataset}