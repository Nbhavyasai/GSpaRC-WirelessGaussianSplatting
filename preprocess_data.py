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
    
class WirelessCSIData:
    """
    Container for one CSI sample (one TX snapshot).
    Keeps uplink + downlink (real/imag packed) and geometry metadata.
    """
    def __init__(self):
        self.uplink = torch.empty(0)     # [G, 52] where 52 = 26real+26imag (uplink)
        self.downlink = torch.empty(0)   # [G, 52] where 52 = 26real+26imag (downlink)
        self.tx_pos = torch.empty(0)     # optional (if you have it)
        self.rx_pos = torch.empty(0)     # base station position
        self.ray_directions = torch.empty(0)  # [n_rays, 3]
        self.view_matrix = torch.eye(4)

    def to(self, device):
        self.uplink = self.uplink.to(device)
        self.downlink = self.downlink.to(device)
        self.tx_pos = self.tx_pos.to(device) if self.tx_pos.numel() else self.tx_pos
        self.rx_pos = self.rx_pos.to(device)
        self.ray_directions = self.ray_directions.to(device) if self.ray_directions.numel() else self.ray_directions
        self.view_matrix = self.view_matrix.to(device)
        return self

class Spectrum_dataset(Dataset):
    """spectrum dataset class
    """
    def __init__(self, datadir, indexdir, scale_worldsize=1):
        super().__init__()
        self.datadir = datadir
        self.rx_pos_dir = os.path.join(datadir, 'rx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_info.yml')
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.npy')])
        example_spt = np.load(os.path.join(self.spectrum_dir, self.spt_names[0]))
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

        rx_pos = pd.read_csv(self.rx_pos_dir).values
        rx_pos = torch.tensor(rx_pos, dtype=torch.float32)
        tx_pos_fixed = torch.tensor([0.00, 0.50, 3.00], device=self.rx_pos.device, dtype=self.rx_pos.dtype)
        tx_pos = tx_pos_fixed.view(1, 3).repeat(rx_pos.shape[0], 1)
        print(rx_pos.shape, tx_pos.shape)

        ## Load data, each spectrum contains 90x360 pixels(rays)
        for i, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            # Read spectrum data
            spectrum_path = os.path.join(self.spectrum_dir, idx + '.npy')
            spectrum = np.load(spectrum_path)          # expected shape [90, 360]
            spectrum = torch.tensor(spectrum, dtype=torch.float32)

            data_sample = WirelessData()
            data_sample.spectrum_real = spectrum
            data_sample.spectrum_elevation = spectrum.shape[0]
            data_sample.spectrum_azimuth = spectrum.shape[1]
            data_sample.tx_pos = tx_pos[int(idx)-1]
            data_sample.rx_pos = rx_pos[int(idx)-1]
            data_sample.view_matrix = self.build_view_matrix(rx_pos[int(idx)-1])
            data_sample.to(device)
            wireless_data_list.append(data_sample)

        return wireless_data_list

    def build_view_matrix(self, rx_pos_world: torch.Tensor):
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_info = yaml.safe_load(f)
            gateway_orientation_quat = gateway_info['gateway1']['orientation']

        device = rx_pos_world.device

        # (3,3) camera->world rotation from quat
        R_c2w = torch.from_numpy(
            Rotation.from_quat(gateway_orientation_quat).as_matrix()
        ).float().to(device)
        # world->camera rotation
        R_w2c = R_c2w.transpose(0, 1)  # invert rotation

        # camera position C_w for this sample
        C_w = rx_pos_world.float().to(device)  # (3,)
        # world->camera translation
        t = -(R_w2c @ C_w)  # (3,)   # sign of this is based on kernel impl

        view_matrix = torch.eye(4, device=device, dtype=torch.float32)
        view_matrix[:3, :3] = R_w2c
        view_matrix[:3,  3] = t

        view_matrix = view_matrix.transpose(0, 1).contiguous()  # (4,4)

        return view_matrix

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
        gateway_pos = torch.tensor(gateway_pos, dtype=torch.float32).reshape(3, 1)  # [3, 1] gateway position in world coordinate
        view_matrix = torch.eye(4, device='cuda')
        gateway_pos = gateway_pos.to(view_matrix.device)
        R = R.to(view_matrix.device)
        view_matrix[:3, :3] = R.transpose(0,1)  # invert rotation
        view_matrix[:3, 3] = gateway_pos.squeeze()
        view_matrix = view_matrix.transpose(0,1)

        r_o = torch.tile(gateway_pos, (self.rays_per_spectrum,)).reshape(-1, 3)  # [n_rays, 3]

        return gateway_pos, r_d.T, view_matrix
    
class CSI_dataset(Dataset):
    """
    Adapted from reference CSI_dataset, but returns per-sample objects like your Spectrum_dataset.

    Reference format assumptions (kept the same):
      - datadir/csidata.npy : complex-valued CSI of shape [N, G, 52]  (commonly [N, 8, 52])
      - datadir/base-station.yml : base station position (and optionally orientation)
      - indexdir : train_index.txt / test_index.txt with integer indices

    What this dataset returns per __getitem__:
      WirelessCSIData with:
        - uplink  : [G, 52] float (26 real + 26 imag)
        - downlink: [G, 52] float (26 real + 26 imag)
        - rx_pos, ray_directions, view_matrix
    """

    def __init__(self, datadir, indexdir, scale_worldsize=1, keep_per_gateway=True, normalize=True):
        super().__init__()
        self.datadir = datadir
        self.csidata_dir = os.path.join(datadir, "csidata.npy")
        self.bs_pos_dir = os.path.join(datadir, "base-station.yml")
        self.dataset_index = np.loadtxt(indexdir, dtype=int)

        # ray sampling resolution (kept same as reference CSI/BLE style)
        self.beta_res, self.alpha_res = 9, 36
        self.n_rays = self.beta_res * self.alpha_res

        self.keep_per_gateway = keep_per_gateway  # if False, uplink/downlink are flattened to [G*52]
        self.normalize = normalize
        self.csi_max = None

        # Load base station position (and optional orientation)
        self.bs_pos, self.bs_orientation = self._load_base_station(scale_worldsize=scale_worldsize)

        # Geometry metadata (one BS -> one origin + shared ray directions)
        self.rx_pos, self.ray_directions, self.view_matrix = self._gen_bs_geometry()

        # Load CSI once into memory (like reference does)
        self.uplink_all, self.downlink_all = self._load_and_process_csi()

        # Build per-sample list of WirelessCSIData objects (like your Spectrum_dataset)
        self.wireless_data = self._pack_samples()

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        return self.wireless_data[index]

    def get_data(self):
        return self.wireless_data

    def _load_base_station(self, scale_worldsize=1):
        """
        Reference uses base-station.yml. The reference code assumes something like:
          bs_pos_dict["base_station"] -> [x, y, z]
        It doesn't reliably handle "n_bs" (it used len(bs_pos) which becomes 3). We fix that here.
        """
        with open(self.bs_pos_dir, "r") as f:
            bs_info = yaml.safe_load(f)

        # Common patterns:
        # 1) {"base_station": [x,y,z]}
        # 2) {"base_station": {"position":[x,y,z], "orientation":[qx,qy,qz,qw]}}
        base_station = bs_info.get("base_station", bs_info)

        if isinstance(base_station, dict):
            pos = base_station.get("position", base_station.get("pos", None))
            ori = base_station.get("orientation", None)
        else:
            pos = base_station
            ori = None

        if pos is None:
            raise ValueError(f"Could not parse base station position from {self.bs_pos_dir}")

        bs_pos = torch.tensor(pos, dtype=torch.float32) / scale_worldsize  # [3]
        bs_orientation = None
        if ori is not None:
            bs_orientation = torch.tensor(ori, dtype=torch.float32)  # quat expected

        return bs_pos, bs_orientation

    def _gen_bs_geometry(self):
        """
        Generate:
          rx_pos        : [3]
          ray_directions: [n_rays, 3]
          view_matrix   : [4,4]
        If orientation is available in YAML, rotate ray directions accordingly (like reference spectrum did with quat).
        Otherwise use identity rotation.
        """
        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)
        betas = betas.repeat_interleave(self.alpha_res)

        radius = 1.0
        x = radius * torch.cos(alphas) * torch.cos(betas)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], dim=0)  # [3, n_rays] in BS local coords

        # Apply orientation if provided
        if self.bs_orientation is not None:
            R = torch.from_numpy(Rotation.from_quat(self.bs_orientation.cpu().numpy()).as_matrix()).float()
            r_d = R @ r_d

        ray_directions = r_d.T.contiguous()  # [n_rays, 3]

        # Construct view_matrix
        # If you want spectrum-style "camera" convention, you can swap axes here.
        # For CSI we keep a simple rigid transform: rotation (if any) + translation.
        view_matrix = torch.eye(4, dtype=torch.float32, device=device)
        if self.bs_orientation is not None:
            R = torch.from_numpy(Rotation.from_quat(self.bs_orientation.cpu().numpy()).as_matrix()).float().to(device)
            # store an inverse-like convention (similar to your Spectrum_dataset) if needed
            view_matrix[:3, :3] = R.transpose(0, 1)
        view_matrix[:3, 3] = (self.bs_pos.to(device)).view(-1)

        rx_pos = self.bs_pos.clone().detach()

        return rx_pos, ray_directions, view_matrix

    def _load_and_process_csi(self):
        """
        Load csidata.npy and convert to (uplink, downlink) with real/imag packed like the reference:

          csi: [N, G, 52] complex
          uplink_complex   = csi[..., :26]
          downlink_complex = csi[..., 26:]

          uplink  = cat([real(uplink), imag(uplink)], -1)   -> [N, G, 52] float
          downlink= cat([real(downlink),imag(downlink)],-1) -> [N, G, 52] float
        """
        csi = np.load(self.csidata_dir)  # expected complex array
        csi = torch.from_numpy(csi)

        if self.normalize:
            self.csi_max = torch.max(torch.abs(csi))
            csi = csi / (self.csi_max + 1e-12)

        uplink_c = csi[..., :26]
        downlink_c = csi[..., 26:]

        uplink = torch.cat([torch.real(uplink_c), torch.imag(uplink_c)], dim=-1).float()
        downlink = torch.cat([torch.real(downlink_c), torch.imag(downlink_c)], dim=-1).float()

        return uplink, downlink

    def _pack_samples(self):
        """
        Convert indexed CSI into list[WirelessCSIData], matching your Spectrum_dataset style.
        """
        data_list = []

        # Optional: if you have UE/TX positions in your CSI dataset folder, you can load them here.
        # This keeps the class compatible with your pipeline style.
        tx_pos_path = os.path.join(self.datadir, "tx_pos.csv")
        tx_pos = None
        if os.path.exists(tx_pos_path):
            tx_pos = torch.tensor(pd.read_csv(tx_pos_path).values, dtype=torch.float32)

        for i, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            ul = self.uplink_all[idx]     # [G, 52]
            dl = self.downlink_all[idx]   # [G, 52]

            if not self.keep_per_gateway:
                ul = ul.reshape(-1)       # [G*52]
                dl = dl.reshape(-1)       # [G*52]

            sample = WirelessCSIData()
            sample.uplink = ul
            sample.downlink = dl
            sample.rx_pos = self.rx_pos
            sample.ray_directions = self.ray_directions
            sample.view_matrix = self.view_matrix

            if tx_pos is not None:
                # If your indices are 1-based like spectrum ("1..N"), adjust here.
                # Reference CSI indices are 0-based. We default to 0-based.
                if idx < tx_pos.shape[0]:
                    sample.tx_pos = tx_pos[idx]
            sample.to(device)
            data_list.append(sample)

        return data_list

dataset_dict = {"rfid": Spectrum_dataset}