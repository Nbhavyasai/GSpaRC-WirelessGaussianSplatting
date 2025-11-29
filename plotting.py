######Functions to plot scene and training/test stats######
from initialization import *
from gaussian import *

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os
import imageio.v2 as imageio  # imageio for saving GIFs
from scipy.spatial.transform import Rotation as R



def batch_quaternion_to_matrix(quats):
    """
    Converts a batch of quaternions (N, 4) to rotation matrices (N, 3, 3)
    Quaternion format: (w, x, y, z)
    """
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rot_mats = np.empty((quats.shape[0], 3, 3))
    rot_mats[:, 0, 0] = 1 - 2*(yy + zz)
    rot_mats[:, 0, 1] = 2*(xy - wz)
    rot_mats[:, 0, 2] = 2*(xz + wy)

    rot_mats[:, 1, 0] = 2*(xy + wz)
    rot_mats[:, 1, 1] = 1 - 2*(xx + zz)
    rot_mats[:, 1, 2] = 2*(yz - wx)

    rot_mats[:, 2, 0] = 2*(xz - wy)
    rot_mats[:, 2, 1] = 2*(yz + wx)
    rot_mats[:, 2, 2] = 1 - 2*(xx + yy)

    return rot_mats

def manage_figures(func):
    """Decorator to ensure figures are properly closed after saving"""
    def wrapper(*args, **kwargs):
        plt.close('all')  # Close any existing figures
        result = func(*args, **kwargs)
        plt.close('all')  # Close figures after saving
        return result
    return wrapper

@manage_figures
def plot_initialization(xyz_min, xyz_max, pc, save_path="results/initial_gaussians.png"):
    """Plot the initial Gaussians with optimized performance."""
    # Get and detach data
    centers = pc.get_centre.detach().cpu().numpy()      
    scales = pc.get_scaling.detach().cpu().numpy()      
    quaternions = pc.get_rotation.detach().cpu().numpy()
    opacities = pc.get_opacity.detach().cpu().numpy()   

    # Convert quaternions to rotation matrices
    rotations = batch_quaternion_to_matrix(quaternions)   

    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Initial 3D Gaussian Scene")

    # Create unit sphere with lower resolution
    u = np.linspace(0, 2 * np.pi, 10)  # Reduced from 20 to 10
    v = np.linspace(0, np.pi, 10)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)

    # Use size threshold to determine plotting method
    scale_magnitudes = np.linalg.norm(scales, axis=1)
    size_threshold = 0.5  # Adjust this value based on your needs
    
    # Plot small Gaussians as points
    small_indices = scale_magnitudes < size_threshold
    if np.any(small_indices):
        ax.scatter(
            centers[small_indices, 0],
            centers[small_indices, 1],
            centers[small_indices, 2],
            c='red',
            alpha=opacities[small_indices],
            s=20
        )

    # Plot large Gaussians as surfaces
    large_indices = ~small_indices
    for i in np.where(large_indices)[0]:
        center = centers[i]
        scale = scales[i].reshape(3,)
        R = rotations[i]
        alpha = float(opacities[i])

        transform = R * scale[np.newaxis, :]
        ellipsoid = np.einsum('ij,klj->kli', transform, sphere)
        ellipsoid += center

        ax.plot_surface(
            ellipsoid[:, :, 0],
            ellipsoid[:, :, 1],
            ellipsoid[:, :, 2],
            rstride=2, cstride=2,  # Increased stride
            color='red',
            alpha=alpha,
            edgecolor='none'
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)  # Reduced DPI
    plt.close(fig)  # Explicitly close the figure

@manage_figures
def plot_final_gaussians(xyz_min, xyz_max, pc, save_path="results/final_gaussians.png"):
    """Plot the final Gaussians with optimized performance."""
    # Use the same optimized implementation as plot_initialization
    plot_initialization(xyz_min, xyz_max, pc, save_path)

def generate_gif(xyz_min, xyz_max, pc, gif_path='results/gaussian_scene.gif', num_frames=36):
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    centers = pc.get_centre.detach().cpu().numpy()
    scales = pc.get_scaling.detach().cpu().numpy()
    quats = pc.get_rotation.detach().cpu().numpy()
    opacities = pc.get_opacity.detach().cpu().numpy()

    rot_mats = batch_quaternion_to_matrix(quats)

    # Generate base ellipsoids
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)

    frames = []

    for i_frame in range(num_frames):
        angle = (i_frame / num_frames) * 360

        fig = go.Figure()

        for i in range(centers.shape[0]):
            center = centers[i]
            scale = scales[i]
            R = rot_mats[i]
            alpha = float(opacities[i])

            transform = R * scale[np.newaxis, :]
            ellipsoid = np.einsum('ij,klj->kli', transform, sphere) + center

            fig.add_trace(go.Surface(
                x=ellipsoid[:, :, 0],
                y=ellipsoid[:, :, 1],
                z=ellipsoid[:, :, 2],
                showscale=False,
                opacity=alpha,
                colorscale='Blues',
                lighting=dict(ambient=0.5),
            ))

        # Set rotating camera
        camera = dict(
            eye=dict(
                x=2*np.sin(np.radians(angle)),
                y=1.5,
                z=2*np.cos(np.radians(angle))
            )
        )

        fig.update_layout(
            scene_camera=camera,
            scene=dict(
                xaxis=dict(range=[xyz_min[0], xyz_max[0]]),
                yaxis=dict(range=[xyz_min[1], xyz_max[1]]),
                zaxis=dict(range=[xyz_min[2], xyz_max[2]]),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # Save each frame as PNG using Kaleido
        frame_path = f'results/frame_{i_frame:03d}.png'
        fig.write_image(frame_path, width=800, height=600)
        frames.append(imageio.imread(frame_path))

    # Save GIF
    imageio.mimsave(gif_path, frames, fps=12)
    print(f"🎞️ GIF saved to {gif_path}")

    # Optional: cleanup individual frame files
    for frame_file in [f'results/frame_{i:03d}.png' for i in range(num_frames)]:
        os.remove(frame_file)

@manage_figures
def plot_spectrum(spectrum, save_path=None, cmap='jet'):
    """
    Plot a spectrum in polar coordinates.

    Args:
        spectrum (torch.Tensor or np.ndarray): The spectrum data to plot, shape (90, 360).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        cmap (str, optional): Colormap to use for the plot. Default is 'jet'.
    """
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.cpu().numpy()

    spectrum = spectrum.reshape(90, 360)
    spectrum = np.flipud(spectrum)  # Flip vertically for correct orientation

    # Create polar grid
    r = np.linspace(0, 1, spectrum.shape[0] + 1)
    theta = np.linspace(0, 2 * np.pi, spectrum.shape[1] + 1)
    r, theta = np.meshgrid(r, theta)

    # Plot spectrum
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    cax = ax.pcolormesh(theta, r, np.pad(spectrum.T, ((0, 1), (0, 1)), mode='edge'), cmap=cmap, shading='auto')
    ax.axis('off')

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

@manage_figures
def plot_spectrum_compare(pred_spectrum, gt_spectrum, save_path=None):
    """
    Compare predicted and ground truth spectra in polar coordinates.

    Args:
        pred_spectrum (np.ndarray): Predicted spectrum, shape (90, 360).
        gt_spectrum (np.ndarray): Ground truth spectrum, shape (90, 360).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Normalize the spectra to [0, 1]
    # pred_spectrum = (pred_spectrum - np.min(pred_spectrum)) / (np.max(pred_spectrum) - np.min(pred_spectrum) + 1e-8)
    # gt_spectrum = (gt_spectrum - np.min(gt_spectrum)) / (np.max(gt_spectrum) - np.min(gt_spectrum) + 1e-8)

    # Create a polar grid
    r = np.linspace(0, 1, 91)  # Radial distance
    theta = np.linspace(0, 2 * np.pi, 361)
    r, theta = np.meshgrid(r, theta)

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    # Plot predicted spectrum
    cax1 = axs[0].pcolormesh(theta, r, np.flipud(pred_spectrum).T, cmap='viridis', shading='auto')
    axs[0].set_title("Predicted Spectrum")
    axs[0].axis('off')

    # Plot ground truth spectrum
    cax2 = axs[1].pcolormesh(theta, r, np.flipud(gt_spectrum).T, cmap='viridis', shading='auto')
    axs[1].set_title("Ground Truth Spectrum")
    axs[1].axis('off')

    # Save or display the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)


