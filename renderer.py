######Render function that returns desired spatial spectrum#######
import torch
import math
from gaussian import GaussianModel
from cuda_kernel_rasterize.cuda_rasterizer import GaussianRasterizerSetting, GaussianRasterizer

def world_to_view_transform(centers_3D, view_matrix):

    '''
    Transforms Gaussian mean coordinates into view coordinates using a view transformation matrix.

    Parameters:
        centers_3D (torch.Tensor): A tensor of shape (N, 3), where N is the number of Gaussians.
                                       Each row is a 3D point [x, y, z].
        view_matrix (torch.Tensor): A 4x4 view transformation matrix as.

    Returns:
        centers_view (torch.Tensor): Gaussian means in view coordinates (N, 3).
    '''
    # Ensure input tensors have the correct shape
    if centers_3D.shape[1] != 3:
        raise ValueError("Centers should have shape (N, 3).")
    if view_matrix.shape != (4, 4):
        raise ValueError("View matrix should have shape (4, 4).")
    
    # Add a fourth dimension (homogeneous coordinate) to the centers_3D
    ones = torch.ones((centers_3D.shape[0], 1), dtype=centers_3D.dtype, device=centers_3D.device)
    centers_hom = torch.cat([centers_3D, ones], dim=1)  # Shape: (N_g, 4)

    # Apply the view transformation matrix to get homogeneous coordinates
    centers_view_hom = torch.matmul(centers_hom, view_matrix.T)  # Shape: (N_g, 4)

    # Convert back from homogeneous coordinates to 3D coordinates (no division as no projective transform)
    centers_view = centers_view_hom[:, :3]

    return centers_view


def render(wireless_data, pc : GaussianModel, wavelength=0, scaling_modifier=1.0):
    """
    Render the signal. 
    """
    # Get the centre, scaling, rotation and opacity of the Gaussian model
    centers_3D = pc.get_centre
    scales_3D = pc.get_scaling
    rots_3D = pc.get_rotation
    opacities = pc.get_opacity
    emission_mlps = pc.get_emission_mlps

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_centre, dtype=pc.get_centre.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    centers_2D = screenspace_points

    raster_settings = GaussianRasterizerSetting(
        image_height=int(wireless_data.spectrum_elevation),
        image_width=int(wireless_data.spectrum_azimuth),
        scale_modifier=scaling_modifier,
        viewmatrix=wireless_data.view_matrix,
        projmatrix=torch.eye(4).cuda(),
        tx_pos = wireless_data.tx_pos,
        rx_pos = wireless_data.rx_pos
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # rasterize the gaussians
    signal_real, signal_imag, radii = rasterizer(
            means3D = centers_3D,
            means2D = centers_2D,
            emission_mlps = emission_mlps,
            opacities = opacities,
            scales = scales_3D,
            rotations = rots_3D)

    #Return the signal
    out = {
        "render": (signal_real.squeeze(0),signal_imag.squeeze(0)),
        "radii": radii,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero()

    }

    return out