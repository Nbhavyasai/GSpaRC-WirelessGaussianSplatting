import torch
import sys
import os
import torch.nn as nn
from typing import NamedTuple


# Import the compiled module
import wireless_rasterizer as wr


def rasterize_gaussians(means3D, means2D, emission_mlps, opacities, scales, rotations, raster_settings):
    return _RasterizeGaussians.apply(
        means3D, means2D, emission_mlps, opacities, scales, rotations, raster_settings
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        emission_mlps,
        opacities,
        scales,
        rotations,
        raster_settings
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            emission_mlps,
            opacities,
            scales,
            rotations,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tx_pos,
            raster_settings.rx_pos,
        )
        num_rendered, out_signal_real, out_signal_imag, radii, geomBuffer, binningBuffer, imgBuffer = wr.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(emission_mlps, out_signal_real, out_signal_imag, means3D, scales, rotations, opacities, radii, geomBuffer, binningBuffer, imgBuffer)

        return out_signal_real, out_signal_imag, radii
    
    @staticmethod
    def backward(ctx, grad_output_real, grad_output_imag, grad_radii):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        emission_mlps, out_signal_real, out_signal_imag, means3D, scales, rotations, opacities, radii, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        args = (means3D,
            emission_mlps,
            radii,
            out_signal_real,
            out_signal_imag,
            opacities,
            scales,
            rotations,   
            raster_settings.scale_modifier,   
            raster_settings.viewmatrix,
            raster_settings.projmatrix,       
            grad_output_real,
            grad_output_imag,
            raster_settings.tx_pos,
            raster_settings.rx_pos,
            geomBuffer,
            num_rendered,               
            binningBuffer,
            imgBuffer,              
            )

        grad_means2D, grad_means3D, grad_emission_mlps, grad_opacities, grad_cov3d, grad_scales, grad_rotations = wr.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,  # grad of means2D (screen-space)           
            grad_emission_mlps,
            grad_opacities,
            grad_scales,
            grad_rotations,
            None)

        return grads

# Define Rasterizer Setting class 
class GaussianRasterizerSetting(NamedTuple):
    image_height : int
    image_width : int
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    tx_pos : torch.Tensor
    rx_pos : torch.Tensor


# Define Rasterizer class
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings


    # Define markVisible function
    def markVisible(self, positions):
        # Mark visible points
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = wr.mark_visisble(positions, raster_settings.viewmatrix, raster_settings.projmatrix)
        return visible
    

    # Define forward function
    def forward(self, means3D, means2D, emission_mlps, opacities,  scales = None, rotations = None):
        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])

        # call the rasterize_gaussians function that invokes c++ lib
        return rasterize_gaussians(
            means3D, means2D, emission_mlps, opacities, scales, rotations,  raster_settings
        )
