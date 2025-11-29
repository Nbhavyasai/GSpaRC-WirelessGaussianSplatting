#include <torch/extension.h>  // Required for PyTorch C++ Extensions
#pragma once
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& means3D,         // 3D positions of Gaussians
    const torch::Tensor& emission_mlps,
    const torch::Tensor& opacities,       // Opacity values per Gaussian
    const torch::Tensor& scales,          // Scale values per Gaussian
    const torch::Tensor& rotations,       // Rotation matrices per Gaussian
    const int image_height,              // Image height
    const int image_width,               // Image width
    const float scale_modifier,          // Scale modifier for Gaussian size
    const torch::Tensor& viewmatrix,      // view matrix
    const torch::Tensor& projmatrix,      // projection matrix
    const torch::Tensor& tx_pos,          // Transmitter position
    const torch::Tensor& rx_pos           // Receiver position
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,  torch::Tensor,  torch::Tensor, torch::Tensor,  torch::Tensor> 
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& emission_mlps,
    const torch::Tensor& radii,
    const torch::Tensor& out_signal_real,
    const torch::Tensor& out_signal_imag,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& dL_dout_signal_real,
    const torch::Tensor& dL_dout_signal_imag,
    const torch::Tensor& tx_pos,
    const torch::Tensor& rx_pos,          // Receiver position
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer
);

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix
);

