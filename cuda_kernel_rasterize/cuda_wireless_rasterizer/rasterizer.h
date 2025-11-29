/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

 #ifndef CUDA_RASTERIZER_H_INCLUDED
 #define CUDA_RASTERIZER_H_INCLUDED
 
 #include <vector>
 #include <functional>
 
 namespace CudaRasterizer
 {
     class Rasterizer
     {
     public:
 
         static void markVisible(
             int P,
             float* means3D,
             float* viewmatrix,
             float* projmatrix,
             bool* present);
 
         static int forward(
             std::function<char* (size_t)> geometryBuffer,
             std::function<char* (size_t)> binningBuffer,
             std::function<char* (size_t)> imageBuffer,
             const int P, 
             const int width, int height,
             const float* means3D,
             const float* emission_mlps,
             const float* opacities,
             const float* scales,
             const float scale_modifier,
             const float* rotations,
             const float* viewmatrix,
             const float* projmatrix,
             const float* tx_pos,
             const float* rx_pos,
             float* out_signal_real,
             float* out_signal_imag,
             int* radii = nullptr
            );
 
         static void backward(
             const int P, int R,
             const int width, int height,
             const float* means3D,
             const float* emission_mlps,
             const float* out_signal_real,
             const float* out_signal_imag,
             const float* opacities,
             const float* scales,
             const float scale_modifier,
             const float* rotations,
             const float* viewmatrix,
             const float* projmatrix,
             const float* tx_pos,
             const float* rx_pos,
             const int* radii,
             char* geom_buffer,
             char* binning_buffer,
             char* image_buffer,
             const float* dL_dout_signal_real,
             const float* dL_dout_signal_imag,
             float* dL_dmeans2D,
             float* dL_dconic,
             float* dL_dopacity,
             float* dL_dmeans3D,
             float* dL_dsigreal,
             float* dL_dsigimag,
             float* dL_dcov3D,
             float* dL_dscale,
             float* dL_drotations,
             float* dL_demission_mlps,
             float* dpx_dt,
             float* dpy_dt);
     };
 };
 
 #endif
 