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

 #ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
 #define CUDA_RASTERIZER_BACKWARD_H_INCLUDED
 
 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>
 
 namespace BACKWARD
 {
     void render(
         const dim3 grid, dim3 block,
         const uint2* ranges,
         const uint32_t* point_list,
         int W, int H,
         const float2* means2D,
         const float4* conic_opacity,
         const float* out_signal_real,
         const float* out_signal_imag,
         const float* final_Ts,
         const uint32_t* n_contrib,
         const float* dL_dout_signal_real,
         const float* dL_dout_signal_imag,
         float3* dL_dmeans2D,
         float4* dL_dconic2D,
         float* dL_dopacity,
         float* dL_dsigreal,
         float* dL_dsigimag,
         float* dL_demission_mlps
         );
 
     void preprocess(
         int P, int W, int H,
         const float3* means3D,
         const float* emission_mlps,
         const int* radii,
         const bool* clamped,
         const float* opacities,
         const glm::vec3* scales,
         const glm::vec4* rotations,
         const float scale_modifier,
         const float* cov3Ds,
         const float* viewmatrix,
         const float* projmatrix,
         const glm::vec3* tx_pos,
         const glm::vec3* rx_pos,
         const float3* dL_dmeans2D,
         const float* dL_dconics,
         const float* dL_dout_signal_real,
         const float* dL_dout_signal_imag,
         float* dL_dopacity,
         glm::vec3* dL_dmeans3D,
         float* dL_dsigreal,
         float* dL_dsigimag,
         float* dL_demission_mlps,
         float* dL_dcov3D,
         glm::vec3* dL_dscale,
         glm::vec4* dL_drotations,
         float3* dpx_dt,
         float3* dpy_dt);
 }
 
 #endif
 