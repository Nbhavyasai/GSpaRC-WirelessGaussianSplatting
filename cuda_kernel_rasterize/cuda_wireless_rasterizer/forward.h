#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P,
		const float* means3D,
		const float* emission_mlps,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		bool* clamped,
		const float* viewmatrix,
		const float* projmatrix,
    	const glm::vec3* tx_pos,
		const glm::vec3* rx_pos,
		const int W, int H,
		int* radii,
		float2* means2D,
		float* depths,
		float* cov3Ds,
		float* signal_reals,
		float* signal_imags,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float* features_real,
		const float* features_imag,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		float* out_signal_real,
		float* out_signal_imag);
}

#endif