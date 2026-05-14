#define DEBUG_MODE 0  // Set to 0 to disable all prints

#if DEBUG_MODE
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the emission MLP parameters
// to signal real and imaginary components.
__device__ glm::vec3 computeSignalFromMLP(
    int idx, int num_gaussians, 
    const float* mlp_params, 
    const glm::vec3* means, 
    glm::vec3 rx_pos, glm::vec3 tx_pos
) 
{
    const int input_size = 3;
    const int hidden_size = 32;
    const int output_size = 2;
    const float leaky_slope = 0.01f;

    float input[input_size] = {rx_pos.x, rx_pos.y, rx_pos.z};

    int fc1_weights_size = hidden_size * input_size;
    int fc1_bias_size = hidden_size;
    int fc2_weights_size = output_size * hidden_size;
    int fc2_bias_size = output_size;

    int offset1 = idx * fc1_weights_size;
    int offset2 = num_gaussians * fc1_weights_size + idx * fc1_bias_size;
    int offset3 = num_gaussians * fc1_weights_size + num_gaussians * fc1_bias_size + idx * fc2_weights_size;
    int offset4 = num_gaussians * fc1_weights_size + num_gaussians * fc1_bias_size + num_gaussians * fc2_weights_size + idx * fc2_bias_size;

    const float* fc1_weights = &mlp_params[offset1];
    const float* fc1_bias    = &mlp_params[offset2];
    const float* fc2_weights = &mlp_params[offset3];
    const float* fc2_bias    = &mlp_params[offset4];

    glm::vec3 diff = tx_pos - means[idx];
    float d = glm::length(diff);
    float log_d = logf(d + 1e-6f);

    // Hidden layer with leaky ReLU
    float hidden[hidden_size] = {0.0f};
    for (int i = 0; i < hidden_size; ++i) {
        float sum = fc1_bias[i];
        for (int j = 0; j < input_size; ++j) {
            sum += fc1_weights[i * input_size + j] * input[j];
        }
        hidden[i] = (sum > 0.0f) ? sum : (leaky_slope * sum);
    }

    // Output: linear pre-activation (NO output ReLU), then sigmoid with log-distance bias.
    // This preserves the (0, 1) output range the rasterizer expects, while letting
    // the network reach the lower half of the sigmoid (which the prior ReLU blocked).
    float output[output_size] = {0.0f};
    for (int i = 0; i < output_size; ++i) {
        float sum = fc2_bias[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += fc2_weights[i * hidden_size + j] * hidden[j];
        }
        float attenuated = sum - log_d;                      // no ReLU here
        float sigmoid_val = 1.0f / (1.0f + expf(-attenuated));
        output[i] = sigmoid_val;
    }

    return glm::vec3(output[0], output[1], 0.0f);
}


__device__ float3 computeCov2D(const float3& mean, const int width, const int height, const float* cov3D, const float* viewmatrix)
{
    float3 t = transformPoint4x3(mean, viewmatrix);

    float trxztrxz = t.x * t.x + t.z * t.z;
    float trxztrxz_inv = 1.0f / (trxztrxz + 1e-7f);
    float trxz = sqrtf(trxztrxz);
    float trxz_inv = 1.0f / (trxz + 1e-7f);
    float trtr = trxztrxz + t.y * t.y;
    float trtr_inv = 1.0f / (trtr + 1e-7f);

    float W_div_2pi = width * 0.5f * M_1_PIf32;
    float H_div_pi = height * M_1_PIf32;

    float dpx_dtx = W_div_2pi * t.z * trxztrxz_inv;
    float dpx_dtz = -W_div_2pi * t.x * trxztrxz_inv;

    float dpy_dtx = -H_div_pi * t.x * t.y * trxz_inv * trtr_inv;
    float dpy_dty = H_div_pi * trxz * trtr_inv;
    float dpy_dtz = -H_div_pi * t.z * t.y * trxz_inv * trtr_inv;

    glm::mat3 J = glm::mat3(
        dpx_dtx, 0.0f, dpx_dtz,
        dpy_dtx, dpy_dty, dpy_dtz,
        0.0f, 0.0f, 0.0f);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    glm::vec4 q = rot;
    float r = q.x, x = q.y, y = q.z, z = q.w;

    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;
    glm::mat3 Sigma = glm::transpose(M) * M;

    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features_real,
    const float* __restrict__ features_imag,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    float* __restrict__ out_signal_real,
    float* __restrict__ out_signal_imag)
{

    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    float T = 1.0f;
    float bg_color = 0.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float sig_real[CHANNELS] = { 0 };
    float sig_imag[CHANNELS] = { 0 };

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            contributor++;
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)
                continue;

            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f / 255.0f)
                continue;

            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }

            for (int ch = 0; ch < CHANNELS; ch++) {
                sig_real[ch] += features_real[collected_id[j] * CHANNELS + ch] * alpha * T;
                sig_imag[ch] += features_imag[collected_id[j] * CHANNELS + ch] * alpha * T;
            }
			
            T = test_T;
            last_contributor = contributor;
        }
    }

    if (inside)
    {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;
        for (int ch = 0; ch < CHANNELS; ch++) {
            out_signal_real[ch * H * W + pix_id] = sig_real[ch] + T * bg_color;
            out_signal_imag[ch * H * W + pix_id] = sig_imag[ch] + T * bg_color;
			DEBUG_PRINT("out_signal_real=%.3f\n", out_signal_real[ch * H * W + pix_id]);
        }
    }
}

// ==================== preprocessCUDA Kernel ====================
template<int C>
__global__ void preprocessCUDA(int P,
    const float* orig_points,
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
    float2* points_xy_image,
    float* depths,
    float* cov3Ds,
    float* signal_reals,
    float* signal_imags,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float4 p_view;
    if (too_close(p_orig, viewmatrix, p_view)) {
        DEBUG_PRINT("[Preprocess] IDX %d CULLED: Too close\n", idx);
        return;
    }
    
    float2 p_proj = transformProj(p_view);
    
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    const float* cov3D = cov3Ds + idx * 6;
    float3 cov = computeCov2D(p_orig, W, H, cov3D, viewmatrix);

    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f) {
        return;
    }

    float det_inv = 1.f / det;
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrtf(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrtf(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) {
        return;
    }
    
    // Compute signal real and imaginary using emission MLP
    glm::vec3 result = computeSignalFromMLP(idx, P, emission_mlps, (glm::vec3*)orig_points, *rx_pos, *tx_pos);
    signal_reals[idx] = result.x;
    signal_imags[idx] = result.y;

    depths[idx] = p_view.w;
    radii[idx] = my_radius;
    points_xy_image[idx] = point_image;
    float opacity = opacities[idx];
    conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity };
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* signal_real,
    const float* signal_imag,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	float* out_signal_real,
    float* out_signal_imag)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		signal_real,
        signal_imag,
		conic_opacity,
		final_T,
		n_contrib,
		out_signal_real,
        out_signal_imag);
}


void FORWARD::preprocess(int P,
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
	uint32_t* tiles_touched)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, 
		means3D,
        emission_mlps,
		scales,
		scale_modifier,
		rotations,
		opacities,
		clamped,
		viewmatrix, 
		projmatrix,
		tx_pos,
        rx_pos,
		W, H,
		radii,
		means2D,
		depths,
		cov3Ds,
		signal_reals,
        signal_imags,
		conic_opacity,
		grid,
		tiles_touched
		);
}