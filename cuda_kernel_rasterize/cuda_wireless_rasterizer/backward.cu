#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }

__device__ void computeSignalFromMLP(
    int idx, int num_gaussians, 
    const glm::vec3* means, glm::vec3 rx_pos, glm::vec3 tx_pos, 
    const float* mlp_params, 
    float* dL_dsignalreal, float* dL_dsignalimag, 
    glm::vec3* dL_dmeans3D, float* dL_demission_mlps
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
    const float* fc1_bias = &mlp_params[offset2];
    const float* fc2_weights = &mlp_params[offset3];
    const float* fc2_bias = &mlp_params[offset4];

    glm::vec3 diff = tx_pos - means[idx];
    float d = glm::length(diff);
    float log_d = logf(d + 1e-6f);
    glm::vec3 unit_dir = diff / (d + 1e-6f);

    // Hidden layer with leaky ReLU; remember pre-activation for the backward derivative.
    float hidden_pre[hidden_size] = {0.0f};
    float hidden    [hidden_size] = {0.0f};
    for (int i = 0; i < hidden_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; ++j) {
            sum += fc1_weights[i * input_size + j] * input[j];
        }
        sum += fc1_bias[i];
        hidden_pre[i] = sum;
        hidden[i] = (sum > 0.0f) ? sum : (leaky_slope * sum);
    }

    // Output layer: linear pre-activation, then sigmoid with log-distance bias.
    // No ReLU between fc2 and sigmoid (the prior version blocked the lower half of the sigmoid).
    float pre_out       [output_size] = {0.0f};
    float sigmoid_output[output_size] = {0.0f};
    for (int i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += fc2_weights[i * hidden_size + j] * hidden[j];
        }
        sum += fc2_bias[i];
        pre_out[i] = sum;
        float attenuated = sum - log_d;
        sigmoid_output[i] = 1.0f / (1.0f + expf(-attenuated));
    }

    // ── Backward ──
    // d output_i / d attenuated_i = sigmoid * (1 - sigmoid)
    // d attenuated_i / d pre_out_i = 1
    // d attenuated_i / d log_d     = -1
    float d_output[output_size] = {
        dL_dsignalreal[idx] * sigmoid_output[0] * (1.0f - sigmoid_output[0]),
        dL_dsignalimag[idx] * sigmoid_output[1] * (1.0f - sigmoid_output[1])
    };

    // log_d gets a -1 contribution from every output (no ReLU gate).
    float dL_dlogd = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        dL_dlogd -= d_output[i];
    }

    // log_d = log(d + eps),  d/dd = 1/(d + eps)
    // d = ||diff||,           d(d)/d(diff) = unit_dir
    // diff = tx_pos - means,  d(diff)/d(means) = -I
    // => dL/dmeans = -(dL/dlogd / (d+eps)) * unit_dir
    float dL_dd = dL_dlogd / (d + 1e-6f);
    glm::vec3 dL_dmean = -dL_dd * unit_dir;
    dL_dmeans3D[idx] += dL_dmean;

    float d_hidden[hidden_size] = {0.0f};
    float d_input[input_size] = {0.0f};

    // fc2 backprop  (d_output[i] is dL/dpre_out[i], since pre_out -> attenuated is just a shift)
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            dL_demission_mlps[offset3 + i * hidden_size + j] = d_output[i] * hidden[j];
            d_hidden[j] += d_output[i] * fc2_weights[i * hidden_size + j];
        }
        dL_demission_mlps[offset4 + i] = d_output[i];
    }

    // fc1 backprop with leaky-ReLU derivative
    for (int i = 0; i < hidden_size; ++i) {
        float dpre = (hidden_pre[i] > 0.0f) ? d_hidden[i] : (leaky_slope * d_hidden[i]);
        for (int j = 0; j < input_size; ++j) {
            dL_demission_mlps[offset1 + i * input_size + j] = dpre * input[j];
            d_input[j] += dpre * fc1_weights[i * input_size + j];
        }
        dL_demission_mlps[offset2 + i] = dpre;
    }

    // (Optional) use d_input if tx_pos / rx_pos is learnable
}


// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int width,
	const int height,
	const int* radii,
	const float* cov3Ds,
	const float* view_matrix,
	const float* opacities,
	const float* dL_dconics,
	float* dL_dopacity,
	float3* dL_dmeans3D,
	float* dL_dcov,
	float3* dpx_dt,
	float3* dpy_dt)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);

	float txtx = t.x * t.x;
	float tyty = t.y * t.y;
	float tztz = t.z * t.z;
	float txtytz = t.x * t.y * t.z;

	float trxztrxz = txtx + tztz;
	float trxztrxz_inv = 1.0f / (trxztrxz + 0.0000001f);
	float trxztrxztrxztrxz_inv = trxztrxz_inv * trxztrxz_inv;
	float trxz = sqrtf(trxztrxz);
	float trxz_inv = 1.0f / (trxz + 0.0000001f);
	float trtr = trxztrxz + tyty;
	float trtr_inv = 1.0f / (trtr + 0.0000001f);
	float trtrtrtr_inv = trtr_inv * trtr_inv;
	float trxz_trtrtrtr_inv = trxz_inv * trtrtrtr_inv;
	float trxztrxztrxz_trtrtrtr_inv = trxztrxz_inv * trxz_trtrtrtr_inv;
	float tyty_minus_trxztrxz = tyty - trxztrxz;

	float W_div_2pi = width * 0.5f * M_1_PIf32;
	float H_div_pi = height * M_1_PIf32;

	float dpx_dtx = W_div_2pi * t.z * trxztrxz_inv * 0.5f;
	float dpx_dtz = -W_div_2pi * t.x * trxztrxz_inv * 0.5f;

	float dpy_dtx = -H_div_pi * t.x * t.y * trxz_inv * trtr_inv;
	float dpy_dty = H_div_pi * trxz * trtr_inv;
	float dpy_dtz = -H_div_pi * t.z * t.y * trxz_inv * trtr_inv;

	dpx_dt[idx].x = dpx_dtx;
	dpx_dt[idx].y = 0.0f;
	dpx_dt[idx].z = dpx_dtz;

	dpy_dt[idx].x = dpy_dtx;
	dpy_dt[idx].y = dpy_dty;
	dpy_dt[idx].z = dpy_dtz;

	glm::mat3 J = glm::mat3(
		dpx_dtx, 0.0f, dpx_dtz,
		dpy_dtx, dpy_dty, dpy_dtz,
		0.0f, 0.0f, 0.0f); // actually J^T

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]); // actually W^T

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	// Low-pass filtering is implemented with direct addition of variance without generating additional gradients
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		// Actually the dL_dconic.y calculated by renderCUDA before was half the correct value. Here each is *2 and the final result is correct (presumably so that dL_db can come up with a common factor of 2)
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else // Numerical stability: the gradient disappears when the determinant of the covariance matrix is super large, in order to prevent the multiplication denom from becoming inf, it is directly 0 and no longer processed
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ10 = W[0][0] * dL_dT10 + W[0][1] * dL_dT11 + W[0][2] * dL_dT12;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	// [Camera model] From here, the internal definition of J is involved, which is equivalent to finding the sum of the second-order derivatives of the proj function with respect to t
	float temp1 = H_div_pi * tyty_minus_trxztrxz * trxz_trtrtrtr_inv;
	float temp2 = H_div_pi * txtytz * (trtr + 2.0f * trxztrxz) * trxztrxztrxz_trtrtrtr_inv;
	float temp3 = W_div_2pi * (txtx - tztz) * trxztrxztrxztrxz_inv;
	float temp4 = W_div_2pi * 2.0f * t.x * t.z * trxztrxztrxztrxz_inv;
	float temp5 = H_div_pi * t.y * trxztrxztrxz_trtrtrtr_inv;

	float dL_dtx = -dL_dJ00 * temp4
	               +dL_dJ02 * temp3
				   +dL_dJ10 * temp5 * (2.0f * txtx * trxztrxz - tztz * trtr)
				   +dL_dJ11 * t.x * temp1
				   +dL_dJ12 * temp2;

	float dL_dty =  dL_dJ10 * t.x * temp1
	               -dL_dJ11 * H_div_pi * 2.0f * trxz * t.y * trtrtrtr_inv
				   +dL_dJ12 * t.z * temp1;

	float dL_dtz =  dL_dJ00 * temp3
	               +dL_dJ02 * temp4
				   +dL_dJ10 * temp2
				   +dL_dJ11 * t.z * temp1
				   +dL_dJ12 * temp5 * (2.0f * tztz * trxztrxz - txtx * trtr);


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans3D[idx] = dL_dmean;
}


// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}



// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, 
	const float3* means3D,
	const float* emission_mlps,
	const int width,
	const int height,
	const int* radii,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view_matrix,
	const float* proj,
	const glm::vec3* tx_pos,
	const glm::vec3* rx_pos,
	const float3* dpx_dt,
	const float3* dpy_dt,
	const float3* dL_dmeans2D,
	const float* dL_dout_signal_real,
    const float* dL_dout_signal_imag,
	glm::vec3* dL_dmeans3D,
	float* dL_dsigreal,
	float* dL_dsigimag,
	float* dL_demission_mlps,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dopacity)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Taking care of gradients from the screenspace points
	float dsx_dpx = 2.0f / (float)width;
	float dsy_dpy = 2.0f / (float)height;

	float dL_dpx = dL_dmeans2D[idx].x * dsx_dpx;
	float dL_dpy = dL_dmeans2D[idx].y * dsy_dpy;

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	// [Camera model] Previously computeCov2DCUDA calculated indirect 3D-2D gradient components conducted through cov, now it calculates 3D-2D gradient components due to 3D-2D direct coordinate transformation
	float dL_dtx = dL_dpx * dpx_dt[idx].x + dL_dpy * dpy_dt[idx].x;
	float dL_dty = dL_dpx * dpx_dt[idx].y + dL_dpy * dpy_dt[idx].y;
	float dL_dtz = dL_dpx * dpx_dt[idx].z + dL_dpy * dpy_dt[idx].z;
	float3 dL_dmean3D = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);
	glm::vec3 dL_dmean(dL_dmean3D.x, dL_dmean3D.y, dL_dmean3D.z);

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans3D[idx] += dL_dmean;

	computeSignalFromMLP(idx, P, (glm::vec3*)means3D, *rx_pos, *tx_pos, emission_mlps, dL_dsigreal, dL_dsigimag, (glm::vec3*)dL_dmeans3D, dL_demission_mlps);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}



// Backward version of the rendering procedure.
template <uint32_t C = 1> // Default C to 1
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ out_signal_real,
    const float* __restrict__ out_signal_imag,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dout_signal_real,
    const float* __restrict__ dL_dout_signal_imag,
	float3* __restrict__ dL_dmeans2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dsigreal,
	float* __restrict__ dL_dsigimag,
	float* __restrict__ dL_demission_mlps
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_signal_real[C * BLOCK_SIZE];
    __shared__ float collected_signal_imag[C * BLOCK_SIZE];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec_real[C] = { 0 };
	float accum_rec_imag[C] = { 0 };
	float dL_dpixel_real[C];
    float dL_dpixel_imag[C];


	if (inside)
	{
		for (int i = 0; i < C; i++){
			dL_dpixel_real[i] = dL_dout_signal_real[i * H * W + pix_id];
            dL_dpixel_imag[i] = dL_dout_signal_imag[i * H * W + pix_id];
		}
	}

	float last_alpha = 0;
	float last_signal_real[C] = { 0 };
	float last_signal_imag[C] = { 0 };


	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in reverse order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();

		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++) {
				collected_signal_real[i * BLOCK_SIZE + block.thread_rank()] = out_signal_real[coll_id * C + i];
				collected_signal_imag[i * BLOCK_SIZE + block.thread_rank()] = out_signal_imag[coll_id * C + i];
			}
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dsignal = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha_real = 0.0f;
			float dL_dalpha_imag = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float sig_real = collected_signal_real[ch * BLOCK_SIZE + j];
				const float sig_imag = collected_signal_real[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec_real[ch] = last_alpha * last_signal_real[ch] + (1.f - last_alpha) * accum_rec_real[ch];
				accum_rec_imag[ch] = last_alpha * last_signal_imag[ch] + (1.f - last_alpha) * accum_rec_imag[ch];
				last_signal_real[ch] = sig_real;
				last_signal_imag[ch] = sig_imag ;

				const float dL_dchannel_real = dL_dpixel_real[ch];
				const float dL_dchannel_imag = dL_dpixel_imag[ch];
				dL_dalpha_real += (sig_real - accum_rec_real[ch]) * dL_dchannel_real;
				dL_dalpha_imag += (sig_imag - accum_rec_real[ch]) * dL_dchannel_real;
				atomicAdd(&(dL_dsigreal[global_id * C + ch]), dchannel_dsignal * dL_dchannel_real);
				atomicAdd(&(dL_dsigimag[global_id * C + ch]), dchannel_dsignal * dL_dchannel_imag);
			}

			dL_dalpha_real *= T;
			dL_dalpha_imag *= T;
			last_alpha = alpha;

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha_real;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmeans2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmeans2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha_real);
		}
	}
}

void BACKWARD::preprocess(
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
	const float* dL_dconic,
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
	float3* dpy_dt)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		W,
		H,
		radii,
		cov3Ds,
		viewmatrix,
		opacities,
		dL_dconic,
		dL_dopacity,
		(float3*)dL_dmeans3D,
		dL_dcov3D,
		dpx_dt,
		dpy_dt);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desired), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, 
		(float3*)means3D,
		emission_mlps,
		W,
		H,
		radii,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		tx_pos,
		rx_pos,
		dpx_dt,
		dpy_dt,
		(float3*)dL_dmeans2D,
		dL_dout_signal_real,
    	dL_dout_signal_imag,
		(glm::vec3*)dL_dmeans3D,
		dL_dsigreal,
		dL_dsigimag,
		dL_demission_mlps,
		dL_dcov3D,
		dL_dscale,
		dL_drotations,
		dL_dopacity);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
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
	float* dL_demission_mlps)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		out_signal_real,
        out_signal_imag,
		final_Ts,
		n_contrib,
		dL_dout_signal_real,
        dL_dout_signal_imag,
		dL_dmeans2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dsigreal,
		dL_dsigimag,
		dL_demission_mlps
	);
}