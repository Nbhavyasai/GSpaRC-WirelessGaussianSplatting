#define DEBUG_MODE 0  // Set to 0 to disable all prints

#if DEBUG_MODE
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_wireless_rasterizer/config.h"
#include "cuda_wireless_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

// torch::Tensor RasterizeGaussiansCUDA(torch::Tensor a, torch::Tensor b) {
//     return a + b;  // Simple tensor addition (this is just a placeholder)
// }


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& means3D,         // 3D positions of Gaussians
    const torch::Tensor& emission_mlps,     // MLPs to compute emission
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
)
{

    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    

    const int P = means3D.size(0); // Number of points
    const int H = image_height;    // Image height
    const int W = image_width;     // Image width

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    //std::cout << "Inside CUDA rasterizer" << std::endl;

    torch::Tensor out_signal_real = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor out_signal_imag = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);

    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
    
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
    
    int rendered = 0;
    if(P != 0)
    {
        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            P, 
            W, H,
            means3D.contiguous().data_ptr<float>(),
            emission_mlps.contiguous().data_ptr<float>(),
            opacities.contiguous().data_ptr<float>(), 
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(), 
            projmatrix.contiguous().data_ptr<float>(),
            tx_pos.contiguous().data_ptr<float>(),
            rx_pos.contiguous().data_ptr<float>(),
            out_signal_real.contiguous().data_ptr<float>(),
            out_signal_imag.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>()
        );
    }

    return std::make_tuple(rendered, out_signal_real, out_signal_imag, radii, geomBuffer, binningBuffer, imgBuffer);    
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    const torch::Tensor& rx_pos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer)
{
  const int P = means3D.size(0);
  const int H = dL_dout_signal_real.size(1);
  const int W = dL_dout_signal_real.size(2);

  int M = 0;
  if(emission_mlps.size(0) != 0)
  {	
	M = emission_mlps.size(0);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_demission_mlps = torch::zeros({M}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dsigreal = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dsigimag = torch::zeros({P, 1}, means3D.options());


  if(P != 0)
  {  

      torch::Tensor dpx_dt = torch::zeros({P, 3}, means3D.options());
      torch::Tensor dpy_dt = torch::zeros({P, 3}, means3D.options());

	  CudaRasterizer::Rasterizer::backward(P, R,
	  W, H, 
	  means3D.contiguous().data_ptr<float>(),
      emission_mlps.contiguous().data_ptr<float>(),
	  out_signal_real.contiguous().data_ptr<float>(),
      out_signal_imag.contiguous().data_ptr<float>(),
	  opacities.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  tx_pos.contiguous().data_ptr<float>(),
      rx_pos.contiguous().data_ptr<float>(),
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_signal_real.contiguous().data_ptr<float>(),
      dL_dout_signal_imag.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),  
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
      dL_dsigreal.contiguous().data_ptr<float>(),
      dL_dsigimag.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
      dL_demission_mlps.contiguous().data_ptr<float>(),
      dpx_dt.contiguous().data_ptr<float>(),
      dpy_dt.contiguous().data_ptr<float>());
  }

  return std::make_tuple(
    dL_dmeans2D,
    dL_dmeans3D,
    dL_demission_mlps,
    dL_dopacity,
    dL_dcov3D,
    dL_dscales,
    dL_drotations
    );
}



torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix)
{ 
const int P = means3D.size(0);

torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

if(P != 0)
{
CudaRasterizer::Rasterizer::markVisible(P,
    means3D.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    present.contiguous().data<bool>());
}

return present;
}