Tree:

```bash
├── cuda_rasterizer
│   ├── __init__.py
│       └──
├── cuda_wireless_rasterizer
│   ├── auxiliary.h
│   ├── backward.cu
|   |        ├──void BACKWARD::preprocess
|   |         ├──void BACKWARD::render     
│   ├── backward.h
|   |        ├──namespace BACKWARD{
|	|                    void preprocess()
|   |                     void render() }
│   ├── config.h
│   ├── forward.cu
|   |        ├──void FORWARD::preprocess
|   |        ├──void FORWARD::render    
│   ├── forward.h
|   |       ├──namespace FORWARD{
|	|                    void preprocess()
|   |                     void render() }
│   ├── rasterizer.h
|   |       ├──namespace CudaRasterizer{
|   |             class Rasterizer{
|   |             public:
|   |               static void markVisible()
|   |               static int forward()
|   |               static void backward()}
│   ├── rasterizer_impl.cu
|   |         ├── void CudaRasterizer::Rasterizer::markVisible()
|   |         ├── int CudaRasterizer::Rasterizer::forward()
|   |                             FORWARD::preprocess()
|   |                             FORWARD::render()
|   |         ├── int CudaRasterizer::Rasterizer::backward()
│   └── rasterizer_impl.h
|            
|                
├── ext.cpp
|    ├── binds functions ├──"rasterize_gaussians", &RasterizeGaussiansCUDA
|                        ├──"rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA
|                        ├──"mark_visible", &markVisible 
├── rasterize_points.cu
|        ├── def RasterizeGaussiansCUDA:
|                CudaRasterizer::Rasterizer::forward()
|        ├── def RasterizeGaussiansBackwardCUDA:
|                CudaRasterizer::Rasterizer::backward()
|        ├── def markVisible:
|                CudaRasterizer::Rasterizer::markVisible()

├── rasterize_points.h
|        ├── RasterizeGaussiansCUDA
|        ├── RasterizeGaussiansBackwardCUDA
|        ├── markVisible
├── README.md
├── setup.py
|    ├──contains all the files to be compiled

```

