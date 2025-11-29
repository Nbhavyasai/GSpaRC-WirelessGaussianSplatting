from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))
conda_prefix = os.environ['CONDA_PREFIX']

setup(
    name="wireless_rasterizer",  # Module name
    ext_modules=[
        CUDAExtension(
            name="wireless_rasterizer",
            sources=[
                "cuda_wireless_rasterizer/rasterizer_impl.cu",
                "cuda_wireless_rasterizer/forward.cu",
                "cuda_wireless_rasterizer/backward.cu",
                "rasterize_points.cu", 
                "ext.cpp"],  # Your C++/CUDA files
            include_dirs=['/usr/include', '/usr/local/include', conda_prefix+'/include'],
            # extra_compile_args={
            #     "cxx": ["-I/usr/include", "-I/usr/lib/gcc/x86_64-linux-gnu/11/include"],
            #     "nvcc": ["-I/usr/include", "-I/usr/lib/gcc/x86_64-linux-gnu/11/include"]
            # },
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
