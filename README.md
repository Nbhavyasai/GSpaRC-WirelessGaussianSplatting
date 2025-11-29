# GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels
This repository contains the implementation associated with the paper "GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels".

![pipeline](assets/pipeline.png)

Abstract: *Channel state information (CSI) is essential for adaptive beamforming and maintaining robust links in wireless communication systems. However, acquiring CSI incurs significant overhead, consuming up to 25% of spectrum resources in 5G networks due to frequent pilot transmissions at sub-millisecond intervals. Recent approaches aim to reduce this burden by reconstructing CSI from spatiotemporal RF measurements, such as signal strength and direction-of-arrival. While effective in offline settings, these methods often suffer from inference latencies in the 5--100~ms range, making them impractical for real-time systems. We present GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels, the first algorithm to break the 1 ms latency barrier while maintaining high accuracy. GSpaRC represents the RF environment using a compact set of 3D Gaussian primitives, each parameterized by a lightweight neural model augmented with physics-informed features such as distance-based attenuation. Unlike traditional vision-based splatting pipelines, GSpaRC is tailored for RF reception: it employs an equirectangular projection onto a hemispherical surface centered at the receiver to reflect omnidirectional antenna behavior. A custom CUDA pipeline enables fully parallelized directional sorting, splatting, and rendering across frequency and spatial dimensions. Evaluated on multiple RF datasets, GSpaRC achieves similar CSI reconstruction fidelity to recent state-of-the-art methods while reducing training and inference time by over an order of magnitude. By trading modest GPU computation for a substantial reduction in pilot overhead, GSpaRC enables scalable, low-latency channel estimation suitable for deployment in 5G and future wireless systems.*

![comparison](assets/circles.png)

## Setup

Set up the conda environment with all necessary dependencies for training and evaluation.
```bash
conda env create --file environment.yml
conda activate gsparc
```

Prepare the data by extracting the data tar files.
```bash
cd datasets/NERF2
tar -xvzf s23-dataset.tar.gz
```


## Wireless CUDA Rasterizer Compilation
Compile the custom CUDA rasterizer kernel, which is used to accelerate wireless signal projection during training.
```bash
cd cuda_kernel_rasterize
mkdir build
python setup.py install 
```
Make sure that `nvcc` uses a GCC version supported by your CUDA toolkit
(e.g., GCC 11.x for CUDA 11.8).

## Running

### Spectrum prediction

**Training a model**
Train the model on RFID spectrum data using the specified configuration. This will periodically save checkpoints for evaluation or later resumption.
```bash
python train.py \
    --mode train \
    --config configs/rfid-spectrum.yml \
    --dataset_type rfid \
    --gpu 0 \
    --checkpoint_interval 100000 \
    --debug
```

**Evaluating a model**
Evaluate a trained model by loading a specific checkpoint. Useful for generating predictions or assessing performance on the validation/test set.
```bash
python train.py \
    --mode eval \
    --config configs/rfid-spectrum.yml \
    --dataset_type rfid \
    --gpu 0 \
    --load_checkpoint <path to checkpoint> \
    --debug
```