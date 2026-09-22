# Project Submission

## Project Title

CUDA GPU-Accelerated Image Processing and Enhancement

## Public Repository

[https://github.com/TechBot-X/cuda-gpu-image-processing-capstone](https://github.com/TechBot-X/cuda-gpu-image-processing-capstone)

## Project Description

This CUDA C++ application loads an RGB PPM image, transfers its pixels from CPU host memory to NVIDIA GPU global memory, processes pixels in parallel, transfers the result back, saves output images, and records CPU/GPU timing data. It demonstrates a complete host-to-device-to-host image-processing pipeline suitable for a headless Linux CUDA environment.

## CUDA/GPU Usage

Image processing is suitable for GPU parallelism because the same arithmetic is applied independently to many pixels. The project contains four `__global__` kernels: `grayscale_kernel`, `enhance_kernel`, `blur_kernel`, and `edge_kernel`. Each thread computes one output pixel using 2D coordinates derived from `blockIdx`, `blockDim`, and `threadIdx`.

## Implemented Features

- Grayscale conversion using RGB luminance weights.
- Brightness and contrast enhancement.
- 3x3 weighted Gaussian-style blur.
- Sobel edge detection.
- CPU reference implementations for comparison.
- PPM `P3` and `P6` input; binary PPM output.
- CSV timing output and an `all` operation for all four images.

## GPU Architecture

- 16x16 CUDA thread blocks and a rounded-up 2D grid.
- One thread processes one image pixel.
- RGB arrays reside in device global memory after `cudaMalloc` and host-to-device `cudaMemcpy`.
- Results return through device-to-host `cudaMemcpy`.
- `cudaGetLastError`, the `CUDA_CHECK` wrapper, and `cudaDeviceSynchronize()` handle errors and completion.
- CUDA events measure host-to-device transfer, kernel execution, and device-to-host transfer.
- `cudaFree` and `cudaEventDestroy` release device resources.

## Hardware and Software

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Driver: 616.92
- CUDA Toolkit: 13.3.73
- Compilation architecture: `sm_89`
- Environment: WSL Ubuntu 24.04
- Compiler: `nvcc`
- Host language standard: C++17

## Execution Evidence

The repository includes `results/gpu_info.txt`, `results/execution_log.txt`, `results/output.txt`, `results/performance.txt`, and `results/performance.csv`. Generated outputs are `output/grayscale.ppm`, `output/enhanced.ppm`, `output/blurred.ppm`, and `output/edges.ppm`.

## Performance Results

The supplied demonstration image is only 8x8 pixels. The latest verified run recorded:

```text
operation  CPU ms  GPU kernel ms  H2D ms  D2H ms
grayscale  0.001   1.360           0.411   0.118
enhance    0.002   0.341           0.176   0.067
blur       0.003   0.109           0.069   0.104
edge       0.002   0.354           0.095   0.169
```

These measurements demonstrate successful CUDA GPU execution. They do **not** claim GPU speedup because the tiny image is dominated by launch and transfer overhead. Meaningful CPU-versus-GPU benchmarking requires a much larger image, such as 1920x1080, repeated trials, and the same parameters for both implementations.

## Reproducibility

From the repository root on a CUDA-enabled Linux system:

```bash
make clean
make info
make
make run
```

The direct equivalent is:

```bash
./cuda_image_processing --input input/sample.ppm --output output --operation all
```

## Outputs

- `output/grayscale.ppm`
- `output/enhanced.ppm`
- `output/blurred.ppm`
- `output/edges.ppm`
- `results/performance.csv`
- GPU and execution evidence under `results/`

## Repository Contents

- CUDA source: `src/main.cu`, `src/image_processing.cu`, `src/image_processing.h`
- Build system: `Makefile`
- Documentation: `README.md`, `PROJECT_SUBMISSION.md`, `PRESENTATION.md`
- Reproducible input: `input/sample.ppm`
- Generated outputs and evidence: `output/`, `results/`

## Future Improvements

Future work includes meaningful large-image benchmarking, repeated statistical trials, shared-memory tiling for neighborhood kernels, pinned host memory, CUDA streams, additional filters, and optional PNG/JPEG support.

## Coursera Submission Description

**Project title:** CUDA GPU-Accelerated Image Processing and Enhancement

This project is a CUDA C++ image-processing application that uses four GPU kernels for grayscale conversion, brightness/contrast enhancement, Gaussian-style blur, and Sobel edge detection. It explicitly demonstrates device allocation, host-to-device and device-to-host transfers, 2D 16x16 CUDA blocks, per-pixel parallelism, synchronization, CUDA error checking, and CUDA event timing. It was verified on an NVIDIA GeForce RTX 4060 Laptop GPU with CUDA Toolkit 13.3.73 using `sm_89`. The repository contains source code, build instructions, generated outputs, `performance.csv`, and GPU execution evidence. The supplied 8x8 image demonstrates successful CUDA execution and functionality; it is too small to support a meaningful GPU speedup claim.

Repository: [https://github.com/TechBot-X/cuda-gpu-image-processing-capstone](https://github.com/TechBot-X/cuda-gpu-image-processing-capstone)
