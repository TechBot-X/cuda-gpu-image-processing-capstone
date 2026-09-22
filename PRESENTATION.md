# Presentation and Demo Script

This 15-slide script is designed for a 5-10 minute Coursera peer-review video. Spend approximately 20-35 seconds per slide and use the live terminal demonstration on Slides 9-10.

## Slide 1 - Title

**Display:** CUDA GPU-Accelerated Image Processing and Enhancement.

**Say:** This project is a CUDA C++ image-processing application that demonstrates genuine NVIDIA GPU parallel processing from input loading through output generation.

## Slide 2 - Problem and Motivation

**Display:** An image grid with many independent pixels.

**Say:** Grayscale conversion, enhancement, blur, and edge detection repeat similar arithmetic over many pixels. That makes the workload a natural example of data parallelism. The current sample is deliberately small for a portable smoke test, while larger images are the appropriate target for throughput benchmarking.

## Slide 3 - Project Objective

**Display:** Process pixels on the GPU, compare with CPU, and measure transfers and kernels.

**Say:** The objective is to show explicit CUDA concepts rather than hide them inside an image library: device allocation, memory copies, kernel launches, synchronization, error checking, and measured CUDA event timing.

## Slide 4 - Image Processing Pipeline

**Display:** Input PPM -> host RGB vector -> GPU buffers -> CUDA kernel -> host result -> output PPM and CSV.

**Say:** The CPU loads a PPM image, the program copies its RGB bytes to device global memory, launches one operation, copies the result back, saves a PPM image, and appends timing data to `results/performance.csv`.

## Slide 5 - CUDA/GPU Architecture

**Display:** Four kernels: grayscale, enhance, blur, edge.

**Say:** The four CUDA kernels are `grayscale_kernel`, `enhance_kernel`, `blur_kernel`, and `edge_kernel`. Grayscale and enhancement are per-pixel operations. Blur and Sobel edge detection read a small clamped neighborhood, but each output pixel is still independently computed.

## Slide 6 - CUDA Kernels

**Display:** The 2D indexing code:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

**Say:** Each thread owns one pixel. `blockIdx` selects the block, `blockDim` gives its dimensions, and `threadIdx` identifies the thread within it. Boundary checks protect the extra threads in the rounded-up grid.

## Slide 7 - Memory Transfers and Synchronization

**Display:** `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaFree`.

**Say:** The application allocates input and output buffers with `cudaMalloc`, performs host-to-device and device-to-host `cudaMemcpy` calls, checks launch errors, synchronizes with `cudaDeviceSynchronize`, and releases resources. The `CUDA_CHECK` wrapper converts runtime failures into readable errors.

## Slide 8 - Hardware and Software Environment

**Display:** GPU and toolkit evidence from `results/gpu_info.txt`.

**Say:** The verified environment is WSL Ubuntu 24.04 with an NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.92, CUDA Toolkit 13.3.73, and `sm_89`. The project uses `nvcc`, C++17, `make`, and the self-contained PPM format.

## Slide 9 - Execution Demo: Build

**Display:** Terminal commands.

```bash
make clean
make info
make
```

**Say:** These commands remove the previous executable and generated images, show the GPU and compiler, and compile the unchanged CUDA implementation with the verified architecture.

## Slide 10 - Execution Demo: Run

**Display:** Runtime command and console output.

```bash
make run
cat results/performance.csv
```

**Say:** `make run` executes all four operations and writes the output images. The console reports CPU time, GPU kernel time, and both transfer times. No CUDA runtime errors occurred in the verified run.

## Slide 11 - Performance Results

**Display:** The actual rows from `results/performance.csv`.

```text
grayscale,8,8,0.001,1.360,0.411,0.118
enhance,8,8,0.002,0.341,0.176,0.067
blur,8,8,0.003,0.109,0.069,0.104
edge,8,8,0.002,0.354,0.095,0.169
```

**Say:** These are measured values from the RTX 4060 run. Because the supplied image is only 8x8, launch and transfer overhead dominate. I am not claiming GPU speedup from this tiny workload. A meaningful comparison needs a much larger image and repeated trials.

## Slide 12 - Output Images

**Display:** `grayscale.ppm`, `enhanced.ppm`, `blurred.ppm`, and `edges.ppm` converted for viewing if desired.

**Say:** The outputs demonstrate the behavior of each operation: grayscale removes color, enhancement changes brightness and contrast, blur softens local detail, and Sobel detection highlights contours.

## Slide 13 - Limitations

**Display:** 8x8 sample, PPM format, global-memory neighborhood reads.

**Say:** The current demonstration image is intentionally 8x8 and is for execution and functionality verification, not speedup measurement. The implementation uses PPM for portability and keeps neighborhood operations simple rather than using shared-memory tiling.

## Slide 14 - Future Improvements

**Display:** Larger benchmarks, shared memory, pinned memory, streams, and more filters.

**Say:** Future work would benchmark realistic 1920x1080 images with repeated trials, use shared-memory tiles for blur and edges, test pinned memory and streams, add more filters, and optionally support PNG or JPEG through an external library.

## Slide 15 - Conclusion and GitHub Repository

**Display:** Public repository URL and evidence files.

**Say:** This project demonstrates a complete CUDA host-to-device-to-host pipeline with four kernels, explicit memory management, synchronization, error checking, and timing. The source, outputs, CSV, and execution evidence are available at [github.com/TechBot-X/cuda-gpu-image-processing-capstone](https://github.com/TechBot-X/cuda-gpu-image-processing-capstone).

## Live Demo Checklist

```bash
make clean
make info
make
make run
cat results/performance.csv
find output -maxdepth 1 -type f -print
```

Capture the GPU model and toolkit version from `make info`, retain the terminal output, and show the actual files in `results/` during the presentation. Do not claim a GPU speedup from the 8x8 sample.
