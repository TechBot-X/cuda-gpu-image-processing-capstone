# CUDA GPU-Accelerated Image Processing and Enhancement

A beginner-friendly CUDA C++ capstone project for **CUDA at Scale for the Enterprise**. The application loads an RGB image, sends its pixels from host memory to NVIDIA GPU global memory, launches 2D CUDA kernels, copies the result back, saves it, and records measured CPU/GPU timings.

Public repository: [https://github.com/TechBot-X/cuda-gpu-image-processing-capstone](https://github.com/TechBot-X/cuda-gpu-image-processing-capstone)

The implementation uses PPM (`P3` or binary `P6`) images. PPM keeps the build self-contained on a headless Linux/Coursera machine: no OpenCV or JPEG development package is required. PPM output can be converted to PNG/JPEG afterward with ImageMagick or another image viewer.

## Project Overview

Image processing is a strong GPU workload because pixels are mostly independent. A 1920x1080 image contains more than two million pixels, and the same arithmetic can be applied to every pixel concurrently. CUDA maps one thread to one output pixel and uses a 2D grid so the image's spatial structure is easy to understand.

The problem addressed is repetitive per-pixel image enhancement: applying consistent color, intensity, neighborhood, and edge calculations across an entire image. The project objective is to implement those operations visibly in CUDA C++, compare them with CPU reference functions, and record real transfer and kernel timings.

## Motivation

This project makes GPU acceleration visible rather than hiding it behind a library. The source contains explicit `cudaMalloc`, `cudaMemcpy`, kernel launches, CUDA events, synchronization, error checking, and `cudaFree` calls. It also computes a CPU baseline for every operation so a reviewer can compare methods on the same input.

## Features

- Grayscale conversion using RGB luminance weights.
- Brightness and contrast enhancement with configurable values.
- 3x3 weighted Gaussian-style blur.
- Sobel edge detection.
- CPU implementations of all four operations for comparison.
- CUDA event timing for host-to-device transfer, kernel execution, and device-to-host transfer.
- Command-line operation selection, output paths, optional resize, and CSV results.
- `all` mode that generates all four output images.
- Portable PPM input and output with no external image library dependency.

## Technologies Used

- CUDA C++ (`.cu`) and NVIDIA CUDA Runtime API.
- `nvcc` compiler and CUDA kernels.
- C++17 standard library for file I/O, timing, argument parsing, and filesystem paths.
- Binary or ASCII PPM image format.

## CUDA/GPU Architecture

1. The CPU loads RGB bytes into a `std::vector<unsigned char>`.
2. `cudaMalloc` allocates two device buffers in GPU global memory.
3. `cudaMemcpy(..., cudaMemcpyHostToDevice)` transfers the input pixels to the GPU.
4. A 2D kernel launch uses `dim3 block(16, 16)` and a grid large enough to cover the image.
5. Each thread computes its `(x, y)` pixel from `blockIdx`, `blockDim`, and `threadIdx`.
6. `cudaDeviceSynchronize` and CUDA events ensure the kernel has finished before timing and copying.
7. `cudaMemcpy(..., cudaMemcpyDeviceToHost)` returns the processed pixels.
8. Device allocations and timing events are released with `cudaFree` and `cudaEventDestroy`.

The key indexing expression is:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

Threads outside the image dimensions return immediately. This boundary check is required because the grid is rounded up to complete 16x16 blocks.

## How GPU Parallelism Is Used

The grayscale and enhancement kernels perform independent per-pixel arithmetic. Blur and edge detection read a small neighborhood around each pixel, but each output pixel is still independent of other output pixels. The four kernels use global memory for the input and output arrays. The neighborhood operations are intentionally written plainly for teaching; a future version could use shared-memory tiles to reduce repeated global-memory reads.

The kernel launch has this shape:

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
blur_kernel<<<grid, block>>>(device_input, device_output, width, height);
```

## Project Structure

```text
cuda-image-processing/
├── src/
│   ├── main.cu
│   ├── image_processing.cu
│   └── image_processing.h
├── input/
│   ├── sample.ppm
│   └── README.md
├── output/                 Generated PPM images
├── results/                Generated performance.csv
├── screenshots/            Add presentation screenshots here
├── Makefile
├── README.md
├── PROJECT_SUBMISSION.md
├── PRESENTATION.md        15-slide demo script
├── requirements.txt
└── .gitignore
```

## Requirements

- Linux with an NVIDIA GPU and a working driver.
- CUDA Toolkit with `nvcc` available in `PATH`.
- `make` and a C++17-capable host compiler.
- No Python packages or image libraries are required.

Check the environment with:

```bash
nvidia-smi
nvcc --version
make --version
```

The Makefile also provides the reproducible environment check:

```bash
make info
```

## Installation and Setup

Clone or copy this repository into the CUDA machine. Confirm that `input/sample.ppm` is present. For a larger real image, convert it to PPM:

```bash
convert photograph.jpg -depth 8 input/photograph.ppm
```

The program accepts both ASCII `P3` and binary `P6` PPM. PPM is used here to keep compilation deterministic and easy to reproduce in a course environment.

## Compilation

From the repository root:

```bash
make clean
make
make run
```

`make run` is equivalent to running the `all` example below and writes all four
outputs to `output/`.

The default Makefile target is `sm_89`, matching the verified NVIDIA GeForce RTX 4060 environment. If the target GPU needs another architecture, override it, for example:

```bash
make clean
make ARCH=-arch=sm_75
```

## Usage

Single operation:

```bash
./cuda_image_processing --input input/sample.ppm --output output/grayscale.ppm --operation grayscale
```

Generate all operations into a directory:

```bash
./cuda_image_processing --input input/sample.ppm --output output --operation all
```

Use a custom brightness and contrast:

```bash
./cuda_image_processing --input input/sample.ppm --output output/enhanced.ppm \
  --operation enhance --brightness 30 --contrast 1.35
```

Resize during loading, useful for controlled benchmarks:

```bash
./cuda_image_processing --input input/large.ppm --output output/blurred.ppm \
  --operation blur --width 1920 --height 1080
```

Choose another CSV location:

```bash
./cuda_image_processing --input input/sample.ppm --output output/edges.ppm \
  --operation edge --results results/performance.csv
```

## Command-Line Arguments

- `--input FILE`: Required PPM input path.
- `--output FILE_OR_DIRECTORY`: Required output path. In `all` mode it is a directory.
- `--operation OP`: Required: `grayscale`, `enhance`, `blur`, `edge`, or `all`.
- `--width N --height N`: Optional nearest-neighbor resize; provide both together.
- `--brightness N`: Enhancement offset, default `20`.
- `--contrast N`: Enhancement multiplier, default `1.20`.
- `--results FILE`: CSV path, default `results/performance.csv`.
- `--help`: Print usage information.

## CPU vs GPU Methodology

For each selected operation, the application runs the matching CPU implementation and measures it with `std::chrono`. It then runs the CUDA implementation on the same image and parameters. The GPU timing is split into:

- `host_to_device_ms`: host-to-device copy measured with CUDA events.
- `gpu_kernel_ms`: actual kernel execution measured with CUDA events.
- `device_to_host_ms`: device-to-host copy measured with CUDA events.

The console reports all values, and each run appends a row to `results/performance.csv`:

```text
operation,image_width,image_height,cpu_time_ms,gpu_kernel_ms,host_to_device_ms,device_to_host_ms
```

Do not invent or manually edit performance numbers. Run the executable on the target GPU to generate actual measurements. For a meaningful comparison, benchmark a large image and repeat each operation several times after one warm-up run. The simple program reports one measured invocation per command.

## Performance Results

The verified WSL run used an NVIDIA GeForce RTX 4060 Laptop GPU with CUDA toolkit 13.3.73 and `-arch=sm_89`. The 8x8 sample produced the following actual measurements:

```text
operation  cpu_ms  gpu_kernel_ms  host_to_device_ms  device_to_host_ms
grayscale  0.001   1.360          0.411              0.118
enhance    0.002   0.341          0.176              0.067
blur       0.003   0.109          0.069              0.104
edge       0.002   0.354          0.095              0.169
```

The sample is intentionally tiny, so these results prove GPU execution but do not claim a speedup. Kernel launch and transfer overhead dominate at 8x8. For a meaningful comparison, run the same program with a large image and compare the CPU and GPU columns. The raw CSV is available at `results/performance.csv`, and the evidence summary is at `results/performance.txt`.

Inspect the generated CSV with:

```bash
cat results/performance.csv
```

For a report, include the GPU model from `nvidia-smi`, image dimensions, CUDA version, block size, and the CSV rows produced on that machine. Compare CPU time with `gpu_kernel_ms`, and separately discuss transfer overhead because small images can spend more time moving data than computing.

## Screenshots and Output Explanation

On a headless machine, proof of execution is the terminal output, generated PPM files, and `results/performance.csv`. To create presentation screenshots:

```bash
./cuda_image_processing --input input/sample.ppm --output output --operation all
ls -lh output results/performance.csv
cat results/performance.csv
```

Copy terminal output, an `nvidia-smi` result, and visualizations of the four PPM outputs into `screenshots/`. Convert outputs for a desktop viewer if desired:

```bash
convert output/grayscale.ppm screenshots/grayscale.png
convert output/enhanced.ppm screenshots/enhanced.png
convert output/blurred.ppm screenshots/blurred.png
convert output/edges.ppm screenshots/edges.png
```

Expected visual behavior: grayscale removes color, enhancement changes luminance and contrast, blur softens local detail, and edges produces bright contours on a dark background.

## Challenges Encountered

- Keeping the project portable without assuming OpenCV or JPEG headers.
- Handling image boundaries for 3x3 neighborhood kernels.
- Measuring copies and kernel execution separately rather than presenting an ambiguous total.
- Ensuring rounded-up CUDA grids never access pixels outside the image.

## Lessons Learned

- A CUDA kernel launch is asynchronous, so explicit error checks and synchronization matter.
- GPU acceleration includes transfer costs, not just arithmetic throughput.
- Coalesced row-major access is a useful first memory-access pattern.
- Shared memory and tiled neighborhoods are natural next optimizations for blur and edge detection.
- A CPU baseline provides context for interpreting GPU measurements.

## Future Improvements

- Add shared-memory tiling for blur and Sobel kernels.
- Add batched images and pinned host memory for transfer experiments.
- Add JPEG/PNG support through an optional library such as stb_image or OpenCV.
- Run repeated trials and report mean, minimum, and standard deviation.
- Add CUDA streams to overlap transfers with computation.
- Add automated image-quality checks against CPU output.

## Conclusion

This capstone demonstrates a complete CUDA image-processing pipeline: host loading, explicit host-to-device transfer, parallel CUDA kernel execution, synchronization, device-to-host transfer, output generation, and measured CPU/GPU comparison. Image pixels provide substantial data parallelism, making the workload a clear and approachable example of why GPUs can accelerate repetitive numerical operations.

## Proof-of-Execution Procedure

Run the following on the CUDA machine and retain the terminal output and generated artifacts:

```bash
make clean
make
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
./cuda_image_processing --input input/sample.ppm --output output --operation all
cat results/performance.csv
find output -maxdepth 1 -type f -print
```

This demonstrates compilation, GPU availability, four CUDA runs, measured timings, and four generated images. For the final repository, commit the source and README; generated outputs and CSV may be retained as evidence if the course submission requires artifacts.

For peer-submission evidence, the repository includes `results/execution_log.txt`, `results/gpu_info.txt`, `results/output.txt`, `results/performance.txt`, and the generated `results/performance.csv`. These files were produced by the actual WSL CUDA execution described above.

## 5-10 Minute Presentation/Demo Outline

1. **0:00-1:00 - Problem:** Explain why repeated per-pixel image work is a good GPU workload.
2. **1:00-2:00 - Architecture:** Show the host/device data flow and the PPM portability choice.
3. **2:00-4:00 - Code:** Walk through 2D indexing, `<<<grid, block>>>`, global memory, error checking, and synchronization.
4. **4:00-6:00 - Live run:** Build, run `--operation all`, and show the four outputs and console timings.
5. **6:00-8:00 - Results:** Open `performance.csv`, explain CPU versus kernel and transfer timings, and identify the GPU model.
6. **8:00-10:00 - Reflection:** Discuss boundary handling, transfer overhead, lessons learned, and shared-memory tiling as future work.

The complete slide-by-slide script is in [PRESENTATION.md](PRESENTATION.md), and the concise submission checklist is in [PROJECT_SUBMISSION.md](PROJECT_SUBMISSION.md).
