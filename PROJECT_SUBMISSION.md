# Coursera Capstone Submission

## Project

**CUDA GPU-Accelerated Image Processing and Enhancement** is a CUDA C++ application that processes RGB PPM images on an NVIDIA GPU. It implements grayscale conversion, brightness/contrast enhancement, weighted blur, and Sobel edge detection. Each CUDA operation maps image pixels to a 2D grid of GPU threads.

## Repository Contents

- `src/main.cu`: command-line interface, CPU timing, output handling, and CSV logging.
- `src/image_processing.cu`: PPM I/O, CPU reference operations, CUDA kernels, transfers, timing, and cleanup.
- `src/image_processing.h`: shared image and timing declarations.
- `Makefile`: `clean`, `all`, `info`, and reproducible `run` targets.
- `input/sample.ppm`: small portable test image.
- `output/`: generated processed images.
- `results/`: generated performance CSV and proof files from the CUDA machine.
- `PRESENTATION.md`: ten-slide presentation and speaker notes.
- `README.md`: full setup, architecture, usage, methodology, and reproduction guide.

## Build and Run

Run these commands from the repository root on Linux with CUDA installed:

```bash
make clean
make info
make
make run
```

The `make run` target executes all four operations. Equivalent direct command:

```bash
./cuda_image_processing --input input/sample.ppm --output output --operation all
```

The program writes `grayscale.ppm`, `enhanced.ppm`, `blurred.ppm`, and `edges.ppm` under `output/`, and appends measured timings to `results/performance.csv`.

## GPU and Results

The project was compiled and executed through WSL Ubuntu 24.04 with an NVIDIA GeForce RTX 4060 Laptop GPU, driver 616.92, CUDA toolkit 13.3.73, and `-arch=sm_89`. The final run completed without CUDA runtime errors. The 8x8 sample produced these measured kernel times: grayscale 1.423 ms, enhance 0.083 ms, blur 0.188 ms, and edge 0.176 ms. Because the sample is tiny, these numbers demonstrate execution rather than a speedup; launch and transfer overhead dominate.

Preserved evidence files are:

- `results/gpu_info.txt`
- `results/execution_log.txt`
- `results/output.txt`
- `results/performance.csv`
- `results/performance.txt`

These are actual outputs from the WSL CUDA run, not invented values. For a stronger throughput comparison, rerun with a large 1920x1080 PPM image.

## CUDA Concepts Demonstrated

- `cudaMalloc` and `cudaFree` for device allocation.
- `cudaMemcpy` for host-to-device and device-to-host transfers.
- `__global__` kernels and `<<<grid, block>>>` launches.
- `blockIdx`, `blockDim`, and `threadIdx` for 2D pixel indexing.
- 16x16 blocks and rounded-up 2D grids.
- Global-memory RGB input/output buffers.
- `cudaGetLastError` and a `CUDA_CHECK` wrapper for API and launch errors.
- `cudaDeviceSynchronize` for explicit completion.
- CUDA events for transfer and kernel timing.

## Presentation Demonstration

Show the source indexing expression, run `make info`, build with `make`, execute `make run`, open the generated CSV, and explain why image pixels expose enough independent work for GPU parallelism. The complete 5-10 minute script is in `PRESENTATION.md`.

## Submission Status

The source, build workflow, documentation, generated images, timing CSV, and proof-of-execution files are ready for peer review. A reviewer can reproduce the run on a CUDA-enabled Linux system with `make clean`, `make info`, `make`, and `make run`.
