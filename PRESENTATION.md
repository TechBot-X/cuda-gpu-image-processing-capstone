# Presentation and Demo Script

## Slide 1 - Project Title and Objective

**On slide:** CUDA GPU-Accelerated Image Processing and Enhancement.

**Speaker notes:** This project applies four common image-processing operations on an NVIDIA GPU. The objective is to demonstrate explicit CUDA memory management, kernel execution, synchronization, and measured CPU-versus-GPU timing in a complete command-line application.

## Slide 2 - Problem and Use Case

**On slide:** One image contains many pixels; the same operation is repeated across them.

**Speaker notes:** Grayscale, enhancement, blur, and edge detection repeat similar arithmetic for every pixel. That makes image processing a natural data-parallel workload. A CPU can process pixels sequentially, while a GPU can schedule many pixel calculations concurrently.

## Slide 3 - Solution Architecture

**On slide:** Input file -> host vector -> device global memory -> CUDA kernel -> host vector -> output file.

**Speaker notes:** The CPU loads a PPM image into RGB bytes. The program allocates device buffers, copies the pixels to the GPU, launches one selected kernel, synchronizes, copies the result back, and writes a PPM output. The same command also records timings in a CSV file.

## Slide 4 - CUDA/GPU Implementation

**On slide:** `cudaMalloc`, `cudaMemcpy`, `cudaGetLastError`, `cudaDeviceSynchronize`, `cudaFree`.

**Speaker notes:** The implementation intentionally exposes the CUDA runtime calls. A `CUDA_CHECK` wrapper turns API failures into readable exceptions. Launch errors are checked immediately, and synchronization makes kernel completion explicit before the result is copied back.

## Slide 5 - Kernel, Grid, and Block Explanation

**On slide:** `block(16, 16)` and rounded-up 2D grid.

**Speaker notes:** Each thread computes one pixel. The coordinates are `x = blockIdx.x * blockDim.x + threadIdx.x` and `y = blockIdx.y * blockDim.y + threadIdx.y`. The grid is rounded up so every image pixel is covered; boundary checks prevent extra threads from accessing invalid coordinates.

## Slide 6 - Execution Demonstration

**On slide:** Terminal commands and generated output names.

**Speaker notes:** On the CUDA Linux machine I run `make clean`, `make info`, `make`, and `make run`. The `all` operation creates grayscale, enhanced, blurred, and edge images. I show the terminal output, the GPU information, and the four files in the output directory.

## Slide 7 - Results and Performance

**On slide:** `results/performance.csv` columns.

**Speaker notes:** The CSV records CPU time, GPU kernel time, host-to-device time, and device-to-host time for each operation. I compare the actual measurements from the target GPU. I do not claim a speedup unless the collected data shows one, and I explain that small images may be dominated by transfer overhead.

## Slide 8 - Challenges and Lessons Learned

**On slide:** Boundaries, transfers, synchronization, portability.

**Speaker notes:** The neighborhood kernels must clamp coordinates at image edges. CUDA launches are asynchronous, so synchronization and error checks are important. PPM was selected to keep the project reproducible without requiring OpenCV or JPEG packages. The CPU reference provides a useful correctness and timing baseline.

## Slide 9 - Future Improvements

**On slide:** Shared-memory tiles, pinned memory, streams, repeated trials.

**Speaker notes:** Blur and Sobel could use shared-memory tiles to reduce repeated global-memory reads. Future experiments could use pinned host memory, CUDA streams, repeated trials with statistics, larger images, and optional PNG/JPEG support.

## Slide 10 - Conclusion

**On slide:** Complete host-to-device-to-host CUDA pipeline.

**Speaker notes:** This project demonstrates the central course concepts in a visible workflow: parallel kernels, 2D indexing, global memory, transfers, synchronization, error handling, and measurement. The repository is ready for peer review after the target CUDA machine generates its evidence files.

## Live Demo Checklist

```bash
make clean
make info
make
make run
cat results/performance.csv
find output -maxdepth 1 -type f -print
```

Capture the GPU model and toolkit version from `make info`, retain the terminal output, and place the actual files in `results/` before submission.
