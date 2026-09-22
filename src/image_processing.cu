#include "image_processing.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + ": " + cudaGetErrorString(error__)); \
    } \
} while (false)

__device__ unsigned char clamp_byte(float value) {
    return static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, value)));
}

// Each CUDA thread owns one pixel. blockIdx identifies the block, blockDim its
// dimensions, and threadIdx identifies the thread within that block.
__global__ void grayscale_kernel(const unsigned char* input, unsigned char* output,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int pixel = (y * width + x) * 3;
    float gray = 0.299f * input[pixel] + 0.587f * input[pixel + 1] + 0.114f * input[pixel + 2];
    unsigned char value = clamp_byte(gray);
    output[pixel] = output[pixel + 1] = output[pixel + 2] = value;
}

__global__ void enhance_kernel(const unsigned char* input, unsigned char* output,
                               int width, int height, float brightness, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int pixel = (y * width + x) * 3;
    for (int channel = 0; channel < 3; ++channel) {
        float adjusted = (static_cast<float>(input[pixel + channel]) - 128.0f) * contrast + 128.0f + brightness;
        output[pixel + channel] = clamp_byte(adjusted);
    }
}

__global__ void blur_kernel(const unsigned char* input, unsigned char* output,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    int pixel = (y * width + x) * 3;
    for (int channel = 0; channel < 3; ++channel) {
        int total = 0;
        int weight_total = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int sample_x = min(width - 1, max(0, x + dx));
                int sample_y = min(height - 1, max(0, y + dy));
                total += input[(sample_y * width + sample_x) * 3 + channel] * weights[dy + 1][dx + 1];
                weight_total += weights[dy + 1][dx + 1];
            }
        }
        output[pixel + channel] = static_cast<unsigned char>(total / weight_total);
    }
}

__global__ void edge_kernel(const unsigned char* input, unsigned char* output,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int horizontal = 0;
    int vertical = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int sample_x = min(width - 1, max(0, x + dx));
            int sample_y = min(height - 1, max(0, y + dy));
            int source = (sample_y * width + sample_x) * 3;
            int gray = (299 * input[source] + 587 * input[source + 1] + 114 * input[source + 2]) / 1000;
            horizontal += gray * gx[dy + 1][dx + 1];
            vertical += gray * gy[dy + 1][dx + 1];
        }
    }
    unsigned char magnitude = clamp_byte(sqrtf(static_cast<float>(horizontal * horizontal + vertical * vertical)));
    int pixel = (y * width + x) * 3;
    output[pixel] = output[pixel + 1] = output[pixel + 2] = magnitude;
}

void validate_image(const Image& image) {
    if (image.width <= 0 || image.height <= 0 || image.pixels.size() != static_cast<std::size_t>(image.width) * image.height * 3) {
        throw std::runtime_error("Invalid RGB image dimensions or pixel data");
    }
}

void read_token(std::istream& stream, std::string& token) {
    while (stream >> token) {
        if (!token.empty() && token[0] == '#') {
            stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        return;
    }
    throw std::runtime_error("Unexpected end of PPM file");
}

Image cpu_transform(const Image& input, const std::string& operation, float brightness, float contrast) {
    Image output = input;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            int pixel = (y * input.width + x) * 3;
            if (operation == "grayscale") {
                unsigned char value = static_cast<unsigned char>(std::clamp(0.299f * input.pixels[pixel] + 0.587f * input.pixels[pixel + 1] + 0.114f * input.pixels[pixel + 2], 0.0f, 255.0f));
                output.pixels[pixel] = output.pixels[pixel + 1] = output.pixels[pixel + 2] = value;
            } else if (operation == "enhance") {
                for (int channel = 0; channel < 3; ++channel) {
                    output.pixels[pixel + channel] = static_cast<unsigned char>(std::clamp((input.pixels[pixel + channel] - 128.0f) * contrast + 128.0f + brightness, 0.0f, 255.0f));
                }
            }
        }
    }
    if (operation == "blur") {
        const int weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
        for (int y = 0; y < input.height; ++y) for (int x = 0; x < input.width; ++x) {
            int pixel = (y * input.width + x) * 3;
            for (int channel = 0; channel < 3; ++channel) {
                int total = 0;
                for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                    int sx = std::clamp(x + dx, 0, input.width - 1), sy = std::clamp(y + dy, 0, input.height - 1);
                    total += input.pixels[(sy * input.width + sx) * 3 + channel] * weights[dy + 1][dx + 1];
                }
                output.pixels[pixel + channel] = static_cast<unsigned char>(total / 16);
            }
        }
    } else if (operation == "edge") {
        const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}, gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        for (int y = 0; y < input.height; ++y) for (int x = 0; x < input.width; ++x) {
            int horizontal = 0, vertical = 0;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                int sx = std::clamp(x + dx, 0, input.width - 1), sy = std::clamp(y + dy, 0, input.height - 1), source = (sy * input.width + sx) * 3;
                int gray = (299 * input.pixels[source] + 587 * input.pixels[source + 1] + 114 * input.pixels[source + 2]) / 1000;
                horizontal += gray * gx[dy + 1][dx + 1]; vertical += gray * gy[dy + 1][dx + 1];
            }
            unsigned char value = static_cast<unsigned char>(std::clamp(std::sqrt(static_cast<float>(horizontal * horizontal + vertical * vertical)), 0.0f, 255.0f));
            int pixel = (y * input.width + x) * 3; output.pixels[pixel] = output.pixels[pixel + 1] = output.pixels[pixel + 2] = value;
        }
    }
    return output;
}

} // namespace

Image load_ppm(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Could not open input image: " + path);
    std::string magic, token;
    read_token(stream, magic);
    if (magic != "P6" && magic != "P3") throw std::runtime_error("Only PPM (P3 or P6) input is supported");
    read_token(stream, token); int width = std::stoi(token);
    read_token(stream, token); int height = std::stoi(token);
    read_token(stream, token); int max_value = std::stoi(token);
    if (width <= 0 || height <= 0 || max_value != 255) throw std::runtime_error("PPM must have positive dimensions and max value 255");
    Image image{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height * 3)};
    if (magic == "P6") {
        stream.get();
        stream.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
        if (stream.gcount() != static_cast<std::streamsize>(image.pixels.size())) throw std::runtime_error("PPM pixel data is truncated");
    } else {
        for (unsigned char& pixel : image.pixels) {
            read_token(stream, token);
            int value = std::stoi(token);
            if (value < 0 || value > 255) throw std::runtime_error("P3 pixel value is outside 0-255");
            pixel = static_cast<unsigned char>(value);
        }
    }
    return image;
}

void save_ppm(const Image& image, const std::string& path) {
    validate_image(image);
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Could not create output image: " + path);
    stream << "P6\n" << image.width << " " << image.height << "\n255\n";
    stream.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (!stream) throw std::runtime_error("Failed while writing output image: " + path);
}

Image resize_nearest(const Image& image, int width, int height) {
    validate_image(image);
    if (width <= 0 || height <= 0) throw std::runtime_error("Resize dimensions must be positive");
    Image result{width, height, std::vector<unsigned char>(static_cast<std::size_t>(width) * height * 3)};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        int source_x = x * image.width / width, source_y = y * image.height / height;
        for (int channel = 0; channel < 3; ++channel) result.pixels[(y * width + x) * 3 + channel] = image.pixels[(source_y * image.width + source_x) * 3 + channel];
    }
    return result;
}

Image process_on_cpu(const Image& input, const std::string& operation, float brightness, float contrast) {
    validate_image(input);
    return cpu_transform(input, operation, brightness, contrast);
}

Image process_on_gpu(const Image& input, const std::string& operation, float brightness, float contrast, ProcessingTimings& timings) {
    validate_image(input);
    if (operation != "grayscale" && operation != "enhance" && operation != "blur" && operation != "edge") throw std::runtime_error("Unknown operation: " + operation);
    std::size_t bytes = input.pixels.size();
    unsigned char *device_input = nullptr, *device_output = nullptr;
    cudaEvent_t transfer_start = nullptr, kernel_start = nullptr, kernel_stop = nullptr, transfer_stop = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(&device_input, bytes));
        CUDA_CHECK(cudaMalloc(&device_output, bytes));
        CUDA_CHECK(cudaEventCreate(&transfer_start)); CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop)); CUDA_CHECK(cudaEventCreate(&transfer_stop));
        CUDA_CHECK(cudaEventRecord(transfer_start));
        // Host-to-device transfer places image data in global memory on the GPU.
        CUDA_CHECK(cudaMemcpy(device_input, input.pixels.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(kernel_start));
        dim3 block(16, 16);
        dim3 grid((input.width + block.x - 1) / block.x, (input.height + block.y - 1) / block.y);
        if (operation == "grayscale") grayscale_kernel<<<grid, block>>>(device_input, device_output, input.width, input.height);
        else if (operation == "enhance") enhance_kernel<<<grid, block>>>(device_input, device_output, input.width, input.height, brightness, contrast);
        else if (operation == "blur") blur_kernel<<<grid, block>>>(device_input, device_output, input.width, input.height);
        else edge_kernel<<<grid, block>>>(device_input, device_output, input.width, input.height);
        CUDA_CHECK(cudaGetLastError());
        // Synchronization makes kernel completion visible before timing/copying.
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(kernel_stop)); CUDA_CHECK(cudaEventSynchronize(kernel_stop));
        Image output{input.width, input.height, std::vector<unsigned char>(bytes)};
        // Device-to-host transfer returns processed pixels to CPU memory.
        CUDA_CHECK(cudaMemcpy(output.pixels.data(), device_output, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(transfer_stop));
        CUDA_CHECK(cudaEventSynchronize(transfer_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timings.host_to_device_ms, transfer_start, kernel_start));
        CUDA_CHECK(cudaEventElapsedTime(&timings.kernel_ms, kernel_start, kernel_stop));
        CUDA_CHECK(cudaEventElapsedTime(&timings.device_to_host_ms, kernel_stop, transfer_stop));
        CUDA_CHECK(cudaEventDestroy(transfer_start)); CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop)); CUDA_CHECK(cudaEventDestroy(transfer_stop));
        CUDA_CHECK(cudaFree(device_input)); CUDA_CHECK(cudaFree(device_output));
        return output;
    } catch (...) {
        if (transfer_start) cudaEventDestroy(transfer_start); if (kernel_start) cudaEventDestroy(kernel_start);
        if (kernel_stop) cudaEventDestroy(kernel_stop); if (transfer_stop) cudaEventDestroy(transfer_stop);
        if (device_input) cudaFree(device_input); if (device_output) cudaFree(device_output);
        throw;
    }
}
