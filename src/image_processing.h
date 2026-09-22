#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <cstddef>
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;

    std::size_t size() const { return pixels.size(); }
};

Image load_ppm(const std::string& path);
void save_ppm(const Image& image, const std::string& path);
Image resize_nearest(const Image& image, int width, int height);

struct ProcessingTimings {
    float host_to_device_ms = 0.0f;
    float kernel_ms = 0.0f;
    float device_to_host_ms = 0.0f;
};

Image process_on_gpu(const Image& input, const std::string& operation,
                     float brightness, float contrast,
                     ProcessingTimings& timings);
Image process_on_cpu(const Image& input, const std::string& operation,
                     float brightness, float contrast);

#endif
