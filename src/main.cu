#include "image_processing.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string input;
    std::string output;
    std::string operation;
    std::string results = "results/performance.csv";
    int width = 0;
    int height = 0;
    float brightness = 20.0f;
    float contrast = 1.20f;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " --input FILE --output FILE --operation OP [options]\n"
              << "Operations: grayscale, enhance, blur, edge, all\n"
              << "Options: --width N --height N --brightness N --contrast N --results FILE\n"
              << "Input accepts ASCII PPM (P3) or binary PPM (P6); output is binary PPM.\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto value = [&](const std::string& name) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return std::string(argv[++i]);
        };
        if (flag == "--input") args.input = value(flag);
        else if (flag == "--output") args.output = value(flag);
        else if (flag == "--operation") args.operation = value(flag);
        else if (flag == "--width") args.width = std::stoi(value(flag));
        else if (flag == "--height") args.height = std::stoi(value(flag));
        else if (flag == "--brightness") args.brightness = std::stof(value(flag));
        else if (flag == "--contrast") args.contrast = std::stof(value(flag));
        else if (flag == "--results") args.results = value(flag);
        else if (flag == "--help" || flag == "-h") { print_usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("Unknown argument: " + flag);
    }
    if (args.input.empty() || args.output.empty() || args.operation.empty()) throw std::runtime_error("--input, --output, and --operation are required");
    if (args.width < 0 || args.height < 0 || (args.width == 0) != (args.height == 0)) throw std::runtime_error("--width and --height must be provided together and positive");
    if (args.operation != "grayscale" && args.operation != "enhance" && args.operation != "blur" && args.operation != "edge" && args.operation != "all") throw std::runtime_error("Invalid operation: " + args.operation);
    return args;
}

void append_csv(const std::string& path, const std::string& operation, const Image& image, double cpu_ms, const ProcessingTimings& timings) {
    std::filesystem::path result_path(path);
    if (result_path.has_parent_path()) std::filesystem::create_directories(result_path.parent_path());
    bool needs_header = !std::filesystem::exists(result_path) || std::filesystem::file_size(result_path) == 0;
    std::ofstream csv(path, std::ios::app);
    if (!csv) throw std::runtime_error("Could not open results CSV: " + path);
    if (needs_header) csv << "operation,image_width,image_height,cpu_time_ms,gpu_kernel_ms,host_to_device_ms,device_to_host_ms\n";
    csv << std::fixed << std::setprecision(3) << operation << ',' << image.width << ',' << image.height << ',' << cpu_ms << ',' << timings.kernel_ms << ',' << timings.host_to_device_ms << ',' << timings.device_to_host_ms << '\n';
}

void run_operation(const Image& input, const std::string& operation, const std::string& output, const Arguments& args) {
    auto cpu_start = std::chrono::steady_clock::now();
    Image cpu_output = process_on_cpu(input, operation, args.brightness, args.contrast);
    auto cpu_stop = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
    ProcessingTimings timings;
    Image gpu_output = process_on_gpu(input, operation, args.brightness, args.contrast, timings);
    std::filesystem::path output_path(output);
    if (output_path.has_parent_path()) std::filesystem::create_directories(output_path.parent_path());
    save_ppm(gpu_output, output);
    append_csv(args.results, operation, input, cpu_ms, timings);
    std::cout << std::fixed << std::setprecision(3) << operation << ": CPU " << cpu_ms << " ms, GPU kernel " << timings.kernel_ms
              << " ms, H2D " << timings.host_to_device_ms << " ms, D2H " << timings.device_to_host_ms << " ms\n"
              << "Saved " << output << " (" << input.width << "x" << input.height << ")\n";
    (void)cpu_output;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Arguments args = parse_arguments(argc, argv);
        Image image = load_ppm(args.input);
        if (args.width > 0) image = resize_nearest(image, args.width, args.height);
        if (args.operation == "all") {
            const std::vector<std::pair<std::string, std::string>> jobs = {{"grayscale", "grayscale.ppm"}, {"enhance", "enhanced.ppm"}, {"blur", "blurred.ppm"}, {"edge", "edges.ppm"}};
            std::filesystem::path directory(args.output);
            for (const auto& job : jobs) run_operation(image, job.first, (directory / job.second).string(), args);
        } else {
            run_operation(image, args.operation, args.output, args);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
