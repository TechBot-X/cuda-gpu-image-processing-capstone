# Respect an explicit CUDA=... override, otherwise find nvcc in PATH or the
# standard /usr/local/cuda installation locations used by Linux CUDA hosts.
CUDA ?= $(shell command -v nvcc 2>/dev/null || find /usr/local/cuda*/bin -type f -name nvcc -print -quit 2>/dev/null)
TARGET := cuda_image_processing
SOURCES := src/main.cu src/image_processing.cu
CXXFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall,-Wextra
# The verified development GPU is an RTX 4060 (compute capability 8.9).
# Override this, for example ARCH=-arch=sm_75, for another GPU.
ARCH ?= -arch=sm_89

.PHONY: all clean run info

all: $(TARGET)

$(TARGET): $(SOURCES) src/image_processing.h
	$(CUDA) $(CXXFLAGS) $(ARCH) $(SOURCES) -o $@

clean:
	rm -f $(TARGET)
	rm -f output/*.ppm results/performance.csv

info:
	nvidia-smi
	$(CUDA) --version

run: $(TARGET)
	./$(TARGET) --input input/sample.ppm --output output --operation all
