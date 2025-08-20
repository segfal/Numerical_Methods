#!/bin/bash

set -e

if [[ "$1" == "opencl" ]]; then
    echo "Building OpenCL example..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        g++ -std=c++17 opencl_matmul.cpp -framework OpenCL -o opencl_matmul
    else
        g++ -std=c++17 opencl_matmul.cpp -lOpenCL -o opencl_matmul
    fi
    echo "Build successful"
    echo "Run with ./opencl_matmul"
    ./opencl_matmul
    exit 0
fi

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use C++ version
    echo "Building for macOS..."
    g++ -std=c++17 main.cpp src/RootFinding.cpp -I includes -o root_finder
    if [ $? -eq 0 ]; then
        echo "Build successful"
        echo "Run with ./root_finder"
        ./root_finder
    else
        echo "Build failed"
        exit 1
    fi
else
    # Linux - use CUDA version
    echo "Building for Linux with CUDA..."
    
    # Check if nvcc is installed
    if ! command -v nvcc &> /dev/null; then
        echo "Error: nvcc is not installed"
        sudo apt-get update
        sudo apt-get install -y nvidia-cuda-toolkit
    fi
    
    nvcc main.cu src/RootFinding.cu -I includes -o root_finder
    if [ $? -eq 0 ]; then
        echo "Build successful"
        echo "Run with ./root_finder"
        ./root_finder
    else
        echo "Build failed"
        exit 1
    fi
fi


