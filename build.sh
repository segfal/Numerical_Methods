#!/bin/bash

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


