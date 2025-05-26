cmake_minimum_required(VERSION 3.20)

# Function to check for NVIDIA GPU
function(check_nvidia_gpu)
    if(WIN32)
        execute_process(
            COMMAND nvidia-smi
            RESULT_VARIABLE NVIDIA_SMI_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(NVIDIA_SMI_RESULT EQUAL 0)
            set(HAS_NVIDIA_GPU TRUE PARENT_SCOPE)
        else()
            set(HAS_NVIDIA_GPU FALSE PARENT_SCOPE)
        endif()
    elseif(UNIX AND NOT APPLE)
        execute_process(
            COMMAND lspci | grep -i nvidia
            RESULT_VARIABLE NVIDIA_LSPCI_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(NVIDIA_LSPCI_RESULT EQUAL 0)
            set(HAS_NVIDIA_GPU TRUE PARENT_SCOPE)
        else()
            set(HAS_NVIDIA_GPU FALSE PARENT_SCOPE)
        endif()
    else()
        set(HAS_NVIDIA_GPU FALSE PARENT_SCOPE)
    endif()
endfunction()

# Check for NVIDIA GPU
check_nvidia_gpu()

# Set up project based on NVIDIA GPU availability
if(HAS_NVIDIA_GPU)
    project(my_project LANGUAGES CXX CUDA)
    message(STATUS "NVIDIA GPU detected, enabling CUDA support")
else()
    project(my_project LANGUAGES CXX)
    message(STATUS "No NVIDIA GPU detected, using CPU-only build")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package if NVIDIA GPU is available
if(HAS_NVIDIA_GPU)
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DUSE_CUDA)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/includes)

# Source files
set(CPU_SOURCES
    src/cpu/RootFinding.cpp
    src/cpu/main.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/RootFinding.cu
    src/cuda/main.cu
)

if(HAS_NVIDIA_GPU)
    # CUDA build
    add_executable(${PROJECT_NAME} ${CPU_SOURCES} ${CUDA_SOURCES})
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
else()
    # CPU-only build
    add_executable(${PROJECT_NAME} ${CPU_SOURCES})
endif()

# Set compiler flags
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
endif()

# Add clean target
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/build
    COMMENT "Cleaning all build files"
) 