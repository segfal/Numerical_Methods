
# check if machine is mac or windows, if true then return error
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Error: This script is only supported on Linux"
    exit 1
fi

# if nvcc is not installed, then install it
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc is not installed"
    sudo apt-get update
    sudo apt-get install nvidia-cuda-toolkit
    sudo apt-get install nvidia-cuda-toolkit-dev
    sudo apt-get install nvidia-cuda-toolkit-doc
    sudo apt-get install nvidia-cuda-toolkit-samples
    sudo apt-get install nvidia-cuda-toolkit-tools
    sudo apt-get install nvidia-cuda-toolkit-utils
    sudo apt-get install nvidia-cuda-toolkit-libs
    sudo apt-get install nvidia-cuda-toolkit-libs-dev
    sudo apt-get install nvidia-cuda-toolkit-libs-doc
    sudo apt-get install nvidia-cuda-toolkit-libs-dev
    sudo apt-get install nvidia-cuda-toolkit-libs-doc
    sudo apt-get install nvidia-cuda-toolkit-libs-dev
    sudo apt-get install nvidia-cuda-toolkit-libs-doc
    sudo apt-get install nvidia-cuda-toolkit-libs-dev
fi

# check if nvcc is installed
git pull

echo "Building..."
nvcc main.cu -o root_finder

if [ $? -eq 0 ]; then
    echo "Build successful"
else
    echo "Build failed"
    exit 1
fi

echo "Build successful"
echo "Run with ./root_finder"

./root_finder
