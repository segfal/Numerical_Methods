
# check if machine is mac or windows, if true then return error
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Error: This script is only supported on Linux"
    exit 1
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
