# Check if Docker is running
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running. Please start Docker Desktop first."
    exit 1
}

# Pull the latest changes
Write-Host "Pulling latest changes from git..."
git pull

# Build the Docker image
Write-Host "Building Docker image..."
docker build -t cuda-root-finder .

# Run the container in interactive mode
Write-Host "Starting container in interactive mode..."
docker run -it --gpus all cuda-root-finder /bin/bash 