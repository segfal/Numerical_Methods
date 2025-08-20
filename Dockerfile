FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the source code
COPY . .

# Build the program
RUN nvcc main.cu -o root_finder

# Command to run when container starts
CMD ["./root_finder"] 