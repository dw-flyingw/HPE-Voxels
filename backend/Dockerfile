# FLUX.1-dev Backend Dockerfile
# Optimized for NVIDIA H200 GPU

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy project configuration
COPY pyproject.toml .

# Install PyTorch with CUDA support first (for better caching)
# Using latest stable versions for CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core AI dependencies with compatible versions
RUN pip3 install --no-cache-dir \
    huggingface_hub==0.17.3 \
    transformers==4.35.2 \
    diffusers==0.24.0 \
    accelerate==0.24.1

# Install web server dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6

# Install remaining dependencies
RUN pip3 install --no-cache-dir \
    Pillow==10.1.0 \
    numpy==1.24.4 \
    scipy==1.11.4 \
    python-dotenv==1.0.0

# Copy application code
COPY flux_server.py .

# Expose the server port (default 8000)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${FLUX_SERVER_PORT:-8000}/health || exit 1

# Run the server
CMD ["python3", "flux_server.py"]

