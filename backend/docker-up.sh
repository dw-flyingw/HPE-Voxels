#!/bin/bash

###############################################################################
# FLUX.1-dev Docker Compose Quick Start Script
###############################################################################

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "FLUX.1-dev Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed!"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not available!"
    echo "Please install Docker Compose plugin"
    exit 1
fi

# Check if nvidia-smi is available (GPU check)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
    echo "FLUX.1-dev requires NVIDIA GPU with CUDA support."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "No .env file found. Creating from template..."
    if [ -f env.template ]; then
        cp env.template .env
        echo "✓ Created .env file from env.template"
    else
        echo "Error: env.template not found!"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "IMPORTANT: Configure your .env file"
    echo "=========================================="
    echo "Please edit the .env file and set your HUGGINGFACE_TOKEN"
    echo ""
    echo "1. Get your token from: https://huggingface.co/settings/tokens"
    echo "2. Accept the FLUX.1-dev license at: https://huggingface.co/black-forest-labs/FLUX.1-dev"
    echo "3. Edit .env and set HUGGINGFACE_TOKEN=your_actual_token_here"
    echo ""
    read -p "Press Enter after you've configured the .env file..."
fi

# Verify Hugging Face token is set
source .env
if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_token_here" ]; then
    echo "Error: HUGGINGFACE_TOKEN not properly set in .env file!"
    echo "Please edit .env and set your Hugging Face token."
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Port: ${FLUX_SERVER_PORT:-8000}"
echo "  Model: ${FLUX_MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
echo ""

# Build and start the service
echo "Building Docker image..."
docker compose build

echo ""
echo "Starting FLUX backend..."
docker compose up -d

echo ""
echo "=========================================="
echo "Waiting for service to be healthy..."
echo "=========================================="
echo "This may take a few minutes as the model downloads (~24GB)"
echo ""

# Wait for health check to pass
MAX_WAIT=600  # 10 minutes
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if docker compose ps | grep -q "healthy"; then
        echo "✓ Service is healthy!"
        break
    fi
    
    if docker compose ps | grep -q "unhealthy"; then
        echo "✗ Service is unhealthy. Check logs with: docker compose logs -f"
        exit 1
    fi
    
    echo "Waiting... ($WAIT_TIME seconds elapsed)"
    sleep 10
    WAIT_TIME=$((WAIT_TIME + 10))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "Warning: Service did not become healthy within $MAX_WAIT seconds"
    echo "Check logs with: docker compose logs -f"
fi

echo ""
echo "=========================================="
echo "FLUX.1-dev Backend is Running!"
echo "=========================================="
echo ""
echo "API URL: http://localhost:${FLUX_SERVER_PORT:-8000}"
echo "Health Check: http://localhost:${FLUX_SERVER_PORT:-8000}/health"
echo "API Docs: http://localhost:${FLUX_SERVER_PORT:-8000}/docs"
echo ""
echo "View logs: docker compose logs -f"
echo "Stop service: docker compose down"
echo ""
echo "Test the API:"
echo "curl -X POST http://localhost:${FLUX_SERVER_PORT:-8000}/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"prompt\": \"A cat holding a sign that says hello world\"}' \\"
echo "  --output test.png"
echo ""

