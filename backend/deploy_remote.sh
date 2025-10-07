#!/bin/bash

###############################################################################
# Remote Flux Server Deployment Script
###############################################################################

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Deploying Flux Server with UV-Guided Generation"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "flux_server.py" ]; then
    echo "❌ Error: flux_server.py not found. Please run this script from the backend directory."
    exit 1
fi

# Check for required files
echo "📋 Checking required files..."
required_files=("flux_server.py" "requirements.txt" "env.template")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file $file not found!"
        exit 1
    fi
done

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.template .env
    echo "⚠️  Please edit .env file with your configuration before starting the server!"
    echo "   Especially set your HUGGINGFACE_TOKEN"
    exit 1
fi

# Load environment variables
echo "🔧 Loading environment variables..."
export $(cat .env | grep -v '^#' | xargs)

# Check for Hugging Face token
if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_token_here" ]; then
    echo "❌ Error: HUGGINGFACE_TOKEN not set in .env file!"
    echo "Please get your token from https://huggingface.co/settings/tokens"
    echo "and set it in the .env file."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📦 Installing/updating Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional packages for UV-guided generation
echo "📦 Installing additional packages for UV-guided generation..."
pip install numpy scipy pillow requests

# Display configuration
echo ""
echo "=========================================="
echo "🚀 Flux Server Configuration"
echo "=========================================="
echo "Model: $FLUX_MODEL_NAME"
echo "Host: $FLUX_HOST"
echo "Port: $FLUX_SERVER_PORT"
echo "Device: CUDA (if available)"
echo "Features: UV-Guided Generation ✅"
echo "=========================================="
echo ""

# Test GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available - will use CPU (slower)')
"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🚀 To start the server, run:"
echo "   ./start_server.sh"
echo ""
echo "🔍 To test the server, run:"
echo "   ./test_server.sh"
echo ""
echo "📊 To check server status, run:"
echo "   curl http://localhost:$FLUX_SERVER_PORT/health"
