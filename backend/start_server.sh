#!/bin/bash

###############################################################################
# Flux Server Startup Script
###############################################################################

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting Flux Server with UV-Guided Generation"
echo "================================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please run ./deploy_remote.sh first to set up the environment."
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please run ./deploy_remote.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check for required packages
echo "🔍 Checking dependencies..."
python3 -c "
try:
    import torch, diffusers, fastapi, uvicorn, PIL, numpy, scipy
    print('✅ All required packages are installed')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    print('Please run ./deploy_remote.sh to install dependencies')
    exit(1)
"

# Display server information
echo ""
echo "=========================================="
echo "🚀 Flux Server Starting"
echo "=========================================="
echo "Model: $FLUX_MODEL_NAME"
echo "Host: $FLUX_HOST"
echo "Port: $FLUX_SERVER_PORT"
echo "Features: UV-Guided Generation ✅"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Start the server with logging
echo "🚀 Starting server..."
echo "📝 Logs will be saved to: logs/flux_server.log"
echo "🌐 Server will be available at: http://$FLUX_HOST:$FLUX_SERVER_PORT"
echo "📚 API docs will be available at: http://$FLUX_HOST:$FLUX_SERVER_PORT/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start server with proper logging
python3 flux_server.py 2>&1 | tee logs/flux_server.log
