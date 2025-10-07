#!/bin/bash

###############################################################################
# FLUX.1-dev Server Runner Script
###############################################################################

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

# Check for Hugging Face token
if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_token_here" ]; then
    echo "Error: HUGGINGFACE_TOKEN not set in .env file!"
    echo "Please get your token from https://huggingface.co/settings/tokens"
    echo "and set it in the .env file."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Display configuration
echo "=========================================="
echo "FLUX.1-dev Server Configuration"
echo "=========================================="
echo "Model: $FLUX_MODEL_NAME"
echo "Host: $FLUX_HOST"
echo "Port: $FLUX_SERVER_PORT"
echo "=========================================="
echo ""

# Run the server
echo "Starting server..."
python3 flux_server.py

