#!/bin/bash
# Run the OBJ Texture Generator Streamlit App

# Check if .env exists in project root
if [ ! -f ../.env ]; then
    echo "Warning: .env file not found in project root"
    echo "Creating .env with default values..."
    cat > ../.env << EOF
# Environment Configuration for HPE-Voxels
FLUX_SERVER_PORT=8000
FLUX_HOST=localhost
HUGGINGFACE_TOKEN=your_token_here
FLUX_MODEL_NAME=black-forest-labs/FLUX.1-dev
FLUX_TORCH_DTYPE=bfloat16
FLUX_DEFAULT_HEIGHT=1024
FLUX_DEFAULT_WIDTH=1024
FLUX_DEFAULT_GUIDANCE_SCALE=3.5
FLUX_DEFAULT_NUM_STEPS=50
FLUX_DEFAULT_MAX_SEQ_LENGTH=512
STREAMLIT_SERVER_PORT=8501
EOF
    echo "Please update ../.env with your Hugging Face token"
fi

# Check if Flux server is running
echo "Checking Flux server status..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "WARNING: Flux server is not running at http://localhost:8000"
    echo "Please start the backend server first:"
    echo "  cd backend"
    echo "  ./run_server.sh"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Get port from .env or use default
PORT=${STREAMLIT_SERVER_PORT:-8501}

echo ""
echo "Starting OBJ Texture Generator..."
echo "Access the app at: http://localhost:$PORT"
echo ""

# Run Streamlit
streamlit run obj_texture_generator.py --server.port $PORT

