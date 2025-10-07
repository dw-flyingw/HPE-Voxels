#!/bin/bash

###############################################################################
# FLUX.1-dev Ubuntu Server Setup Script
# For Ubuntu servers with Nvidia H200 GPUs
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "FLUX.1-dev Ubuntu Server Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if running on Ubuntu
if [ ! -f /etc/os-release ]; then
    echo -e "${RED}✗ Cannot detect OS. This script is for Ubuntu.${NC}"
    exit 1
fi

source /etc/os-release
if [[ "$ID" != "ubuntu" ]]; then
    echo -e "${RED}✗ This script is designed for Ubuntu. Detected: $ID${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Detected Ubuntu $VERSION_ID${NC}"

# Update system packages
echo ""
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# Check for Nvidia drivers and CUDA
echo ""
echo "Checking Nvidia GPU setup..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ Nvidia drivers found${NC}"
    nvidia-smi
else
    echo -e "${YELLOW}⚠ Nvidia drivers not detected!${NC}"
    echo "Please install Nvidia drivers and CUDA toolkit:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create Python virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo ""
echo "Installing FLUX.1-dev dependencies..."
pip install -r requirements.txt

# Verify PyTorch CUDA installation
echo ""
echo "Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else \"N/A\"}')"

# Create .env file if it doesn't exist
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${YELLOW}⚠ IMPORTANT: You need to edit .env and add your Hugging Face token!${NC}"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token (read access is sufficient)"
    echo "3. Accept the FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev"
    echo "4. Edit .env file and set HUGGINGFACE_TOKEN=your_token_here"
    echo ""
    read -p "Do you want to enter your token now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your Hugging Face token: " HF_TOKEN
        sed -i "s/your_token_here/$HF_TOKEN/" .env
        echo -e "${GREEN}✓ .env file created and token configured${NC}"
    else
        echo -e "${YELLOW}⚠ Remember to edit .env and add your token before running the server${NC}"
    fi
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create systemd service file (optional)
echo ""
read -p "Do you want to create a systemd service for auto-start? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/flux-server.service"
    WORK_DIR="$SCRIPT_DIR"
    USER=$(whoami)
    
    echo "Creating systemd service..."
    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=FLUX.1-dev Text-to-Image API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment="PATH=$WORK_DIR/venv/bin:\$PATH"
ExecStart=$WORK_DIR/run_server.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo chmod 644 $SERVICE_FILE
    sudo systemctl daemon-reload
    
    echo -e "${GREEN}✓ Systemd service created${NC}"
    echo ""
    echo "To manage the service:"
    echo "  sudo systemctl start flux-server    # Start the server"
    echo "  sudo systemctl stop flux-server     # Stop the server"
    echo "  sudo systemctl enable flux-server   # Enable auto-start on boot"
    echo "  sudo systemctl status flux-server   # Check status"
    echo "  journalctl -u flux-server -f        # View logs"
fi

# Make run script executable
chmod +x run_server.sh

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Ensure your .env file has a valid HUGGINGFACE_TOKEN"
echo "2. Accept the FLUX.1-dev license at: https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo "3. Run the server with: ./run_server.sh"
echo "4. Test the API at: http://your-server-ip:8000/docs"
echo ""
echo "The first run will download the model (~24GB), which may take some time."
echo ""


