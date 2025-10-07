# FLUX.1-dev Backend Server

This is the backend API server for the FLUX.1-dev text-to-image generation model, optimized for Ubuntu servers with Nvidia H200 GPUs.

## Overview

The backend provides a FastAPI-based REST API for generating images from text prompts using the FLUX.1-dev model from Black Forest Labs.

## Prerequisites

- Ubuntu Server (20.04 or later)
- Nvidia H200 GPU (or other CUDA-capable GPU)
- Nvidia drivers and CUDA toolkit installed
- Python 3.11+
- At least 24GB of free disk space for the model
- Hugging Face account and API token

## Quick Start

### 1. Get Your Hugging Face Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read access is sufficient)
3. Accept the FLUX.1-dev license at [https://huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

### 2. Setup on Ubuntu Server

After cloning/pulling the repository to your Ubuntu server:

```bash
cd backend
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

This script will:
- Install system dependencies
- Check for Nvidia drivers
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install all required packages
- Set up environment configuration
- Optionally create a systemd service

### 3. Configure Environment

Edit the `.env` file and add your Hugging Face token:

```bash
nano .env
```

Set:
```
HUGGINGFACE_TOKEN=your_actual_token_here
```

### 4. Run the Server

```bash
./run_server.sh
```

The server will start on port 8000 (configurable in `.env`).

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Image

**Using curl (get PNG):**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50
  }' \
  --output generated.png
```

**Using curl (get base64):**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "return_base64": true
  }'
```

**Interactive API Documentation:**
Visit `http://your-server-ip:8000/docs` for the interactive Swagger UI.

## Configuration

All configuration is done via environment variables in the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | - | Your Hugging Face API token (required) |
| `FLUX_SERVER_PORT` | 8000 | Port to run the server on |
| `FLUX_HOST` | 0.0.0.0 | Host to bind to |
| `FLUX_MODEL_NAME` | black-forest-labs/FLUX.1-dev | Model to load |
| `FLUX_TORCH_DTYPE` | bfloat16 | Torch dtype (bfloat16 or float16) |
| `FLUX_DEFAULT_HEIGHT` | 1024 | Default image height |
| `FLUX_DEFAULT_WIDTH` | 1024 | Default image width |
| `FLUX_DEFAULT_GUIDANCE_SCALE` | 3.5 | Default guidance scale |
| `FLUX_DEFAULT_NUM_STEPS` | 50 | Default inference steps |
| `FLUX_DEFAULT_MAX_SEQ_LENGTH` | 512 | Default max sequence length |

## Systemd Service (Optional)

If you chose to create a systemd service during setup, you can manage the server with:

```bash
# Start the server
sudo systemctl start flux-server

# Stop the server
sudo systemctl stop flux-server

# Enable auto-start on boot
sudo systemctl enable flux-server

# Check status
sudo systemctl status flux-server

# View logs
journalctl -u flux-server -f
```

## Performance Notes

### H200 GPU Optimization

The H200 has 141GB of HBM3e memory, which is more than enough for FLUX.1-dev. You may want to:

1. **Disable CPU offloading** for better performance (edit `flux_server.py`):
   ```python
   # Comment out this line:
   # pipeline.enable_model_cpu_offload()
   ```

2. **Adjust batch processing** if you plan to generate multiple images simultaneously.

3. **Monitor GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Troubleshooting

### Model fails to download
- Ensure you have accepted the FLUX.1-dev license on Hugging Face
- Verify your token has the correct permissions
- Check your internet connection and firewall settings

### CUDA errors
- Verify Nvidia drivers are installed: `nvidia-smi`
- Check CUDA is available in PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure CUDA toolkit version matches PyTorch requirements

### Out of memory errors
- Enable CPU offloading (it's enabled by default)
- Reduce image dimensions
- Reduce num_inference_steps

## License

This backend uses the FLUX.1-dev model which falls under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

Generated outputs can be used for personal, scientific, and commercial purposes as described in the license.


