# FLUX Backend Docker Setup

This guide explains how to run the FLUX.1-dev backend using Docker Compose with GPU support.

## Prerequisites

### 1. Docker and Docker Compose
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose (if not included)
sudo apt-get install docker-compose-plugin
```

### 2. NVIDIA Container Toolkit
Required for GPU access in Docker containers:

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 3. Hugging Face Token
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read access is sufficient)
3. Accept the FLUX.1-dev license at [https://huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

## Quick Start

### 1. Configure Environment
```bash
cd backend

# Copy the template environment file
cp env.template .env

# Edit the .env file and set your Hugging Face token
nano .env
```

Set at minimum:
```
HUGGINGFACE_TOKEN=your_actual_token_here
```

### 2. Build and Run
```bash
# Build the Docker image
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# The server will be available at http://localhost:8000
```

## Docker Compose Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Rebuild and start
docker-compose up -d --build

# View logs
docker-compose logs -f flux-backend

# Check service status
docker-compose ps

# Execute commands in container
docker-compose exec flux-backend bash

# Remove all data (including model cache)
docker-compose down -v
```

## Configuration

All configuration is done via the `.env` file:

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

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate an Image
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

### Interactive API Documentation
Visit: http://localhost:8000/docs

## Volumes

The Docker setup uses a named volume for persistent data:

- `huggingface-cache`: Stores downloaded models (~24GB)
  - Persists between container restarts
  - Remove with: `docker-compose down -v` (will require re-downloading the model)

## Development Mode

To enable live code reloading during development, uncomment the volume mount in `docker-compose.yml`:

```yaml
volumes:
  - huggingface-cache:/root/.cache/huggingface
  - ./flux_server.py:/app/flux_server.py  # Uncomment this line
```

Then restart the container:
```bash
docker-compose restart flux-backend
```

## Troubleshooting

### GPU Not Available
```bash
# Check if GPU is accessible in Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Check NVIDIA Container Toolkit installation
dpkg -l | grep nvidia-container-toolkit
```

### Model Download Issues
- Ensure you have accepted the FLUX.1-dev license on Hugging Face
- Verify your token has the correct permissions
- Check your internet connection
- View download progress in logs: `docker-compose logs -f`

### Out of Memory
- Check available GPU memory: `nvidia-smi`
- Reduce image dimensions in requests
- Reduce `num_inference_steps`

### Port Already in Use
Change the port in `.env`:
```
FLUX_SERVER_PORT=8001
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

### Container Health Check Failing
```bash
# Check container logs
docker-compose logs flux-backend

# Check if service is responding
docker-compose exec flux-backend curl http://localhost:8000/health

# Restart the container
docker-compose restart flux-backend
```

## Performance Optimization

### For H200 GPU
The H200 has 141GB of memory, so you can:

1. **Disable CPU offloading** for better performance
2. **Increase batch size** if processing multiple requests
3. **Use bfloat16** for optimal H200 performance (default)

### Monitor GPU Usage
```bash
# On host system
watch -n 1 nvidia-smi

# Inside container
docker-compose exec flux-backend nvidia-smi
```

## Security Notes

1. **Never commit `.env` file** - It contains your Hugging Face token
2. **Restrict port access** - Use firewall rules to limit access
3. **Use HTTPS** - In production, use a reverse proxy (nginx/traefik) with SSL
4. **API authentication** - Consider adding authentication for production use

## Production Deployment

For production deployment:

1. **Use a reverse proxy** (nginx, traefik, caddy)
2. **Enable HTTPS** with Let's Encrypt
3. **Add authentication** to the API
4. **Set up monitoring** (Prometheus, Grafana)
5. **Configure log aggregation** (ELK stack, Loki)
6. **Use docker-compose secrets** instead of .env file
7. **Set resource limits** in docker-compose.yml

Example with resource limits:
```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 32G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## License

This backend uses the FLUX.1-dev model which falls under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

