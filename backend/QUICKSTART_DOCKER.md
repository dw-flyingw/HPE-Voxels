# Quick Start: FLUX Backend with Docker

## Prerequisites Checklist

- [ ] Docker installed
- [ ] NVIDIA Container Toolkit installed
- [ ] NVIDIA GPU available
- [ ] Hugging Face account created
- [ ] FLUX.1-dev license accepted at [huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [ ] Hugging Face token obtained from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## One-Command Setup

```bash
cd backend && ./docker-up.sh
```

The script will:
1. Check prerequisites
2. Create `.env` from template if needed
3. Prompt for Hugging Face token configuration
4. Build the Docker image
5. Start the service
6. Wait for it to become healthy

## Manual Setup (3 Steps)

### 1. Configure Environment
```bash
cd backend
cp env.template .env
nano .env  # Set your HUGGINGFACE_TOKEN
```

### 2. Start Service
```bash
docker compose up -d
```

### 3. Check Status
```bash
# View logs
docker compose logs -f

# Check health
curl http://localhost:8000/health
```

## Test the API

```bash
# Generate an image
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat holding a sign that says hello world",
    "width": 1024,
    "height": 1024
  }' \
  --output test.png
```

## Common Commands

```bash
# View logs
docker compose logs -f

# Stop service
docker compose down

# Restart service
docker compose restart

# Rebuild and restart
docker compose up -d --build

# Remove everything including model cache
docker compose down -v
```

## API Endpoints

- Health: `http://localhost:8000/health`
- Generate: `http://localhost:8000/generate`
- API Docs: `http://localhost:8000/docs`

## Environment Variables

Key variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | - | **Required**: Your HF token |
| `FLUX_SERVER_PORT` | 8000 | Server port |
| `FLUX_MODEL_NAME` | black-forest-labs/FLUX.1-dev | Model to use |
| `FLUX_DEFAULT_HEIGHT` | 1024 | Default image height |
| `FLUX_DEFAULT_WIDTH` | 1024 | Default image width |

See `env.template` for all options.

## Troubleshooting

### GPU not available
```bash
# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Port already in use
Change `FLUX_SERVER_PORT` in `.env`, then restart:
```bash
docker compose down
docker compose up -d
```

### Service unhealthy
```bash
# Check logs
docker compose logs flux-backend

# Restart
docker compose restart flux-backend
```

### Model download failed
- Verify Hugging Face token is set correctly
- Check you've accepted the FLUX.1-dev license
- Ensure internet connectivity

## First Run Notes

- **Initial startup takes 5-10 minutes** to download the model (~24GB)
- Model is cached in Docker volume `huggingface-cache`
- Subsequent starts are much faster (~30 seconds)
- Watch progress: `docker compose logs -f`

## Complete Documentation

See [DOCKER.md](DOCKER.md) for detailed documentation including:
- NVIDIA Container Toolkit installation
- Development mode setup
- Production deployment
- Performance optimization
- Security considerations

