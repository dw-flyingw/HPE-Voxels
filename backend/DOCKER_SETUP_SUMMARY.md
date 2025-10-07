# Docker Setup Summary

This document summarizes the Docker Compose setup for the FLUX backend.

## Files Created

### Core Docker Files

1. **`Dockerfile`**
   - Multi-stage build for FLUX.1-dev backend
   - Based on NVIDIA CUDA 12.1.1 runtime
   - Installs Python 3.11, PyTorch with CUDA support, and all dependencies
   - Exposes port 8000 (configurable via environment)
   - Includes health check

2. **`docker-compose.yml`**
   - Defines the `flux-backend` service
   - Configures GPU access via NVIDIA runtime
   - Maps ports from environment variables
   - Creates persistent volume for Hugging Face model cache
   - Sets up health checks and restart policies
   - Uses bridge network for container communication

3. **`.dockerignore`**
   - Excludes unnecessary files from Docker context
   - Reduces build time and image size
   - Prevents sensitive files (.env) from being copied

### Configuration Files

4. **`env.template`**
   - Template for environment configuration
   - Documents all available environment variables
   - Should be copied to `.env` and customized
   - **Important**: `.env` is gitignored (never commit tokens!)

### Scripts

5. **`docker-up.sh`** (executable)
   - One-command setup script
   - Checks prerequisites (Docker, GPU, etc.)
   - Creates .env from template if needed
   - Builds and starts the service
   - Waits for health check to pass
   - Displays usage instructions

### Documentation

6. **`DOCKER.md`**
   - Complete Docker setup guide
   - Prerequisites and installation instructions
   - Configuration reference
   - Development and production deployment
   - Troubleshooting guide
   - Security and performance notes

7. **`QUICKSTART_DOCKER.md`**
   - Quick reference for Docker usage
   - Common commands
   - Testing instructions
   - Troubleshooting checklist

8. **`DOCKER_SETUP_SUMMARY.md`** (this file)
   - Overview of the Docker setup
   - Architecture explanation

## Architecture

```
┌─────────────────────────────────────────────────┐
│                Docker Host                       │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │         flux-backend Container            │  │
│  │                                           │  │
│  │  ┌─────────────────────────────────────┐ │  │
│  │  │      FLUX.1-dev Model              │ │  │
│  │  │      (FastAPI Server)              │ │  │
│  │  │                                     │ │  │
│  │  │  - Text-to-Image Generation        │ │  │
│  │  │  - REST API on port 8000           │ │  │
│  │  │  - GPU Accelerated                 │ │  │
│  │  └─────────────────────────────────────┘ │  │
│  │                                           │  │
│  │  GPU Access: NVIDIA Runtime              │  │
│  │  Volume: huggingface-cache (~24GB)       │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │         NVIDIA GPU (H200)                 │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │
         │ Port 8000 (configurable)
         ▼
    External Access
```

## Key Features

### GPU Access
- Uses NVIDIA Container Toolkit
- Requires `nvidia-docker` runtime
- Automatic GPU device allocation
- Full CUDA support

### Port Configuration
- **Follows project convention**: Uses environment variables ([[memory:4376306]])
- Default port: 8000
- Configure via `FLUX_SERVER_PORT` in `.env`
- Both container and host ports use the same variable

### Persistent Storage
- Named volume: `huggingface-cache`
- Stores downloaded FLUX.1-dev model (~24GB)
- Persists between container restarts
- Remove with: `docker compose down -v`

### Health Checks
- Built-in health monitoring
- Checks `/health` endpoint every 30s
- 120s startup grace period (for model loading)
- Automatic unhealthy detection

### Environment Variables
All configuration via `.env` file:
- `HUGGINGFACE_TOKEN` - Required for model access
- `FLUX_SERVER_PORT` - Server port (default: 8000)
- `FLUX_MODEL_NAME` - Model to load
- `FLUX_DEFAULT_*` - Generation defaults
- See `env.template` for complete list

## Usage Patterns

### Development
```bash
# Start with logs
docker compose up

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Restart after code changes
docker compose restart
```

### Production
```bash
# Build and start
docker compose up -d --build

# Monitor
docker compose ps
docker compose logs -f

# Update
docker compose pull
docker compose up -d --build
```

### Maintenance
```bash
# Stop service
docker compose down

# Remove everything (including cache)
docker compose down -v

# Rebuild from scratch
docker compose build --no-cache
```

## Resource Requirements

### Minimum
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB
- **Disk**: 30GB (25GB for model + 5GB for container)
- **CUDA**: 12.1+

### Recommended (H200)
- **GPU**: NVIDIA H200 (141GB VRAM)
- **RAM**: 32GB+
- **Disk**: 50GB+
- **Network**: High bandwidth for initial model download

## Network Architecture

```
flux-network (bridge)
├── flux-backend (container)
│   └── Port mapping: 8000:8000
└── (Future services can be added here)
```

### Extending the Network
To add more services (e.g., frontend, database):

```yaml
services:
  flux-backend:
    # ... existing config ...
    networks:
      - flux-network
  
  frontend:
    # ... frontend config ...
    networks:
      - flux-network
    depends_on:
      - flux-backend
```

## Security Considerations

1. **Never commit `.env`** - Contains sensitive tokens
2. **Use HTTPS** - Add reverse proxy for production
3. **Firewall rules** - Restrict port access
4. **Token security** - Use read-only Hugging Face tokens
5. **Container isolation** - Service runs as non-root in production
6. **Network security** - Use internal Docker networks

## Monitoring

### Health Status
```bash
# Check if healthy
docker compose ps

# Manual health check
curl http://localhost:8000/health
```

### GPU Utilization
```bash
# On host
watch -n 1 nvidia-smi

# Inside container
docker compose exec flux-backend nvidia-smi
```

### Logs
```bash
# Real-time logs
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100

# Specific service
docker compose logs -f flux-backend
```

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| GPU not found | Install NVIDIA Container Toolkit |
| Port in use | Change `FLUX_SERVER_PORT` in `.env` |
| Model download fails | Check `HUGGINGFACE_TOKEN` and license acceptance |
| Out of memory | Check GPU with `nvidia-smi`, reduce image size |
| Container unhealthy | Check logs: `docker compose logs` |
| Slow startup | First run downloads 24GB model, be patient |

## Migration from Native

If you were running natively with `run_server.sh`:

1. **Copy your `.env`**:
   ```bash
   # Your existing .env should work as-is
   ```

2. **Stop native service**:
   ```bash
   # Kill python process or systemd service
   pkill -f flux_server.py
   # or
   sudo systemctl stop flux-server
   ```

3. **Start Docker service**:
   ```bash
   ./docker-up.sh
   ```

4. **Verify**:
   ```bash
   curl http://localhost:8000/health
   ```

The API endpoints remain the same, so clients don't need changes.

## Next Steps

1. ✅ Docker setup complete
2. 📝 Configure `.env` with your Hugging Face token
3. 🚀 Run `./docker-up.sh` to start
4. 🧪 Test with the example curl command
5. 📚 Read [DOCKER.md](DOCKER.md) for advanced usage
6. 🔒 Review security settings for production
7. 📊 Set up monitoring (optional)

## Support

- **Full Docker Guide**: [DOCKER.md](DOCKER.md)
- **Quick Reference**: [QUICKSTART_DOCKER.md](QUICKSTART_DOCKER.md)
- **Main README**: [README.md](README.md)
- **API Documentation**: http://localhost:8000/docs (when running)

