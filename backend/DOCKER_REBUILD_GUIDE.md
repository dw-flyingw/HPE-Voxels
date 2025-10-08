# Docker Rebuild Guide - UV Conditioning Update

## Problem

The Docker container was built before the UV conditioning dependencies were added, so it's missing:
- `opencv-python` (cv2)
- `controlnet-aux`
- System libraries for OpenCV

## Solution

Rebuild the Docker container with the updated dependencies.

---

## Quick Rebuild

### Option 1: Automated Script (Recommended)

```bash
cd backend
./rebuild_docker.sh
```

This will:
1. Stop existing containers
2. Rebuild with fresh dependencies
3. Start the updated container
4. Show logs

### Option 2: Manual Steps

```bash
cd backend

# Stop existing containers
docker compose down

# Rebuild (no cache to ensure fresh build)
docker compose build --no-cache

# Start the service
docker compose up -d

# Check logs
docker compose logs -f
```

---

## What Was Updated

### 1. `pyproject.toml`
Added to dependencies:
```python
"opencv-python>=4.8.0",
"controlnet-aux>=0.0.7",
```

### 2. `Dockerfile`
Added OpenCV system libraries:
```dockerfile
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
```

---

## Verification

After rebuild, verify the container is working:

### Check Container Status
```bash
docker compose ps
```

Expected output:
```
NAME           IMAGE              STATUS
flux-backend   backend-backend    Up X minutes
```

### Check Server Health
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  ...
}
```

### Test UV Conditioning
```bash
# From project root
python backend/test_uv_conditioning.py --model colon --size 1024
```

---

## Troubleshooting

### Build Fails with "No space left on device"

```bash
# Clean up old Docker images
docker system prune -a

# Try rebuild again
./rebuild_docker.sh
```

### Container Starts but Crashes

Check logs:
```bash
docker compose logs --tail=100
```

Common issues:
- Missing `.env` file (copy from `env.template`)
- Invalid `HUGGINGFACE_TOKEN` in `.env`
- Insufficient GPU memory

### "Cannot connect to Docker daemon"

Make sure Docker is running:
```bash
sudo systemctl start docker  # Linux
# or start Docker Desktop on Mac/Windows
```

### Build is Very Slow

First build can take 10-15 minutes:
- Downloading CUDA base image (~4GB)
- Installing PyTorch and dependencies
- Downloading FLUX model (~24GB)

Subsequent rebuilds are faster due to layer caching.

---

## Complete Rebuild (Fresh Start)

If you need to completely reset:

```bash
cd backend

# Stop and remove containers, volumes, and images
docker compose down -v
docker rmi $(docker images -q flux-backend)

# Rebuild from scratch
docker compose build --no-cache

# Start fresh
docker compose up -d
```

**Warning**: This will re-download the FLUX model (~24GB).

---

## Expected Build Time

- **First build**: 10-15 minutes (+ FLUX model download)
- **Rebuild with cache**: 2-3 minutes
- **Full rebuild (no cache)**: 5-7 minutes

---

## After Successful Rebuild

Your FLUX server will now support UV conditioning! Test it:

```bash
# From project root
cd frontend
python utils/generate_flux_texture.py --organ colon --size 2048
```

You should see improved texture quality with:
- ✅ UV layout awareness
- ✅ Non-repeating patterns
- ✅ Smooth boundaries
- ✅ Better coherence

---

## Files Modified

1. `backend/Dockerfile` - Added OpenCV system dependencies
2. `backend/pyproject.toml` - Added Python dependencies
3. `backend/requirements.txt` - Added dependencies (already done)
4. `backend/flux_server.py` - UV conditioning implementation (already done)

---

## Need Help?

1. Check container logs: `docker compose logs -f`
2. Verify dependencies: `docker compose exec flux-backend pip list | grep -E "opencv|controlnet"`
3. Test imports: `docker compose exec flux-backend python -c "import cv2; print('OK')"`

---

**Ready to rebuild?** Run `./rebuild_docker.sh` from the `backend` directory!

