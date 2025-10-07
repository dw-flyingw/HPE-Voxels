# 🚀 Quick Start Guide: Remote Flux Server with UV-Guided Generation

This guide will help you deploy and run the Flux server with UV-guided texture generation capabilities on your remote server.

## 📋 Prerequisites

1. **Remote server with GPU** (recommended: NVIDIA H200, A100, or similar)
2. **Python 3.8+** installed
3. **CUDA drivers** installed (for GPU acceleration)
4. **Hugging Face account** with access to FLUX.1-dev model
5. **Git** installed for pulling code updates

## 🚀 Deployment Steps

### 1. Pull the Latest Code

```bash
# On your remote server
git pull origin master
cd backend
```

### 2. Configure Environment

```bash
# Copy the environment template
cp env.template .env

# Edit the .env file with your configuration
nano .env
```

**Required configuration in .env:**
```bash
# Your Hugging Face token (REQUIRED)
HUGGINGFACE_TOKEN=your_actual_token_here

# Server configuration
FLUX_SERVER_PORT=8000
FLUX_HOST=0.0.0.0  # Allow external connections

# Model configuration
FLUX_MODEL_NAME=black-forest-labs/FLUX.1-dev
FLUX_TORCH_DTYPE=bfloat16  # Recommended for H200 GPUs
```

### 3. Deploy and Install Dependencies

```bash
# Make scripts executable
chmod +x *.sh

# Run deployment script
./deploy_remote.sh
```

This script will:
- ✅ Create Python virtual environment
- ✅ Install all required dependencies
- ✅ Update requirements with UV-guided generation packages
- ✅ Check GPU availability
- ✅ Validate configuration

### 4. Start the Server

```bash
# Start the Flux server
./start_server.sh
```

The server will:
- 🔄 Load the FLUX.1-dev model (may take 5-10 minutes)
- 🌐 Start serving on `http://your-server:8000`
- 📝 Log all activity to `logs/flux_server.log`

### 5. Test the Server

```bash
# In another terminal, test the server
./test_server.sh
```

## 🔗 Available Endpoints

Once running, your server provides these endpoints:

| Endpoint | Description | Usage |
|----------|-------------|-------|
| `/health` | Server health check | `curl http://your-server:8000/health` |
| `/generate` | Standard text-to-image | Standard Flux generation |
| `/generate_with_control` | **UV-guided generation** | For medical organ textures |
| `/docs` | API documentation | Interactive API docs |

## 🎨 Using UV-Guided Generation

### From Your Local Machine

1. **Set the server URL:**
```bash
export FLUX_SERVER_URL="http://your-server:8000"
```

2. **Generate colon texture:**
```bash
python generate_colon_flux_texture.py --size 1024 --overwrite
```

3. **Generate textures for all organs:**
```bash
python generate_flux_uv_textures.py --size 1024 --overwrite
```

4. **Use the Model Viewer:**
```bash
cd frontend
python model_viewer.py
```
Then select "Flux UV-Guided (Recommended)" in the generation method dropdown.

## 🔧 Server Management

### Check Server Status
```bash
curl http://your-server:8000/health
```

### View Logs
```bash
tail -f logs/flux_server.log
```

### Stop Server
```bash
# Press Ctrl+C in the server terminal
# Or find and kill the process
pkill -f flux_server.py
```

### Restart Server
```bash
./start_server.sh
```

## 🚨 Troubleshooting

### Server Won't Start
1. Check GPU availability: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check logs: `cat logs/flux_server.log`

### Model Loading Issues
1. Verify Hugging Face token: `echo $HUGGINGFACE_TOKEN`
2. Check internet connectivity
3. Ensure you have access to FLUX.1-dev model

### Memory Issues
1. Check GPU memory: `nvidia-smi`
2. Reduce batch size in generation requests
3. Use `FLUX_TORCH_DTYPE=float16` for lower memory usage

### Network Issues
1. Check firewall settings
2. Verify `FLUX_HOST=0.0.0.0` for external access
3. Test connectivity: `curl http://localhost:8000/health`

## 📊 Performance Tips

### For H200 GPUs (141GB VRAM)
```bash
FLUX_TORCH_DTYPE=bfloat16
FLUX_DEFAULT_HEIGHT=1024
FLUX_DEFAULT_WIDTH=1024
FLUX_DEFAULT_NUM_STEPS=50
```

### For Lower Memory GPUs
```bash
FLUX_TORCH_DTYPE=float16
FLUX_DEFAULT_HEIGHT=512
FLUX_DEFAULT_WIDTH=512
FLUX_DEFAULT_NUM_STEPS=30
```

## 🔄 Updating the Server

When you push new code:

```bash
# On remote server
git pull origin master
cd backend

# Update dependencies if needed
source venv/bin/activate
pip install -r requirements.txt

# Restart server
./start_server.sh
```

## 📞 Support

If you encounter issues:

1. Check the logs: `tail -f logs/flux_server.log`
2. Test connectivity: `./test_server.sh`
3. Verify GPU status: `nvidia-smi`
4. Check environment: `cat .env`

---

🎉 **Your Flux server with UV-guided generation is now ready for hyper-realistic medical organ texture generation!**
