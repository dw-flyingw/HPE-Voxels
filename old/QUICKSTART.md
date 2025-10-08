# Quick Start Guide: OBJ Texture Generator

This guide will help you quickly set up and use the OBJ Texture Generator to create hyper-realistic textures for your 3D models.

## Prerequisites

- Python 3.11+
- Nvidia GPU with CUDA support (for backend)
- Hugging Face account and API token

## Step 1: Install Dependencies

### Backend (Flux Server)

```bash
cd backend
./setup_ubuntu.sh
```

### Frontend (Streamlit App)

```bash
cd frontend
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r frontend/requirements.txt
```

## Step 2: Configure Environment

1. **Get your Hugging Face token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token
   - Accept the FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev

2. **Create .env file in project root:**

```bash
cat > .env << EOF
FLUX_SERVER_PORT=8000
FLUX_HOST=localhost
HUGGINGFACE_TOKEN=your_actual_token_here
FLUX_MODEL_NAME=black-forest-labs/FLUX.1-dev
FLUX_TORCH_DTYPE=bfloat16
FLUX_DEFAULT_HEIGHT=1024
FLUX_DEFAULT_WIDTH=1024
FLUX_DEFAULT_GUIDANCE_SCALE=3.5
FLUX_DEFAULT_NUM_STEPS=50
FLUX_DEFAULT_MAX_SEQ_LENGTH=512
STREAMLIT_SERVER_PORT=8501
EOF
```

**Important:** Replace `your_actual_token_here` with your Hugging Face token!

## Step 3: Start the Backend

```bash
cd backend
./run_server.sh
```

Wait for the message: "✓ Model loaded successfully"

This may take a few minutes on first run as it downloads the model (~24GB).

## Step 4: Start the Frontend

In a new terminal:

```bash
cd frontend
./run_texture_generator.sh
```

Or manually:

```bash
cd frontend
streamlit run obj_texture_generator.py
```

The app will open at http://localhost:8501

## Step 5: Generate Your First Texture

1. **Upload an OBJ file:**
   - Click "Choose an OBJ file"
   - Select `output/obj/colon.obj` (or any other model)

2. **Enter a prompt:**
   ```
   hyper photo realistic colon tissue with haustra and natural surface details, medical grade quality, 8k resolution
   ```

3. **Adjust settings** (optional, in sidebar):
   - Texture Size: 1024 (recommended)
   - Guidance Scale: 3.5
   - Inference Steps: 50

4. **Generate:**
   - Click "🎨 Generate Texture"
   - Wait ~30-60 seconds
   - Download the textured OBJ, texture image, and MTL file

## Example Workflow

### Generate Textured Colon

```bash
# 1. Start backend
cd backend && ./run_server.sh

# 2. In new terminal, start frontend
cd frontend && ./run_texture_generator.sh

# 3. In the web app:
#    - Upload: output/obj/colon.obj
#    - Prompt: "hyper photo realistic colon tissue with haustra and natural surface details, medical grade quality, 8k resolution"
#    - Click: Generate Texture
#    - Download: textured_model.obj, texture.png, textured_model.mtl
```

### Generate Textured Heart

```bash
# Same steps, but use:
#    - Upload: output/obj/heart.obj
#    - Prompt: "hyper photo realistic cardiac muscle tissue with blood vessels, medical photography, 8k resolution"
```

## Tips for Best Results

### Prompt Engineering

✅ **Good Prompts:**
- "hyper photo realistic colon tissue with haustra, medical grade quality, 8k resolution"
- "hyper photo realistic cardiac muscle with visible blood vessels, clinical photography, 8k"
- "hyper photo realistic liver tissue surface, medical imaging quality, 8k resolution"

❌ **Avoid:**
- Too vague: "realistic organ"
- Too complex: "colon with bacteria and digestive food particles in bright colors"
- Unrelated elements: "colon tissue on a wooden table"

### Generation Settings

| Setting | Low Quality (Fast) | High Quality (Slow) | Recommended |
|---------|-------------------|---------------------|-------------|
| Texture Size | 512 | 2048 | 1024 |
| Guidance Scale | 2.0 | 7.0 | 3.5 |
| Inference Steps | 20 | 100 | 50 |

### For Medical Models

1. Always include "medical grade" or "medical photography"
2. Mention specific anatomical features
3. Use "8k resolution" for detail
4. Keep guidance scale moderate (3-5) for realistic results

## Troubleshooting

### Backend won't start

```bash
# Check CUDA
nvidia-smi

# Check token
cat .env | grep HUGGINGFACE_TOKEN

# Check logs
cd backend
python flux_server.py
```

### Frontend can't connect to backend

```bash
# Check if backend is running
curl http://localhost:8000/health

# Should return: {"status":"healthy",...}
```

### Generation is too slow

- Reduce texture size to 512
- Reduce inference steps to 30
- Check GPU utilization: `nvidia-smi`

### Texture quality is poor

- Increase texture size to 2048
- Increase inference steps to 75
- Improve your prompt (be more specific)
- Try different guidance scales (3-7)

## Model Viewer with AI Texture Generation

The Model Viewer app now includes built-in AI texture generation!

### Usage:

```bash
# Start the backend (if not already running)
cd backend && ./run_server.sh

# In a new terminal, start Model Viewer
cd frontend
python model_viewer.py
# Opens at http://localhost:8503
```

### Workflow:

1. **Select a model** from the dropdown (e.g., "colon")
2. **Review model info** - see current textures/materials
3. **Click "🚀 Generate Photo-Realistic Texture"** in the sidebar
4. **Wait ~1-2 minutes** for AI generation
5. **Reload the page** to view the enhanced model

### Features:

- **Auto-generated prompts** for all 130+ Nvidia Vista3D anatomical structures
  - Organs, blood vessels, bones, muscles, lungs, airways, and more
  - Hyper photo-realistic, medically accurate descriptions
  - Anatomically correct coloration and texture details
- **Custom prompts** for specialized textures
- **Adjustable texture size** (512, 1024, 2048 - 8K quality)
- **Direct model enhancement** - textures saved to model folder
- **Automatic UV mapping** if not present
- **Multiple format support** (GLTF, GLB, OBJ)

### Example:

```bash
# 1. Start backend
cd backend && ./run_server.sh

# 2. Start Model Viewer
cd frontend && python model_viewer.py

# 3. In the web app:
#    - Select: colon (from dropdown)
#    - Review: Auto-generated prompt
#    - Click: 🚀 Generate Photo-Realistic Texture
#    - Wait: ~60-120 seconds
#    - Reload: Page to see textured model
```

### Comparison: Model Viewer vs OBJ Texture Generator

| Feature | Model Viewer | OBJ Texture Generator |
|---------|-------------|----------------------|
| Model Selection | Dropdown | File upload |
| Formats Supported | GLTF, GLB, OBJ, STL, PLY | OBJ only |
| Auto Prompts | ✅ Yes | ❌ No |
| UV Mapping | ✅ Automatic | ✅ Automatic |
| Output Location | Model folder | Downloads |
| Best For | Existing models | Custom uploads |

## Next Steps

- Explore different prompts for various organs
- Adjust generation parameters for quality/speed tradeoff
- Use the generated textures in 3D modeling software (Blender, etc.)
- View models with existing viewers:
  - `python model_viewer.py` - **With AI texture generation** 🆕
  - `streamlit run glb_viewer.py`
  - `streamlit run obj_viewer.py`

## Project Structure

```
HPE-Voxels/
├── backend/              # Flux AI server
│   ├── flux_server.py
│   └── run_server.sh
├── frontend/             # Streamlit apps
│   ├── obj_texture_generator.py    # Upload-based texture generator
│   ├── model_viewer.py             # Viewer + AI texture generation 🆕
│   ├── logic/
│   │   ├── obj_texture_generator_logic.py
│   │   └── model_viewer_logic.py   # AI texture generation logic 🆕
│   └── run_texture_generator.sh
├── output/
│   ├── obj/              # Input OBJ files
│   ├── glb/              # Input GLB files
│   └── models/           # Model folders (used by Model Viewer)
├── .env                  # Configuration (YOU MUST CREATE THIS)
└── QUICKSTART.md         # This file
```

## Support

For issues or questions:
1. Check the backend logs: `journalctl -u flux-server -f` (if using systemd)
2. Check the frontend logs in the terminal
3. Verify .env configuration
4. Ensure backend is running and healthy

---

Happy texture generation! 🎨

