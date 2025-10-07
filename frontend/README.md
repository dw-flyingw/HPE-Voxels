# HPE-Voxels Frontend

Streamlit-based frontend applications for 3D medical model visualization and texture generation.

## Applications

### 1. OBJ Texture Generator (`obj_texture_generator.py`)

Upload OBJ files and generate hyper-realistic textures using the Flux AI model.

**Features:**
- Upload any `.obj` file (colon, heart, liver, etc.)
- Generate photorealistic textures from text prompts
- Automatic UV mapping if not present
- Adjustable generation parameters (size, guidance, steps)
- Download textured models with materials

**Usage:**
```bash
cd frontend
streamlit run obj_texture_generator.py
```

The app will be available at `http://localhost:8501`

### 2. Model Viewer (`model_viewer.py`)

View and inspect 3D models with AI-powered texture generation.

**Features:**
- Browse models from `output/models/` directory
- Support for multiple formats (GLB, GLTF, OBJ, PLY, STL, DAE)
- Interactive 3D visualization with Three.js
- Mouse controls: Left-click to rotate, Right-click to pan, Scroll to zoom
- Material and texture information display
- **AI Texture Generation:** Generate hyper photo-realistic textures using Flux AI
  - Auto-generated prompts for 130+ Nvidia Vista3D anatomical structures
  - Supports organs, blood vessels, bones, muscles, lungs, airways
  - Medically accurate, anatomically correct texture generation
- Automatic texture mapping and model enhancement

**Usage:**
```bash
# Default port (8503)
python model_viewer.py

# Custom port
MODEL_VIEWER_PORT=8504 python model_viewer.py
```

**AI Texture Generation Workflow:**
1. Select a model from the dropdown
2. The app will show current texture/material info
3. Click "🚀 Generate Photo-Realistic Texture" in the sidebar
4. The AI generates a texture based on the organ type (or custom prompt)
5. Texture is automatically applied and saved
6. Reload the page to view the enhanced model

**Note:** Requires Flux backend server to be running for AI texture generation.

### 3. GLB Viewer (`glb_viewer.py`)

View GLB/GLTF models with textures.

**Usage:**
```bash
streamlit run glb_viewer.py
```

### 4. OBJ Viewer (`obj_viewer.py`)

Simple OBJ file viewer.

**Usage:**
```bash
streamlit run obj_viewer.py
```

## Installation

### Prerequisites

- Python 3.11+
- Active Flux backend server (see `/backend/README.md`)

### Setup

1. **Install dependencies:**

Using uv (recommended):
```bash
cd frontend
uv pip install -r requirements.txt
```

Or using pip:
```bash
pip install -r requirements.txt
```

2. **Configure environment:**

Make sure the `.env` file in the project root has the correct Flux server URL:
```bash
FLUX_SERVER_PORT=8000
FLUX_HOST=localhost
```

3. **Start the Flux backend:**

See `/backend/README.md` for instructions on starting the Flux server.

4. **Run the Streamlit app:**

```bash
streamlit run obj_texture_generator.py
```

## Configuration

The app reads configuration from the `.env` file in the project root:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLUX_SERVER_PORT` | 8000 | Port where Flux server is running |
| `FLUX_HOST` | localhost | Host where Flux server is running |
| `STREAMLIT_SERVER_PORT` | 8501 | Port for Streamlit app |

## Workflow

### Generating Textured Models

1. **Start the Flux Backend Server**
   ```bash
   cd backend
   ./run_server.sh
   ```

2. **Start the Frontend App**
   ```bash
   cd frontend
   streamlit run obj_texture_generator.py
   ```

3. **Upload an OBJ File**
   - Click "Choose an OBJ file"
   - Select a model from `output/obj/` (e.g., `colon.obj`)
   - View model information

4. **Enter a Texture Prompt**
   - Example: "hyper photo realistic colon tissue with natural surface details, medical grade quality, 8k resolution"
   - Or use one of the example prompts

5. **Adjust Generation Settings** (in sidebar)
   - Texture Size: 512, 1024, or 2048
   - Guidance Scale: How closely to follow the prompt
   - Inference Steps: Quality vs speed tradeoff
   - Seed: For reproducible results

6. **Generate and Download**
   - Click "Generate Texture"
   - Wait for generation (typically 30-60 seconds)
   - Download the textured OBJ, texture image, and MTL file

## Example Prompts

### Medical Organs

**Colon:**
```
hyper photo realistic colon tissue with haustra and natural surface details, medical grade quality, 8k resolution
```

**Heart:**
```
hyper photo realistic cardiac muscle tissue with blood vessels, medical photography, 8k resolution
```

**Liver:**
```
hyper photo realistic liver tissue with smooth surface and blood vessels, medical grade, 8k resolution
```

**Aorta:**
```
hyper photo realistic arterial wall tissue with smooth surface, medical imaging quality, 8k resolution
```

### Tips for Best Results

1. **Use "hyper photo realistic"** in prompts for medical accuracy
2. **Include "medical grade" or "medical photography"** for clinical quality
3. **Specify "8k resolution" or "4k resolution"** for detail
4. **Mention specific anatomical features** (e.g., "haustra" for colon)
5. **Higher guidance scale** (4-7) for more literal interpretation
6. **More inference steps** (50-100) for better quality

## Troubleshooting

### "Server is not available"

- Make sure the Flux backend is running (`cd backend && ./run_server.sh`)
- Check that `FLUX_SERVER_PORT` in `.env` matches the backend port
- Verify with: `curl http://localhost:8000/health`

### "No UV coordinates" warning

- The app will automatically generate spherical UV mapping
- For better results, use models with pre-existing UV coordinates

### Texture generation is slow

- Normal generation time: 30-60 seconds
- Reduce texture size (512 instead of 1024)
- Reduce inference steps (30 instead of 50)

### Out of memory errors (backend)

- Reduce texture size
- Make sure backend has sufficient GPU memory (24GB+ recommended)

## Project Structure

```
frontend/
├── obj_texture_generator.py      # Main Streamlit app (UI only)
├── model_viewer.py                # Model viewer app with AI texture generation
├── glb_viewer.py                  # GLB viewer app
├── obj_viewer.py                  # OBJ viewer app
├── logic/                         # Business logic (separate from UI)
│   ├── obj_texture_generator_logic.py
│   └── model_viewer_logic.py      # Model viewer logic (AI texture generation)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Architecture

Following the project convention, the frontend is organized as:
- **GUI files** (`*_generator.py`, `*_viewer.py`): Pure UI code using Streamlit
- **Logic files** (`logic/*_logic.py`): Business logic, API calls, file operations

This separation allows for:
- Easier testing of business logic
- Cleaner UI code
- Better maintainability

## License

This project uses the FLUX.1-dev model which falls under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

