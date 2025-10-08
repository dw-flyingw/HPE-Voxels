# HPE Voxels

A comprehensive toolkit for processing and visualizing 3D medical/anatomical models from NIfTI files, with support for texture generation and interactive viewing.

## Overview

This project provides a complete pipeline for converting NIfTI medical imaging files into interactive 3D models with realistic textures. The toolkit includes conversion utilities, texture generators, and web-based viewers for exploring anatomical structures.

## Features

- **NIfTI to 3D Model Pipeline**: Convert medical imaging data to various 3D formats (OBJ, GLB, GLTF)
- **Realistic Texture Generation**: Create anatomically accurate diffuse textures with vascular networks and tissue variations
- **Interactive 3D Viewers**: Web-based viewers for exploring models with proper lighting and materials
- **Organized Model Structure**: Consistent folder structure with proper texture organization
- **Color Mapping**: Automatic anatomical color assignment based on Vista3D label colors

## Project Structure

```
HPE-Voxels/
├── frontend/             # Web-based viewers (Streamlit apps)
│   ├── glb_viewer.py
│   ├── model_viewer.py
│   └── obj_viewer.py
├── backend/              # FLUX.1-dev API server
│   ├── flux_server.py
│   ├── run_server.sh
│   ├── setup_ubuntu.sh
│   └── requirements.txt
├── input/nifti/          # Input NIfTI files (.nii.gz)
├── output/
│   ├── obj/              # Intermediate OBJ files
│   ├── glb/              # GLB files for conversion
│   └── models/           # Final model folders with textures
│       └── {model_name}/
│           ├── scene.bin
│           ├── scene.glb
│           ├── scene.gltf
│           └── textures/
│               └── diffuse.png
├── images/               # Reference images for texture generation
└── vista3d_label_colors.json  # Anatomical color definitions
```

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

## Usage

### 1. Convert NIfTI to 3D Models

The main conversion pipeline processes NIfTI files through several stages:

```bash
# Convert NIfTI to OBJ files
python nifti2obj.py

# Convert OBJ to GLB files  
python obj2glb.py

# Convert GLB to organized model folders
python glb2model.py
```

### 2. Generate Textures

#### Placeholder Textures
Generate simple placeholder textures with anatomical colors:

```bash
# Generate placeholder textures for all models
python generate_placeholder_diffuse_textures.py

# Custom size and options
python generate_placeholder_diffuse_textures.py --size 1024 --overwrite
```

#### Realistic Heart Textures
Generate highly detailed textures for heart models:

```bash
# Generate realistic heart texture
python generate_heart_diffuse.py

# Custom output and size
python generate_heart_diffuse.py --size 2048 --output heart/textures/diffuse.png
```

### 3. View 3D Models

#### Model Viewer (Recommended)
Interactive viewer for all model types with texture support:

```bash
# Start the model viewer
python frontend/model_viewer.py

# Custom port
MODEL_VIEWER_PORT=8503 python frontend/model_viewer.py
```

#### GLB Viewer
Simple viewer specifically for GLB files:

```bash
# Start GLB viewer
python frontend/glb_viewer.py

# Custom port
GLB_VIEWER_PORT=8502 python frontend/glb_viewer.py
```

#### Object Viewer
Basic viewer for OBJ files:

```bash
python frontend/obj_viewer.py
```

## Model Folder Structure

Each model is organized in a consistent structure:

```
output/models/{model_name}/
├── scene.bin          # Binary mesh data
├── scene.glb          # Binary GLTF (complete model)
├── scene.gltf         # JSON GLTF with external references
└── textures/
    └── diffuse.png    # Diffuse texture map
```

This structure ensures:
- Compatibility with standard 3D viewers
- Proper texture loading and material support
- Easy integration with web-based viewers
- Consistent organization across all models

## Configuration

### Anatomical Colors
The `vista3d_label_colors.json` file contains color definitions for different anatomical structures. The texture generators use this file to assign appropriate colors to models based on their names.

### Environment Variables
- `MODEL_VIEWER_PORT`: Port for the model viewer (default: 8503)
- `GLB_VIEWER_PORT`: Port for the GLB viewer (default: 8502)

## Scripts Overview

### Processing Pipeline
| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `nifti2obj.py` | Convert NIfTI to OBJ | `.nii.gz` files | `.obj` files |
| `obj2glb.py` | Convert OBJ to GLB | `.obj` files | `.glb` files |
| `glb2model.py` | Organize GLB into model folders | `.glb` files | Model folders |
| `generate_placeholder_diffuse_textures.py` | Create placeholder textures | Model folders | `textures/diffuse.png` |
| `generate_heart_diffuse.py` | Create realistic heart textures | Heart models | `textures/diffuse.png` |

### Frontend (Web Viewers)
| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `frontend/model_viewer.py` | Interactive 3D model viewer | Model folders | Web interface |
| `frontend/glb_viewer.py` | GLB file viewer | `.glb` files | Web interface |
| `frontend/obj_viewer.py` | OBJ file viewer | `.obj` files | Web interface |

### Backend (API Server)
| Script | Purpose | Documentation |
|--------|---------|---------------|
| `backend/flux_server.py` | FLUX.1-dev text-to-image API | See `backend/README.md` |

## Quality Optimization

For detailed information on optimizing conversion quality for different use cases, see:
- **[Quality Guide](docs/QUALITY_GUIDE.md)** - Comprehensive guide to quality parameters and optimization

Quick quality presets:
- **Medical review**: `--threshold 0.05 --smoothing 5 --decimation 0.9`
- **Web/education**: `--threshold 0.1 --smoothing 10 --decimation 0.5` (default)
- **Mobile/AR**: `--threshold 0.2 --smoothing 15 --decimation 0.3`

## Dependencies

Key dependencies include:
- **Streamlit**: Web interface for viewers
- **Trimesh**: 3D mesh processing
- **Nibabel**: NIfTI file handling
- **Open3D**: 3D geometry processing
- **PIL/Pillow**: Image processing for textures
- **PyGLTFLib**: GLTF/GLB file handling
- **NumPy/SciPy**: Numerical computations
- **Scikit-image**: Image processing algorithms

## Contributing

When adding new features:
1. Follow the existing model folder structure
2. Ensure textures are placed in the `textures/` subfolder
3. Update GLTF references to use `textures/diffuse.png`
4. Test with the model viewer to ensure compatibility

## License

This project is part of HPE's medical imaging and visualization toolkit.
