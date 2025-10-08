# Frontend Utils

Utility scripts for medical organ model processing and visualization.

## Available Utilities

### `pipeline.py` 🚀 RECOMMENDED
Complete automated pipeline that orchestrates all conversion steps from NIfTI files to 3D models with UV masks.

**Usage:**
```bash
# Run complete pipeline with defaults
python pipeline.py

# Custom input/output directories
python pipeline.py -i ./my_data/nifti -o ./my_output

# High quality settings
python pipeline.py --smoothing 20 --decimation 0.2 --mask-size 2048

# Skip certain steps (useful for re-runs)
python pipeline.py --skip-nifti --skip-obj --overwrite

# Only regenerate UV masks for existing models
python pipeline.py --skip-nifti --skip-obj --skip-glb --overwrite
```

**Options:**
- `-i, --input`: Input directory with `.nii.gz` files (default: input/nifti)
- `-o, --output`: Output base directory (default: output)
- `-t, --threshold`: Iso-surface threshold (default: 0.1)
- `-s, --smoothing`: Smoothing iterations (default: 10)
- `-d, --decimation`: Mesh decimation 0.0-1.0 (default: 0.5)
- `--close-boundaries`: Close mesh boundaries to make watertight
- `--mask-size`: UV mask size in pixels (default: 1024)
- `--mask-variants`: Create all UV mask variants
- `--uv-method`: UV mapping method - 'smart', 'xatlas', 'spherical', 'cylindrical' (default: smart)
- `--skip-nifti`: Skip NIfTI → OBJ conversion
- `--skip-uv-unwrap`: Skip UV unwrapping step
- `--skip-obj`: Skip OBJ → GLB conversion
- `--skip-glb`: Skip GLB → Model expansion
- `--skip-masks`: Skip UV mask creation
- `-v, --verbose`: Enable verbose output
- `--overwrite`: Overwrite existing files

**Pipeline Steps:**
1. **NIfTI → OBJ**: Convert medical images to 3D meshes
2. **UV Unwrap**: Add UV coordinates (xatlas or spherical mapping)
3. **OBJ → Models**: Convert directly to model directories (GLTF/GLB)
4. **UV Masks**: Generate masks for FLUX.1 texture generation
5. **Summary**: Display results and next steps

---

### `nifti2obj.py`
Convert NIfTI medical imaging files (`.nii.gz`) to 3D mesh objects (`.obj`).

**Usage:**
```bash
python nifti2obj.py -i ./input/nifti -o ./output/obj
```

**Options:**
- `-i, --input`: Directory with input `.nii.gz` files
- `-o, --output`: Directory to write resulting `.obj` files
- `-t, --threshold`: Iso-surface threshold for single-label files (default: 0.1)
- `-s, --smoothing`: Number of smoothing iterations (default: 10)
- `-d, --decimation`: Mesh decimation fraction 0.0 to 1.0 (default: 0.5)
- `--close-boundaries`: Close boundary holes to make meshes watertight
- `--hole-filling`: Method for filling holes: 'convex' or 'planar' (default: convex)
- `-v, --verbose`: Enable verbose output

---

### `obj2model.py`
Convert OBJ files directly to model directories with GLTF/GLB files. This streamlined script combines the functionality of the old obj2glb.py and glb2model.py for better performance and simplicity.

**Usage:**
```bash
# Process directory
python obj2model.py -i ./output/obj -o ./output/models

# Verbose output
python obj2model.py -i ./output/obj -o ./output/models -v
```

**Options:**
- `-i, --input`: Directory with input `.obj` files (default: ./output/obj)
- `-o, --output`: Directory for output model folders (default: ./output/models)
- `-v, --verbose`: Enable verbose output

**Output:** Each model directory contains:
- `scene.gltf` - GLTF text format with external binary
- `scene.glb` - GLB binary format (single file)
- `scene.bin` - Binary buffer data
- `textures/diffuse.png` - Placeholder texture

---

### `add_uv_unwrap.py` ⭐ NEW
Add UV coordinates to OBJ files that don't have them. Uses xatlas for optimal unwrapping with spherical mapping as fallback.

**Usage:**
```bash
# Process directory with smart method (recommended)
python add_uv_unwrap.py -i ./output/obj -o ./output/obj_unwrapped

# Process in-place (modify original files)
python add_uv_unwrap.py -i ./output/obj --in-place

# Use specific method
python add_uv_unwrap.py -i ./output/obj -m spherical

# Single file
python add_uv_unwrap.py -i ./model.obj -o ./model_unwrapped.obj
```

**Options:**
- `-i, --input`: Input OBJ file or directory (required)
- `-o, --output`: Output OBJ file or directory
- `-m, --method`: UV mapping method - 'smart', 'xatlas', 'spherical', 'cylindrical' (default: smart)
- `--in-place`: Modify files in place (overwrite originals)
- `--overwrite`: Overwrite existing output files
- `-v, --verbose`: Enable verbose output

**UV Mapping Methods:**
- **smart**: Tries xatlas first, falls back to spherical (recommended)
- **xatlas**: Optimal UV unwrapping (requires `pip install xatlas`)
- **spherical**: Spherical projection (good for organic shapes)
- **cylindrical**: Cylindrical projection (good for tubes/elongated objects)

---

### `create_uv_mask.py`
Create UV unwrap masks for FLUX.1 AI texture generation. These masks show where the model's UV coordinates are mapped in texture space, which guides AI texture generation to only paint on the relevant areas.

**Usage:**
```bash
# Process all models
python create_uv_mask.py

# Process specific organ
python create_uv_mask.py --organ colon

# Create all mask variants (binary, soft, filled)
python create_uv_mask.py --organ heart --variants

# Custom size and overwrite existing
python create_uv_mask.py --size 2048 --overwrite
```

**Options:**
- `--organ`: Specific organ to process (e.g., 'colon', 'heart')
- `--size`: Mask size in pixels (default: 1024)
- `--overwrite`: Overwrite existing masks
- `--variants`: Create all mask variants (binary, soft, filled)
- `--models-dir`: Directory containing model folders (default: output/models)

**Output:**
- `uv_mask.png`: Default filled mask for FLUX.1 (recommended)
- `uv_mask_binary.png`: Sparse point mask (only with --variants)
- `uv_mask_soft.png`: Soft gradient mask (only with --variants)
- `uv_mask_filled.png`: Filled region mask (only with --variants)

**Mask Types:**
- **Binary**: Sparse points showing exact UV coordinate locations
- **Soft**: Gradient mask with soft edges for smooth blending
- **Filled**: Solid regions covering all UV-mapped areas (best for FLUX.1)

---

### `generate_flux_texture.py` ⭐ NEW
Generate anatomically accurate textures for medical organ models using FLUX.1 AI. Uses UV masks and Vista3D prompts to create photo-realistic textures that respect the UV layout and only generate content in mapped areas.

**Usage:**
```bash
# Generate texture for colon model
python generate_flux_texture.py --organ colon

# High quality generation
python generate_flux_texture.py --organ heart --size 2048 --steps 100 --guidance 4.0

# With specific seed for reproducibility
python generate_flux_texture.py --organ liver --seed 42

# Without UV guidance (basic generation)
python generate_flux_texture.py --organ brain --no-uv-guidance

# Custom FLUX server
python generate_flux_texture.py --organ colon --server 192.168.1.100:8000
```

**Options:**
- `--organ`: Organ name (required, must match folder in models directory)
- `--models-dir`: Directory containing model folders (default: output/models)
- `--size`: Texture size - 512, 1024, or 2048 (default: 1024)
- `--guidance`: Guidance scale (default: 3.5, higher = more prompt adherence)
- `--steps`: Number of inference steps (default: 50, higher = better quality)
- `--seed`: Random seed for reproducibility (optional)
- `--output`: Output filename (default: flux_texture.png)
- `--no-mask`: Do not apply UV mask to final texture
- `--no-uv-guidance`: Disable UV-guided generation
- `--server`: FLUX server address (overrides .env, default: localhost:8000)

**How It Works:**
1. Loads the organ model's GLTF file and extracts UV coordinates
2. Loads the UV mask that shows where texture should be generated
3. Retrieves the appropriate anatomical prompt from frontend/conf/vista3d_prompts.json
4. Sends request to FLUX server with UV guidance
5. Applies UV mask to constrain generation to only mapped areas
6. Saves texture to model's textures/ folder

**Requirements:**
- FLUX server running (see backend/README.md)
- UV masks created (run create_uv_mask.py first)
- Vista3D prompts file (frontend/conf/vista3d_prompts.json)
- Frontend .env file with FLUX_SERVER configuration

**Output:**
- `textures/flux_texture.png`: Generated texture
- `textures/diffuse.png`: Copy as default texture (auto-applied in viewers)

## Typical Workflow

### Option A: Automated Pipeline (Recommended)

**One command to do everything:**
```bash
python pipeline.py
```

This runs the complete workflow:
1. NIfTI → OBJ (3D mesh conversion)
2. UV Unwrapping (add UV coordinates)
3. OBJ → Models (GLTF/GLB with textures)
4. UV mask generation (for FLUX.1)

**Then generate AI textures:**
```bash
# Make sure FLUX server is running first
python generate_flux_texture.py --organ colon
python generate_flux_texture.py --organ heart
```

### Option B: Manual Step-by-Step

If you need more control over individual steps:

1. **Convert medical images to 3D meshes:**
   ```bash
   python nifti2obj.py -i ./input/nifti -o ./output/obj
   ```

2. **Add UV coordinates to meshes:**
   ```bash
   python add_uv_unwrap.py -i ./output/obj --in-place
   ```

3. **Convert OBJ to model directories:**
   ```bash
   python obj2model.py -i ./output/obj -o ./output/models
   ```

4. **Create UV masks for AI texture generation:**
   ```bash
   python create_uv_mask.py
   ```

5. **Generate anatomically accurate textures with FLUX.1:**
   ```bash
   # Start FLUX server (in backend directory)
   cd ../../backend
   python flux_server.py
   
   # In another terminal, generate textures
   cd frontend/utils
   python generate_flux_texture.py --organ colon
   python generate_flux_texture.py --organ heart --size 2048 --steps 100
   ```

### Option C: Complete End-to-End with AI Textures

**Full workflow from medical images to textured 3D models:**
```bash
# 1. Process models (one-time setup)
python pipeline.py

# 2. Start FLUX server (keep running in background)
cd ../../backend
python flux_server.py &

# 3. Generate textures for all organs
cd ../frontend/utils
for organ in colon heart liver aorta; do
    python generate_flux_texture.py --organ $organ --size 1024 --steps 50
done

# 4. View the textured models
cd ..
python model_viewer.py
```

## Requirements

All utilities require:
- Python 3.8+
- Dependencies listed in `frontend/requirements.txt`

Install with:
```bash
cd frontend
pip install -r requirements.txt
```

## Directory Structure

```
frontend/utils/
├── README.md                 # This file
├── pipeline.py              # 🚀 Complete automated pipeline
├── nifti2obj.py             # NIfTI → OBJ converter
├── add_uv_unwrap.py         # Add UV coordinates to meshes
├── obj2model.py             # OBJ → Model directories (GLTF/GLB)
├── create_uv_mask.py        # UV mask generator for FLUX.1
└── generate_flux_texture.py # ⭐ AI texture generator using FLUX.1
```

## Output Structure

Models are organized in `output/models/<organ_name>/`:
```
output/models/colon/
├── scene.gltf            # GLTF scene file
├── scene.glb             # Binary GLB file
├── gltf_buffer_*.bin     # Binary buffers
├── textures/
│   ├── diffuse.png       # Main texture
│   └── uv_unwrap_debug.png
└── uv_mask.png          # UV mask for FLUX.1 ⭐
```

## Notes

- UV masks show where model geometry maps to texture space
- The filled mask type is recommended for FLUX.1 as it covers all relevant UV areas
- Some models may not have UV coordinates (you'll see a warning)
- Masks are grayscale images where white = UV-mapped area, black = empty space
- **AI Texture Generation**: The generate_flux_texture.py script uses UV masks to constrain FLUX.1 generation to only mapped areas
- **FLUX Server**: Must be running before texture generation (see backend/README.md)
- **Prompts**: Anatomical prompts are loaded from frontend/conf/vista3d_prompts.json
- **Quality vs Speed**: Higher steps (100+) and larger sizes (2048) produce better textures but take longer

