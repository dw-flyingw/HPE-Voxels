# Frontend Utils

Utility scripts for medical organ model processing and visualization.

## Available Utilities

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

### `obj2glb.py`
Convert OBJ mesh files to GLB (GL Transmission Format Binary) for web viewing.

**Usage:**
```bash
python obj2glb.py -i ./output/obj -o ./output/glb
```

**Options:**
- `-i, --input`: Directory with input `.obj` files (default: ./output/obj)
- `-o, --output`: Directory to write resulting `.glb` files (default: ./output/glb)
- `-v, --verbose`: Enable verbose output

---

### `glb2model.py`
Convert GLB files to expanded GLTF model directories with separate buffer and texture files.

**Usage:**
```bash
python glb2model.py -i ./output/glb -o ./output/models
```

**Options:**
- `-i, --input`: Directory with input `.glb` files
- `-o, --output`: Directory to write model folders
- `-v, --verbose`: Enable verbose output

---

### `create_uv_mask.py` ⭐ NEW
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

### `create_masks.sh`
Convenience shell script to create UV masks.

**Usage:**
```bash
# Create masks for all models
./create_masks.sh

# Create mask for specific organ
./create_masks.sh --organ colon

# Create all variants
./create_masks.sh --variants --overwrite
```

## Typical Workflow

1. **Convert medical images to 3D meshes:**
   ```bash
   python nifti2obj.py -i ./input/nifti -o ./output/obj
   ```

2. **Convert OBJ to GLB for web viewing:**
   ```bash
   python obj2glb.py -i ./output/obj -o ./output/glb
   ```

3. **Expand GLB to model directories:**
   ```bash
   python glb2model.py -i ./output/glb -o ./output/models
   ```

4. **Create UV masks for AI texture generation:**
   ```bash
   python create_uv_mask.py
   ```

5. **Use UV masks with FLUX.1** to generate anatomically accurate textures:
   - Load the model's `uv_mask.png` as a reference
   - Generate textures that respect the UV layout
   - Apply generated textures to the model

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
├── README.md              # This file
├── nifti2obj.py          # NIfTI → OBJ converter
├── obj2glb.py            # OBJ → GLB converter  
├── glb2model.py          # GLB → GLTF model expander
├── create_uv_mask.py     # UV mask generator for FLUX.1
└── create_masks.sh       # Convenience script for mask creation
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

