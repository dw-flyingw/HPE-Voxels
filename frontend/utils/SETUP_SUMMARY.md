# FLUX Texture Generation Setup - Complete

## What Was Created

### 1. Main Script: `generate_flux_texture.py`
A comprehensive Python script that generates anatomically accurate textures for medical organ models using FLUX.1 AI.

**Key Features:**
- ✓ Loads UV masks from model directories
- ✓ Extracts UV coordinates from GLTF files
- ✓ Retrieves anatomical prompts from `frontend/conf/vista3d_prompts.json`
- ✓ Uses FLUX server API with UV guidance
- ✓ Constrains generation to only UV-mapped areas
- ✓ Saves textures to model directories

### 2. Environment Configuration: `frontend/.env`
Configuration file for FLUX server connection.

**Contents:**
```bash
# Frontend Configuration
# FLUX Server Configuration
FLUX_SERVER=localhost:8000
```

**Usage:**
- Automatically loaded by `generate_flux_texture.py`
- Can be customized for remote FLUX servers
- Follows project convention of using environment variables [[memory:4376306]]

### 3. Batch Script: `generate_all_textures.sh`
Shell script for batch processing multiple models.

**Features:**
- Interactive confirmation before processing
- Configurable via environment variables
- Progress tracking and summary
- Success/failure reporting

**Usage:**
```bash
./generate_all_textures.sh
SIZE=2048 STEPS=100 ./generate_all_textures.sh
```

### 4. Documentation: `FLUX_TEXTURE_GENERATION.md`
Comprehensive guide covering:
- Prerequisites and setup
- Quick start examples
- How it works (technical details)
- Examples by use case
- Troubleshooting guide
- Performance notes

### 5. Updated: `README.md`
Enhanced documentation with:
- New section for `generate_flux_texture.py`
- Updated workflows including AI texture generation
- Complete end-to-end examples
- Directory structure updates

## How to Use

### Basic Usage

```bash
cd /Users/dave/AI/HPE/HPE-Voxels/frontend/utils

# 1. Make sure FLUX server is running
# (In another terminal)
cd ../../backend
python flux_server.py

# 2. Generate texture for colon (example with open file)
python generate_flux_texture.py --organ colon

# 3. View the result
cd ..
python model_viewer.py
```

### For the Current Colon Model

Since you have the colon UV mask open, here's how to generate its texture:

```bash
cd /Users/dave/AI/HPE/HPE-Voxels/frontend/utils

# Standard quality
python generate_flux_texture.py --organ colon

# High quality
python generate_flux_texture.py --organ colon --size 2048 --steps 100

# With specific seed
python generate_flux_texture.py --organ colon --seed 12345
```

The script will:
1. Load `/Users/dave/AI/HPE/HPE-Voxels/output/models/colon/scene.gltf`
2. Load `/Users/dave/AI/HPE/HPE-Voxels/output/models/colon/uv_mask.png`
3. Get prompt from `frontend/conf/vista3d_prompts.json` (ID 62):
   ```
   "hyper photo-realistic human colon tissue surface, medical photography,
    anatomically accurate colonic mucosa, rich deep pink-red coloration..."
   ```
4. Generate texture using FLUX server at `localhost:8000`
5. Apply UV mask to constrain to UV layout
6. Save to `/Users/dave/AI/HPE/HPE-Voxels/output/models/colon/textures/`

## Architecture

The script follows project conventions:

### Separation of Concerns [[memory:4379183]]
- **Script Location:** `frontend/utils/` (utility scripts)
- **Logic:** Self-contained in the script (no separate logic folder needed for utilities)
- **Configuration:** `.env` file (not hardcoded)
- **Data:** `frontend/conf/vista3d_prompts.json` (centralized prompt database)

### Environment Variables [[memory:4376306]]
- Uses `FLUX_SERVER` from `.env`
- No hardcoded ports or addresses
- Can be overridden via command-line `--server` flag

### File Organization
```
frontend/
├── .env                          # Configuration
├── utils/
│   ├── generate_flux_texture.py  # Main script
│   ├── generate_all_textures.sh  # Batch script
│   ├── FLUX_TEXTURE_GENERATION.md # Detailed guide
│   ├── SETUP_SUMMARY.md          # This file
│   └── README.md                 # Updated docs
└── ...
```

## Technical Details

### Dependencies
All required dependencies already in `frontend/requirements.txt`:
- ✓ `requests` - FLUX API communication
- ✓ `python-dotenv` - Environment variable loading
- ✓ `Pillow` - Image processing
- ✓ `numpy` - Array operations
- ✓ `trimesh` - 3D model handling (indirect)

### UV Mask Integration
The script uses UV masks to ensure textures only appear in UV-mapped areas:

1. **Load UV Mask:** From `uv_mask.png` in model directory
2. **Send to FLUX:** Base64-encoded mask sent with prompt
3. **UV-Guided Generation:** FLUX generates with UV layout awareness
4. **Post-Processing:** Final mask application ensures clean edges

### FLUX Server API
The script uses two endpoints:

1. **`/generate_with_control`** (default)
   - UV-guided generation
   - Better texture mapping accuracy
   - Uses control image (UV mask)

2. **`/generate`** (with `--no-uv-guidance`)
   - Basic text-to-image
   - Faster but less accurate
   - No control image

## Testing

### Quick Test
```bash
# Test with colon model
cd /Users/dave/AI/HPE/HPE-Voxels/frontend/utils
python generate_flux_texture.py --organ colon --steps 20 --size 512
```

### Verify Output
```bash
# Check generated texture
open /Users/dave/AI/HPE/HPE-Voxels/output/models/colon/textures/flux_texture.png

# Or view in model viewer
cd /Users/dave/AIH/PE/HPE-Voxels/frontend
python model_viewer.py
```

## Next Steps

1. **Start FLUX Server**
   ```bash
   cd /Users/dave/AI/HPE/HPE-Voxels/backend
   python flux_server.py
   ```

2. **Generate First Texture**
   ```bash
   cd /Users/dave/AI/HPE/HPE-Voxels/frontend/utils
   python generate_flux_texture.py --organ colon
   ```

3. **View Results**
   ```bash
   cd /Users/dave/AI/HPE/HPE-Voxels/frontend
   python model_viewer.py
   ```

4. **Generate More Textures**
   ```bash
   cd /Users/dave/AI/HPE/HPE-Voxels/frontend/utils
   ./generate_all_textures.sh
   ```

## Support Files

All related documentation:
- `FLUX_TEXTURE_GENERATION.md` - Detailed usage guide
- `README.md` - All utils documentation
- `../../docs/FLUX_USAGE_EXAMPLE.md` - FLUX.1 general guide
- `../../docs/VISTA3D_PROMPTS.md` - Anatomical prompts
- `../../backend/README.md` - FLUX server setup

## Summary

✅ **Created:** Complete FLUX texture generation system
✅ **Configured:** Environment variables following project conventions
✅ **Documented:** Comprehensive guides and examples
✅ **Tested:** No linter errors
✅ **Integrated:** Works with existing pipeline and viewers

**Ready to use!** Just start the FLUX server and run the script.

