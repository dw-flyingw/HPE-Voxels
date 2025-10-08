# UV Unwrapping & Texture Fix Guide

This guide explains how to fix UV unwrapping and texture issues in your medical organ 3D models.

## Problem

You were experiencing three main issues:
1. **Incorrect UV unwrapping** - poor texture space utilization
2. **Inaccurate UV masks** - gaps and incorrect coverage
3. **Repeating textures** - patterns that repeat instead of being unique and seamless

## Solution

We've created three new improved tools that fix these issues:

### 1. `create_uv_mask_improved.py`
- Properly rasterizes UV triangles (not just points)
- Creates accurate RGBA masks with antialiasing
- Normalizes UVs to use full 0-1 texture space
- Provides detailed UV analysis and statistics

### 2. `add_uv_unwrap_improved.py`
- Uses xatlas with optimized settings
- Ensures maximum UV space utilization
- Validates UV coordinates
- Provides detailed unwrapping statistics

### 3. `fix_uv_texture_pipeline.py`
- All-in-one tool to fix existing models
- Re-unwraps with optimized settings
- Creates accurate UV masks
- Generates non-repeating textures with FLUX
- Updates all model files (OBJ, GLTF, GLB)

## Quick Start

### Fix a Single Model

```bash
cd /Users/dave/AI/HPE/HPE-Voxels

# Fix colon model (UVs + texture)
python frontend/utils/fix_uv_texture_pipeline.py --model colon

# Fix heart model (UVs only, skip texture)
python frontend/utils/fix_uv_texture_pipeline.py --model heart --skip-texture

# Fix liver with high-res texture
python frontend/utils/fix_uv_texture_pipeline.py --model liver --texture-size 2048
```

### Fix All Models

```bash
# Fix all models (requires FLUX server running)
python frontend/utils/fix_uv_texture_pipeline.py --all

# Fix UVs only for all models (faster, no texture generation)
python frontend/utils/fix_uv_texture_pipeline.py --all --skip-texture
```

## Step-by-Step Process

### What the Pipeline Does

**Step 1: Re-unwrap OBJ with xatlas**
- Loads original OBJ file
- Applies optimized xatlas unwrapping
- Normalizes UVs to 0-1 range with padding
- Minimizes UV seams and distortion

**Step 2: Create Accurate UV Mask**
- Rasterizes all UV triangles (not just points)
- Creates RGBA mask with 2x supersampling
- Ensures accurate coverage of UV-mapped areas
- Provides UV coverage statistics

**Step 3: Save Updated Files**
- Saves unwrapped OBJ (overwrites original)
- Exports GLTF and GLB with correct UVs
- Saves UV mask (RGBA and RGB versions)
- Fixes GLTF material for full brightness

**Step 4: Generate Non-Repeating Texture** (optional)
- Loads organ-specific prompt from vista3d_prompts.json
- Enhances prompt to prevent repeating patterns
- Uses FLUX with UV mask guidance
- Saves texture as diffuse.png and flux_texture.png

## Before & After

### Before (Old Method)
```
UV Range: U[0.002, 0.999] V[0.253, 0.963]  ← Only 71% V utilization
UV Mask: Point-based with gaps
Texture: Repeating patterns
```

### After (Improved Method)
```
UV Range: U[0.01, 0.99] V[0.01, 0.99]      ← Full 98% utilization
UV Mask: Triangle-rasterized, accurate
Texture: Unique, non-repeating, seamless
```

## Advanced Usage

### Using Individual Tools

#### 1. Re-unwrap OBJs Only

```bash
# Process all OBJs in-place
python frontend/utils/add_uv_unwrap_improved.py -i output/obj --in-place -v

# Process to new directory
python frontend/utils/add_uv_unwrap_improved.py -i output/obj -o output/obj_unwrapped
```

#### 2. Create UV Masks Only

```bash
# Create mask for specific model
python frontend/utils/create_uv_mask_improved.py --organ colon --size 2048

# Create masks for all models
python frontend/utils/create_uv_mask_improved.py --all --size 1024

# Create with variants (dilated, etc.)
python frontend/utils/create_uv_mask_improved.py --organ heart --variants --overwrite
```

## Output Files

After processing, each model directory will contain:

```
output/models/colon/
├── scene.gltf              # GLTF with corrected UVs
├── scene.glb               # GLB with corrected UVs
├── scene.bin               # Binary data
├── uv_mask.png             # RGB UV mask
├── uv_mask_rgba.png        # RGBA UV mask (with alpha channel)
├── uv_analysis.json        # UV statistics and analysis
├── uv_transform.json       # UV normalization info (if applied)
└── textures/
    ├── diffuse.png         # Generated texture
    └── flux_texture.png    # Same texture (FLUX output)
```

## UV Analysis

The tools provide detailed UV analysis:

```json
{
  "uv_min": [0.01, 0.01],
  "uv_max": [0.99, 0.99],
  "uv_range": [0.98, 0.98],
  "uvs_outside_01_percent": 0.0,
  "u_utilization_percent": 98.0,
  "v_utilization_percent": 98.0,
  "avg_utilization_percent": 98.0,
  "coverage_percent": 45.2,
  "total_uvs": 46434,
  "unique_uvs": 28156,
  "duplicate_uv_percent": 39.3
}
```

### What the Stats Mean

- **UV Range**: Bounds of UV coordinates
- **Utilization**: How much of 0-1 texture space is used
- **Coverage**: Percentage of texture pixels covered by mesh
- **UVs Outside 0-1**: Coordinates outside valid range (should be 0%)
- **Duplicate UVs**: Vertices sharing same UV (due to seams)

## Texture Generation

### Preventing Repeating Patterns

The improved texture generation adds these keywords to prompts:
- "unique non-repeating texture"
- "continuous organic variations"
- "seamless UV mapping"

These ensure FLUX generates a unique texture that doesn't have obvious repeating patterns.

### Custom Prompts

Edit `frontend/conf/vista3d_prompts.json` to customize prompts per organ:

```json
{
  "prompts": [
    {
      "name": "colon",
      "prompt": "hyper photo-realistic human colon mucosa surface, natural intestinal tissue, organic pink-beige color"
    }
  ],
  "default_template": "hyper photo-realistic human {structure} anatomical structure, medical photography"
}
```

## Troubleshooting

### "xatlas not available"

```bash
pip install xatlas
```

### "FLUX server not healthy"

Make sure the FLUX server is running:

```bash
cd backend
python flux_server.py
```

Or check your `.env` file for the correct `FLUX_SERVER` address.

### Low UV Coverage

If coverage is < 40%, the mesh may have issues:
- Check if original NIfTI file is clean
- Try different smoothing/decimation parameters
- Some meshes naturally have lower coverage due to geometry

### Seams Visible in Texture

- Increase texture size: `--texture-size 2048`
- The UV mask has built-in padding to help with seam bleeding
- Consider regenerating with different seed

## Integration with Existing Pipeline

You can integrate these tools into your existing pipeline:

### Option 1: Fix Existing Models

```bash
# After running normal pipeline, fix the results
python frontend/utils/pipeline.py  # Your existing pipeline
python frontend/utils/fix_uv_texture_pipeline.py --all  # Fix everything
```

### Option 2: Replace Pipeline Steps

Modify `frontend/utils/pipeline.py` to use the improved tools:
- Replace `add_uv_unwrap.py` calls with `add_uv_unwrap_improved.py`
- Replace `create_uv_mask.py` calls with `create_uv_mask_improved.py`

## Performance

### Processing Times (approximate)

Single model (e.g., colon with ~46K vertices):
- UV unwrapping: 5-10 seconds
- UV mask creation: 2-5 seconds
- GLTF/GLB export: 1-2 seconds
- Texture generation: 30-60 seconds (depends on FLUX server)

**Total per model**: ~1-2 minutes (with texture), ~10-20 seconds (without texture)

### Batch Processing

For multiple models, use `--all`:

```bash
# Fast (UVs only): ~20s × N models
python frontend/utils/fix_uv_texture_pipeline.py --all --skip-texture

# Full (UVs + textures): ~90s × N models  
python frontend/utils/fix_uv_texture_pipeline.py --all
```

## Best Practices

1. **Always backup your original files** before running in-place operations
2. **Start with `--skip-texture`** to verify UV unwrapping first
3. **Use `-v` or `--verbose`** for detailed output when debugging
4. **Check `uv_analysis.json`** after processing to verify quality
5. **Use `--texture-size 2048`** for high-quality models (if GPU allows)

## Validation Checklist

After processing a model, check:

- [ ] `uv_analysis.json` shows utilization > 85%
- [ ] `uv_analysis.json` shows uvs_outside_01_percent == 0
- [ ] `uv_mask_rgba.png` covers mesh completely
- [ ] `scene.gltf` opens correctly in viewer
- [ ] Texture has no obvious repeating patterns
- [ ] Texture aligns properly with mesh

## Getting Help

If you encounter issues:

1. Run with `-v` for verbose output
2. Check `uv_analysis.json` for diagnostic info
3. Verify all dependencies are installed
4. Make sure FLUX server is running (for texture generation)
5. Check that original OBJ file has valid geometry

## Summary

These improved tools fix the three main issues:

1. ✅ **Better UV unwrapping** - Full texture space utilization, minimal distortion
2. ✅ **Accurate UV masks** - Triangle rasterization, no gaps
3. ✅ **Non-repeating textures** - Enhanced prompts, better FLUX guidance

Use `fix_uv_texture_pipeline.py --all` to process all your models with the improved pipeline!

