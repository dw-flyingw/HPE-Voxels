# Quality Guide: NIfTI to 3D Model Conversion Pipeline

This guide explains the quality implications of each step in the conversion pipeline and provides recommendations for optimal results based on your use case.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Quality Analysis by Step](#quality-analysis-by-step)
- [Parameter Reference](#parameter-reference)
- [Use Case Recommendations](#use-case-recommendations)
- [Best Practices](#best-practices)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)

## Pipeline Overview

The standard conversion pipeline consists of four stages:

```
NIfTI (.nii.gz) → OBJ → GLB → Model Folder
    [CRITICAL]    [MINIMAL]  [ORGANIZATIONAL]
```

### Quality Impact by Stage

| Stage | Quality Loss | Why | Reversible? |
|-------|--------------|-----|-------------|
| **NIfTI → OBJ** | 🔴 **High** | Marching cubes, decimation, smoothing | ❌ No |
| **OBJ → GLB** | 🟢 **Minimal** | Format conversion only | ✅ Yes |
| **GLB → Model** | 🟢 **None** | Organizational only | ✅ Yes |

> **Key Insight:** 95% of quality decisions happen in the first conversion step (NIfTI → OBJ).

## Quality Analysis by Step

### Step 1: NIfTI → OBJ (Critical Quality Stage)

This is where most quality loss occurs. The conversion uses:

#### 1.1 Marching Cubes Algorithm

**What it does:**
- Converts volumetric voxel data into surface triangles
- Fundamentally lossy transformation (3D volume → 2D surface)

**Quality factors:**
- Limited by input NIfTI resolution
- Cannot add detail that doesn't exist in source data
- Algorithm is industry-standard and well-optimized

**Current implementation:**
```python
from skimage import measure
verts, faces, normals, values = measure.marching_cubes(
    data, 
    level=threshold,  # Controls detail level
    spacing=spacing   # Preserves real-world dimensions
)
```

#### 1.2 Threshold Parameter

**Impact:** Determines isosurface detail level

| Value | Effect | Quality | Use Case |
|-------|--------|---------|----------|
| `0.05` | ⭐⭐⭐⭐⭐ High detail, noisy | Maximum detail | High-res scans, static viewing |
| `0.1` | ⭐⭐⭐⭐ Balanced | Good detail | **Default - Recommended** |
| `0.2` | ⭐⭐⭐ Smoother, less detail | Lower detail | Web/mobile viewing |
| `0.5+` | ⭐⭐ Very smooth, minimal detail | Minimal detail | Simple visualization only |

**Recommendation:** Start with `0.1`, adjust based on visual inspection.

#### 1.3 Smoothing Parameter

**Impact:** Reduces surface noise but loses fine details

| Iterations | Effect | Quality | File Size |
|------------|--------|---------|-----------|
| `0` | No smoothing, raw output | ⭐⭐⭐ Noisy | Same |
| `5` | Light smoothing | ⭐⭐⭐⭐⭐ Preserves detail | Same |
| `10` | **Default** - Balanced | ⭐⭐⭐⭐ Good | Same |
| `15` | Heavy smoothing | ⭐⭐⭐ Lost detail | Same |
| `20+` | Very smooth | ⭐⭐ Artificial looking | Same |

**Algorithm used:** Laplacian smoothing via Open3D
```python
o3d_mesh.filter_smooth_simple(number_of_iterations=smoothing)
```

**Recommendation:** Use `5-10` for medical models to preserve anatomical detail.

#### 1.4 Decimation Parameter

**Impact:** Reduces polygon count, MAJOR quality factor

| Value | Face Count | Quality | File Size | Use Case |
|-------|------------|---------|-----------|----------|
| `1.0` | 100% kept | ⭐⭐⭐⭐⭐ Maximum | ⬆️⬆️⬆️ Very large | Archival, high-detail analysis |
| `0.9` | 90% kept | ⭐⭐⭐⭐⭐ Excellent | ⬆️⬆️ Large | Medical review, AR/VR |
| `0.7` | 70% kept | ⭐⭐⭐⭐ Very good | ⬆️ Medium-large | Professional viewing |
| `0.5` | 50% kept | ⭐⭐⭐⭐ Good | ➡️ Medium | **Default - Web viewing** |
| `0.3` | 30% kept | ⭐⭐⭐ Acceptable | ⬇️ Small | Mobile, real-time apps |
| `0.1` | 10% kept | ⭐⭐ Poor | ⬇️⬇️ Very small | Low-poly preview only |

**Algorithm used:** Quadric Error Metric decimation
```python
mesh.simplify_quadric_decimation(face_count=target_faces)
```

**Example impact:**
- Original mesh: 200,000 faces
- Decimation 0.9: 180,000 faces (barely noticeable quality loss)
- Decimation 0.5: 100,000 faces (moderate quality loss)
- Decimation 0.3: 60,000 faces (visible quality loss)

#### 1.5 Hole Filling

**Impact:** Makes meshes watertight, can create artifacts

**Methods available:**
1. **Convex Hull** (`--hole-filling convex`)
   - Best for rounded surfaces
   - Can create bulges on complex holes
   - Default and recommended

2. **Planar** (`--hole-filling planar`)
   - Best for flat cuts
   - Simpler but less robust

**When to use:** 
- Enable `--close-boundaries` when working with clipped/cropped scans
- Necessary for some 3D printing applications
- May introduce artifacts on complex boundaries

### Step 2: OBJ → GLB (Minimal Quality Loss)

**What happens:**
- Geometry data repackaged into binary format
- Vertices, faces, normals preserved exactly
- More efficient storage and loading

**Quality loss:** ~0% for geometry data

**Benefits:**
- Faster loading (binary vs text)
- Single file includes all data
- Better compression
- Web-optimized format

**Note:** OBJ can lose some metadata that GLB preserves better.

### Step 3: GLB → Model Folder (No Quality Loss)

**What happens:**
- Extracts GLB contents into organized structure
- Creates placeholder textures if needed
- No geometry modification

**Quality loss:** 0%

**Purpose:** Organizational and compatibility
- Standardized folder structure
- Easier texture management
- Compatible with web viewers

## Parameter Reference

### NIfTI to OBJ Conversion

**Basic usage:**
```bash
python nifti2obj.py -i ./input/nifti -o ./output/obj [OPTIONS]
```

**All parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-i, --input` | path | Required | Directory with .nii.gz files |
| `-o, --output` | path | Required | Output directory for .obj files |
| `-t, --threshold` | float | `0.1` | Isosurface threshold (0.0-1.0) |
| `-s, --smoothing` | int | `10` | Smoothing iterations (0-50) |
| `-d, --decimation` | float | `0.5` | Mesh decimation ratio (0.0-1.0) |
| `--close-boundaries` | flag | `false` | Fill holes to make watertight |
| `--hole-filling` | choice | `convex` | Method: 'convex' or 'planar' |
| `-v, --verbose` | flag | `false` | Enable detailed output |

### OBJ to GLB Conversion

**Basic usage:**
```bash
python obj2glb.py -i ./output/obj -o ./output/glb [OPTIONS]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-i, --input` | path | `./output/obj` | Directory with .obj files |
| `-o, --output` | path | `./output/glb` | Output directory for .glb files |
| `-v, --verbose` | flag | `false` | Enable detailed output |

### GLB to Model Folder

**Basic usage:**
```bash
python glb2model.py
```

**Automatically processes:** `output/glb/*.glb` → `output/models/{name}/`

## Use Case Recommendations

### Medical Diagnosis & Review

**Goal:** Maximum accuracy and detail

```bash
python nifti2obj.py \
  -i ./input/nifti \
  -o ./output/obj \
  --threshold 0.05 \
  --smoothing 5 \
  --decimation 0.9 \
  --close-boundaries \
  --verbose
```

**Why:**
- Low threshold captures fine details
- Minimal smoothing preserves anatomical features
- High decimation retains most geometry
- Suitable for professional medical review

**File size:** Large (10-50 MB per model)

### Web Presentation & Education

**Goal:** Good quality with reasonable file sizes

```bash
python nifti2obj.py \
  -i ./input/nifti \
  -o ./output/obj \
  --threshold 0.1 \
  --smoothing 10 \
  --decimation 0.5 \
  --verbose
```

**Why:**
- Balanced detail and performance
- Loads quickly in web browsers
- Good visual quality for education
- **Current default settings**

**File size:** Medium (2-10 MB per model)

### Mobile & AR Applications

**Goal:** Small file size, real-time performance

```bash
python nifti2obj.py \
  -i ./input/nifti \
  -o ./output/obj \
  --threshold 0.2 \
  --smoothing 15 \
  --decimation 0.3 \
  --verbose
```

**Why:**
- Higher threshold reduces complexity
- More smoothing for cleaner appearance
- Low decimation for fast rendering
- Optimized for mobile GPUs

**File size:** Small (500 KB - 2 MB per model)

### 3D Printing

**Goal:** Watertight mesh with good surface quality

```bash
python nifti2obj.py \
  -i ./input/nifti \
  -o ./output/obj \
  --threshold 0.1 \
  --smoothing 8 \
  --decimation 0.7 \
  --close-boundaries \
  --hole-filling convex \
  --verbose
```

**Why:**
- Watertight requirement for slicing software
- Moderate smoothing for printable surface
- Higher decimation for detail
- Hole filling essential

**File size:** Medium-large (5-20 MB per model)

### Quick Preview/Sketching

**Goal:** Fast processing, rough visualization

```bash
python nifti2obj.py \
  -i ./input/nifti \
  -o ./output/obj \
  --threshold 0.3 \
  --smoothing 20 \
  --decimation 0.2 \
  --verbose
```

**Why:**
- Fast processing time
- Small file sizes
- Good enough for layout/composition
- Not for final presentation

**File size:** Very small (<500 KB per model)

## Best Practices

### 1. Understand Your Input Data

**Before optimizing parameters, check:**

```bash
# Inspect NIfTI file properties
python -c "
import nibabel as nib
img = nib.load('input/nifti/heart.nii.gz')
print(f'Shape: {img.shape}')
print(f'Voxel size: {img.header.get_zooms()}')
print(f'Data type: {img.get_data_dtype()}')
"
```

**Key metrics:**
- **Voxel resolution:** Higher = better input quality
- **Volume dimensions:** Larger = more detail potential
- **Scan quality:** Clean segmentation = better output

### 2. Iterative Quality Tuning

**Recommended workflow:**

1. **Start with fast preview:**
   ```bash
   python nifti2obj.py -i ./input/nifti -o ./output/obj \
     --decimation 0.2 --smoothing 15 -v
   ```

2. **Review in viewer:**
   ```bash
   python frontend/obj_viewer.py
   ```

3. **Adjust parameters based on results:**
   - Too noisy? → Increase `--smoothing`
   - Too smooth? → Decrease `--smoothing`
   - Need more detail? → Lower `--threshold`, higher `--decimation`
   - File too large? → Increase `--threshold`, lower `--decimation`

4. **Generate final quality version:**
   ```bash
   python nifti2obj.py -i ./input/nifti -o ./output/obj \
     --threshold 0.1 --decimation 0.7 --smoothing 8 -v
   ```

### 3. Quality vs File Size Tradeoffs

**General guidelines:**

| Priority | Threshold | Smoothing | Decimation | Expected Size |
|----------|-----------|-----------|------------|---------------|
| **Quality First** | 0.05 | 5 | 0.9 | 10-50 MB |
| **Balanced** | 0.1 | 10 | 0.5 | 2-10 MB |
| **Size First** | 0.2 | 15 | 0.3 | 500 KB - 2 MB |

### 4. Batch Processing Different Quality Levels

**Create a quality comparison:**

```bash
# High quality
python nifti2obj.py -i ./input/nifti -o ./output/obj_high \
  --threshold 0.05 --smoothing 5 --decimation 0.9 -v

# Medium quality (default)
python nifti2obj.py -i ./input/nifti -o ./output/obj_medium \
  --threshold 0.1 --smoothing 10 --decimation 0.5 -v

# Low quality
python nifti2obj.py -i ./input/nifti -o ./output/obj_low \
  --threshold 0.2 --smoothing 15 --decimation 0.3 -v
```

Compare results to find optimal settings for your use case.

### 5. Preserve Original Data

**Always keep your original NIfTI files!**

- All conversions are lossy
- You cannot recover lost detail
- You may need to re-process with different settings
- Original scans are your source of truth

### 6. Multi-Label Files

**Special considerations for `all.nii.gz` files:**

- Contains multiple anatomical structures
- Each label processed separately
- Uses `frontend/conf/vista3d_label_colors.json` for coloring
- Parameters apply to ALL labels

```bash
# Process multi-label file with balanced settings
python nifti2obj.py -i ./input/nifti -o ./output/obj \
  --threshold 0.5 \      # Multi-label files use different threshold
  --smoothing 10 \
  --decimation 0.5 \
  -v
```

## Advanced Techniques

### 1. Custom Per-Organ Settings

Different anatomical structures may need different settings:

**Complex structures (heart, brain):**
- Lower threshold (0.05-0.1)
- Less smoothing (5-8)
- Higher decimation (0.7-0.9)

**Simple structures (bones):**
- Higher threshold (0.15-0.2)
- More smoothing (10-15)
- Lower decimation (0.4-0.6)

**Vascular structures (arteries):**
- Very low threshold (0.03-0.08)
- Minimal smoothing (3-5)
- High decimation (0.8-0.95)

### 2. Alternative Meshing Algorithms

While the current implementation uses marching cubes (excellent for most cases), other algorithms exist:

**Marching Cubes** (current):
- ✅ Industry standard
- ✅ Good for general use
- ✅ Well-tested implementation
- ❌ Can create triangular artifacts

**Dual Contouring:**
- ✅ Better for sharp features
- ✅ Preserves edges well
- ❌ More complex to implement
- ❌ Not included in scikit-image

**Flying Edges:**
- ✅ Faster than marching cubes
- ✅ Better threading support
- ❌ Requires VTK library
- ❌ Not included by default

**Surface Nets:**
- ✅ Smoother results
- ✅ Good for organic shapes
- ❌ May lose fine details
- ❌ Not widely available

**Current implementation is optimal for medical imaging.**

### 3. Volume Rendering vs Surface Meshes

**When to use volume rendering instead:**

- Very high-resolution scans (>512³ voxels)
- Need to show internal structures
- Transparency/density visualization needed
- Interactive cross-sections required

**When to use surface meshes (current approach):**
- Web deployment (better performance)
- 3D printing
- AR/VR applications
- External surface visualization
- **Recommended for most use cases**

### 4. Post-Processing Techniques

**After mesh generation, consider:**

**Texture generation:**
```bash
# Generate realistic textures
python generate_heart_diffuse.py

# Or simple placeholders
python generate_placeholder_diffuse_textures.py
```

**Mesh repair:**
- Check for non-manifold geometry
- Fix flipped normals
- Remove duplicate vertices
- Fill remaining holes

**UV unwrapping:**
- Required for complex texturing
- Can be done in Blender or programmatically
- See `generate_uv_unwrapped_textures.py`

## Troubleshooting

### Problem: Mesh looks noisy/bumpy

**Causes:**
- Low input resolution
- Threshold too low
- Not enough smoothing

**Solutions:**
```bash
# Increase smoothing
python nifti2obj.py -i ./input -o ./output --smoothing 15

# Raise threshold slightly
python nifti2obj.py -i ./input -o ./output --threshold 0.15
```

### Problem: Lost too much detail

**Causes:**
- Threshold too high
- Too much smoothing
- Over-decimation

**Solutions:**
```bash
# Lower threshold, less smoothing, less decimation
python nifti2obj.py -i ./input -o ./output \
  --threshold 0.05 \
  --smoothing 5 \
  --decimation 0.8
```

### Problem: Files too large for web

**Causes:**
- High decimation ratio
- Low threshold
- Large input volume

**Solutions:**
```bash
# Reduce decimation, increase threshold
python nifti2obj.py -i ./input -o ./output \
  --threshold 0.15 \
  --decimation 0.3

# Or use compression after conversion
# GLB files are already compressed
```

### Problem: Holes in mesh

**Causes:**
- Clipped/cropped scan data
- Incomplete segmentation
- Marching cubes at boundary

**Solutions:**
```bash
# Enable boundary closing
python nifti2obj.py -i ./input -o ./output \
  --close-boundaries \
  --hole-filling convex
```

### Problem: Mesh not watertight (for 3D printing)

**Causes:**
- Open boundaries
- Non-manifold edges
- Incomplete surface

**Solutions:**
```bash
# Use hole filling
python nifti2obj.py -i ./input -o ./output \
  --close-boundaries \
  --hole-filling convex

# Check result
python -c "
import trimesh
mesh = trimesh.load('output/obj/model.obj')
print(f'Watertight: {mesh.is_watertight}')
print(f'Volume: {mesh.volume}')
"
```

### Problem: Conversion fails with memory error

**Causes:**
- Very large input volume
- High decimation ratio
- Insufficient RAM

**Solutions:**
```bash
# Process with lower quality settings first
python nifti2obj.py -i ./input -o ./output \
  --decimation 0.3 \
  --threshold 0.2

# Or crop the NIfTI file before processing
# Or process on a machine with more RAM
```

### Problem: Colors not applied correctly

**Causes:**
- Missing `frontend/conf/vista3d_label_colors.json`
- Name mismatch between file and JSON
- Incorrect label IDs

**Solutions:**
```bash
# Verify color map exists
ls frontend/conf/vista3d_label_colors.json

# Check for name matches
cat frontend/conf/vista3d_label_colors.json | grep -i "heart"

# Enable verbose mode to see matching
python nifti2obj.py -i ./input -o ./output -v
```

## Quality Factors Summary

### Input Quality (Most Important)

**You cannot create detail that doesn't exist in the source data.**

| Factor | Impact | Your Control |
|--------|--------|--------------|
| NIfTI scan resolution | ⭐⭐⭐⭐⭐ Critical | ❌ Fixed (depends on scanner) |
| Segmentation quality | ⭐⭐⭐⭐⭐ Critical | 🟡 Depends on segmentation method |
| Scan noise level | ⭐⭐⭐⭐ High | ❌ Fixed (can smooth) |

### Conversion Parameters (You Control)

| Parameter | Impact | Adjustability |
|-----------|--------|---------------|
| Decimation ratio | ⭐⭐⭐⭐⭐ Critical | ✅ Fully adjustable |
| Threshold | ⭐⭐⭐⭐ High | ✅ Fully adjustable |
| Smoothing iterations | ⭐⭐⭐ Medium | ✅ Fully adjustable |
| Hole filling | ⭐⭐ Low | ✅ Optional |

### Viewing Context

**Quality requirements vary by use case:**

- **Medical diagnosis:** Need 90-100% quality
- **Education/presentation:** 50-70% quality sufficient
- **Mobile/AR:** 30-50% quality acceptable
- **Quick preview:** 20-30% quality fine

## Conclusion

**Key Takeaways:**

1. ✅ **Your pipeline is well-designed** - Uses industry-standard algorithms
2. ✅ **First conversion is critical** - 95% of quality decisions happen at NIfTI → OBJ
3. ✅ **Parameters are flexible** - Adjust based on use case
4. ✅ **Input quality matters most** - Better scans = better output
5. ✅ **Iterate and compare** - Test different settings to find optimal balance

**Quick Decision Matrix:**

| If you need... | Use these settings |
|----------------|-------------------|
| **Best possible quality** | `--threshold 0.05 --smoothing 5 --decimation 0.9` |
| **Balanced (recommended)** | `--threshold 0.1 --smoothing 10 --decimation 0.5` |
| **Smallest files** | `--threshold 0.2 --smoothing 15 --decimation 0.3` |
| **3D printing** | Add `--close-boundaries --hole-filling convex` |
| **Web deployment** | Use balanced settings, GLB format |
| **Mobile/AR** | Use smallest file settings |

**Remember:** You can always re-process from the original NIfTI files with different settings. Keep your source data!

---

For more information, see:
- [README.md](../README.md) - General project overview
- [USAGE.md](../frontend/USAGE.md) - Viewer usage instructions
- [vista3d_label_colors.json](../frontend/conf/vista3d_label_colors.json) - Anatomical color definitions
- [vista3d_prompts.json](../frontend/conf/vista3d_prompts.json) - Texture generation prompts

