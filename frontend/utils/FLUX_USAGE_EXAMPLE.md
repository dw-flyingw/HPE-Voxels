# Using UV Masks with FLUX.1 for Texture Generation

This guide shows how to use the generated UV masks with FLUX.1 to create anatomically accurate textures for your 3D medical models.

## What is a UV Mask?

A UV mask is a grayscale image that shows where your 3D model's surface is mapped in 2D texture space. The black regions indicate where the model geometry exists, and white regions are empty space.

## Quick Start

### 1. Generate UV Masks

First, create UV masks for your models:

```bash
# Create masks for all models
cd /path/to/HPE-Voxels
python frontend/utils/create_uv_mask.py

# Or for a specific organ
python frontend/utils/create_uv_mask.py --organ heart
```

This creates `uv_mask.png` in each model directory (e.g., `output/models/heart/uv_mask.png`).

### 2. Using Masks with FLUX.1

The UV mask can be used in several ways with FLUX.1:

#### Option A: As a Conditioning Mask (Recommended)

Use the mask to tell FLUX.1 where to generate texture:

```python
from diffusers import FluxPipeline
from PIL import Image
import torch

# Load the FLUX.1 pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Load the UV mask
uv_mask = Image.open("output/models/heart/uv_mask.png")

# Generate texture with the mask as guidance
prompt = "realistic human heart muscle tissue, medical anatomy, detailed cardiac muscle fibers"

# Use the mask to ensure texture is only generated in the UV-mapped area
texture = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# Apply mask to generated texture (optional post-processing)
texture_array = np.array(texture)
mask_array = np.array(uv_mask) / 255.0  # Normalize to 0-1
mask_array = 1 - mask_array  # Invert mask (black=model, white=empty)

# Apply mask to each color channel
for i in range(3):
    texture_array[:, :, i] = texture_array[:, :, i] * mask_array

final_texture = Image.fromarray(texture_array.astype(np.uint8))
final_texture.save("output/models/heart/textures/flux_generated.png")
```

#### Option B: As an Inpainting Mask

Use FLUX.1's inpainting capabilities to generate texture only in masked regions:

```python
from diffusers import FluxInpaintPipeline
from PIL import Image

# Load inpainting pipeline
pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Load UV mask and create a blank base image
uv_mask = Image.open("output/models/heart/uv_mask.png")
base_image = Image.new("RGB", (1024, 1024), (128, 128, 128))

# Invert mask (FLUX expects white=area to inpaint)
from PIL import ImageOps
inpaint_mask = ImageOps.invert(uv_mask.convert('L'))

# Generate texture only in the masked area
prompt = "realistic human heart muscle tissue, anatomically accurate"
texture = pipe(
    prompt=prompt,
    image=base_image,
    mask_image=inpaint_mask,
    width=1024,
    height=1024,
    num_inference_steps=50
).images[0]

texture.save("output/models/heart/textures/flux_inpainted.png")
```

#### Option C: Using FLUX.1 API via Backend Server

If you're using the backend FLUX server:

```python
import requests
from PIL import Image
import io
import base64

# Read the UV mask
with open("output/models/heart/uv_mask.png", "rb") as f:
    mask_data = base64.b64encode(f.read()).decode()

# Call FLUX API with mask
response = requests.post(
    "http://localhost:8000/generate_texture",
    json={
        "prompt": "realistic human heart muscle tissue, medical anatomy",
        "organ_name": "heart",
        "width": 1024,
        "height": 1024,
        "uv_mask_base64": mask_data,
        "num_inference_steps": 50
    }
)

# Save generated texture
texture_data = response.json()["texture_base64"]
texture_bytes = base64.b64decode(texture_data)
texture = Image.open(io.BytesIO(texture_bytes))
texture.save("output/models/heart/textures/flux_generated.png")
```

## Mask Variants

The script can generate three types of masks (use `--variants` flag):

1. **`uv_mask.png`** (default, filled): Best for most use cases
   - Solid filled regions showing all UV-mapped areas
   - Recommended for FLUX.1 generation

2. **`uv_mask_binary.png`**: Sparse point mask
   - Shows exact UV coordinate locations
   - Good for debugging UV layouts

3. **`uv_mask_soft.png`**: Gradient mask
   - Soft edges for smooth blending
   - Good for artistic effects

4. **`uv_mask_filled.png`**: Same as default
   - Solid filled mask
   - Best for AI texture generation

## Tips for Best Results

1. **Use anatomically accurate prompts**: Be specific about the organ/tissue type
   - ✅ "realistic human heart cardiac muscle tissue with coronary vessels"
   - ❌ "red texture"

2. **Respect the UV layout**: The mask shows how your model is unwrapped
   - Generate textures that fit the mask shape
   - Consider the UV seams visible in the mask

3. **Post-process generated textures**:
   - Apply the mask to remove artifacts in empty areas
   - Adjust contrast and color balance
   - Add anatomical details

4. **Iterate and refine**:
   - Start with lower inference steps (20-30) for quick tests
   - Increase steps (50-100) for final high-quality textures
   - Try different prompts and seeds

## Directory Structure

After generating masks, your model directory will look like:

```
output/models/heart/
├── scene.gltf
├── scene.glb
├── gltf_buffer_*.bin
├── textures/
│   ├── diffuse.png            # Original texture
│   ├── flux_generated.png     # FLUX-generated texture
│   └── uv_unwrap_debug.png
└── uv_mask.png               # UV mask for FLUX.1 ⭐
```

## Troubleshooting

**Problem**: Generated texture doesn't align with model
- **Solution**: Ensure you're using the correct UV coordinates and mask
- Check that the mask shows the expected UV layout

**Problem**: Empty areas have artifacts
- **Solution**: Apply the mask as post-processing to zero out empty regions

**Problem**: Model shows black areas
- **Solution**: Some parts may have no UV coordinates. Check the mask coverage statistics.

**Problem**: Texture looks stretched or distorted
- **Solution**: This is due to UV unwrapping. Consider re-unwrapping the mesh or using the mask to guide where details should be placed.

## Advanced: Custom Mask Processing

You can create custom masks for specific effects:

```python
from PIL import Image, ImageFilter
import numpy as np

# Load the base mask
mask = Image.open("output/models/heart/uv_mask.png")
mask_array = np.array(mask)

# Example: Create a soft-edge mask
soft_mask = Image.fromarray(mask_array).filter(ImageFilter.GaussianBlur(radius=10))
soft_mask.save("output/models/heart/uv_mask_custom_soft.png")

# Example: Create an outline-only mask
from scipy import ndimage
edges = ndimage.sobel(mask_array)
edge_mask = Image.fromarray((edges * 255).astype(np.uint8))
edge_mask.save("output/models/heart/uv_mask_edges.png")
```

## Next Steps

1. Experiment with different FLUX.1 prompts for your organs
2. Try different mask variants for different effects
3. Combine multiple textures using the masks as layers
4. Use the masks to guide manual texture painting in tools like Substance Painter

For more information, see:
- FLUX.1 Documentation: https://github.com/black-forest-labs/flux
- UV Mapping Guide: https://en.wikipedia.org/wiki/UV_mapping
- Backend FLUX Server: `backend/flux_server.py`

