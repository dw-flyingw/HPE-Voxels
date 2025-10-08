# FLUX Texture Generation Guide

This guide shows how to generate anatomically accurate textures for medical organ models using FLUX.1 AI.

## Prerequisites

1. **FLUX Server Running**
   ```bash
   # In backend directory
   cd backend
   python flux_server.py
   ```
   
2. **Models with UV Coordinates**
   ```bash
   # If not done yet, run the pipeline to create models with UV coordinates
   cd frontend/utils
   python pipeline.py
   ```

3. **UV Masks Created**
   ```bash
   # Create UV masks for all models
   python create_uv_mask.py
   ```

4. **Environment Configuration**
   - Ensure `frontend/.env` exists with `FLUX_SERVER=localhost:8000`
   - This was created automatically, but you can modify the server address if needed

## Quick Start

### Generate Single Texture

```bash
cd frontend/utils

# Generate texture for colon
python generate_flux_texture.py --organ colon

# Generate high-quality texture for heart
python generate_flux_texture.py --organ heart --size 2048 --steps 100 --guidance 4.0

# Generate with specific seed (reproducible)
python generate_flux_texture.py --organ liver --seed 42
```

### Generate All Textures

```bash
cd frontend/utils

# Interactive batch generation
./generate_all_textures.sh

# Or with custom settings via environment variables
SIZE=2048 STEPS=100 ./generate_all_textures.sh
```

## How It Works

The texture generation process:

1. **Loads Model Data**
   - Reads the GLTF file from `output/models/<organ>/scene.gltf`
   - Extracts UV coordinates to understand the texture mapping
   - Loads the UV mask from `output/models/<organ>/uv_mask.png`

2. **Retrieves Prompt**
   - Looks up the organ name in `frontend/conf/vista3d_prompts.json`
   - Uses anatomically accurate prompts like:
     ```
     "hyper photo-realistic human colon tissue surface, medical photography, 
      anatomically accurate colonic mucosa, rich deep pink-red coloration..."
     ```

3. **Sends to FLUX Server**
   - Encodes UV mask as base64
   - Sends request to FLUX server with prompt and UV guidance
   - Uses `/generate_with_control` endpoint for UV-guided generation

4. **Post-Processing**
   - Applies UV mask to constrain generation to only mapped areas
   - Ensures no texture outside the UV layout
   - Saves to `textures/flux_texture.png` and `textures/diffuse.png`

## Examples by Use Case

### Basic Generation (Default Quality)
```bash
python generate_flux_texture.py --organ colon
```
- Size: 1024x1024
- Steps: 50
- Guidance: 3.5
- Time: ~30-60 seconds

### High Quality (Publication/Presentation)
```bash
python generate_flux_texture.py --organ heart --size 2048 --steps 100 --guidance 4.0
```
- Size: 2048x2048
- Steps: 100
- Guidance: 4.0 (stronger prompt adherence)
- Time: ~2-3 minutes

### Fast Preview
```bash
python generate_flux_texture.py --organ liver --size 512 --steps 20 --guidance 3.0
```
- Size: 512x512
- Steps: 20
- Guidance: 3.0
- Time: ~10-15 seconds

### Reproducible Generation
```bash
# Same seed = same result
python generate_flux_texture.py --organ brain --seed 42 --steps 50
python generate_flux_texture.py --organ brain --seed 42 --steps 50  # Identical output
```

### Without UV Guidance
```bash
# Uses basic text-to-image generation (less accurate to UV layout)
python generate_flux_texture.py --organ kidney --no-uv-guidance
```

### Custom FLUX Server
```bash
# Use remote FLUX server
python generate_flux_texture.py --organ colon --server 192.168.1.100:8000
```

## Available Organs

Any organ with a model in `output/models/` and a corresponding prompt in `frontend/conf/vista3d_prompts.json`:

- aorta
- colon
- heart
- liver
- kidney (left/right)
- lung (all lobes)
- brain
- stomach
- bladder
- spleen
- pancreas
- gallbladder
- esophagus
- duodenum
- small bowel
- vertebrae (all levels)
- ribs (left/right)
- hip bones
- femur (left/right)
- And many more...

## Output Files

After generation, you'll find:

```
output/models/colon/
├── scene.gltf
├── scene.glb
├── textures/
│   ├── diffuse.png          # ← Generated texture (auto-applied in viewers)
│   └── flux_texture.png     # ← Copy of generated texture
└── uv_mask.png             # UV mask used for generation
```

## Viewing Results

Use the model viewer to see the textured model:

```bash
cd frontend
python model_viewer.py
```

Select the organ from the dropdown to view with the new AI-generated texture.

## Troubleshooting

### FLUX Server Not Running
```
✗ Cannot connect to FLUX server at http://localhost:8000
```
**Solution:** Start the FLUX server in the backend directory:
```bash
cd backend
python flux_server.py
```

### No UV Coordinates
```
⚠ No UV coordinates found - texture may not map correctly
```
**Solution:** Regenerate models with UV unwrapping:
```bash
cd frontend/utils
python add_uv_unwrap.py -i ../../output/obj --in-place
python obj2model.py -i ../../output/obj -o ../../output/models
```

### No Prompt Found
```
Using default template for organ: unknown_organ
```
**Solution:** Either:
- Use the exact organ name from `frontend/conf/vista3d_prompts.json`
- Add a new prompt to `frontend/conf/vista3d_prompts.json`

### UV Mask Not Found
```
⚠ Warning: UV mask not found
```
**Solution:** Generate UV masks:
```bash
python create_uv_mask.py --organ <organ_name>
```

## Tips for Best Results

1. **Use High Quality for Final Results**
   - Size: 2048
   - Steps: 100+
   - Guidance: 3.5-4.5

2. **Start with Defaults for Testing**
   - Test with default settings first
   - Adjust based on results

3. **Seed for Consistency**
   - Use `--seed` to regenerate exact same texture
   - Useful for comparing different settings

4. **Check UV Masks First**
   - View the UV mask to ensure good coverage
   - Regenerate masks if needed with larger size

5. **Batch Processing**
   - Use `generate_all_textures.sh` for multiple models
   - Set environment variables for custom settings

## Advanced Usage

### Custom Prompts

Edit `frontend/conf/vista3d_prompts.json` to customize prompts for specific organs:

```json
{
  "id": 62,
  "name": "colon",
  "prompt": "your custom prompt here..."
}
```

### Environment Variables

Create custom `.env` settings:

```bash
# frontend/.env
FLUX_SERVER=your.server.com:8000
```

### Integration with Pipeline

Add texture generation to your workflow:

```bash
# Complete pipeline with texture generation
python pipeline.py && \
python generate_flux_texture.py --organ colon && \
python model_viewer.py
```

## Performance Notes

- **GPU Required:** FLUX server needs GPU (H200 recommended)
- **Generation Time:** 
  - 512x512: ~10-15 seconds
  - 1024x1024: ~30-60 seconds
  - 2048x2048: ~2-3 minutes
- **Memory:** ~24GB VRAM for FLUX.1-dev
- **Network:** Keep FLUX server on local network for best performance

## Related Documentation

- [FLUX Usage Example](../../docs/FLUX_USAGE_EXAMPLE.md) - Detailed FLUX.1 guide
- [Vista3D Prompts](../../docs/VISTA3D_PROMPTS.md) - Anatomical prompt documentation
- [Utils README](README.md) - All utility scripts documentation
- [Backend README](../../backend/README.md) - FLUX server setup

## Support

For issues or questions:
1. Check logs from FLUX server terminal
2. Verify FLUX server health: `curl http://localhost:8000/health`
3. Check UV mask quality
4. Review prompt in `frontend/conf/vista3d_prompts.json`

