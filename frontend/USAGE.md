# OBJ Texture Generator - Usage Guide

## Overview

The OBJ Texture Generator is a Streamlit web application that allows you to upload 3D models in OBJ format and generate hyper-realistic textures using the Flux AI model. It's designed specifically for creating photorealistic medical organ textures while preserving the original model's volume.

## Quick Start

```bash
# Terminal 1: Start Flux backend
cd backend
./run_server.sh

# Terminal 2: Start frontend
cd frontend
./run_texture_generator.sh
```

Then open http://localhost:8501 in your browser.

## Features

### ✨ Key Capabilities

- **Upload OBJ Files**: Support for any `.obj` file format
- **AI-Powered Texture Generation**: Uses FLUX.1-dev for photorealistic textures
- **Automatic UV Mapping**: Generates UV coordinates if not present
- **Adjustable Parameters**: Control quality, speed, and style
- **Material Export**: Generates OBJ, MTL, and PNG texture files
- **Example Prompts**: Pre-configured prompts for common organs

### 🎨 Supported Use Cases

1. **Medical Visualization**: Generate realistic organ textures for medical education
2. **3D Modeling**: Add photorealistic textures to anatomical models
3. **Research**: Create consistent, high-quality textures for studies
4. **Prototyping**: Quickly test different texture styles

## User Interface

### Left Panel: Upload Model

1. **File Upload**
   - Click "Choose an OBJ file"
   - Select any `.obj` file (e.g., `colon.obj`, `heart.obj`)
   - Supported from `output/obj/` folder or custom files

2. **Model Information**
   - Displays vertex and face counts
   - Shows file size
   - Indicates if UV coordinates exist
   - Warns if UV mapping will be generated

### Right Panel: Generate Texture

1. **Texture Prompt**
   - Text area for describing desired texture
   - Default: "hyper photo realistic colon tissue..."
   - Supports any descriptive text

2. **Example Prompts** (Expandable)
   - Realistic Colon
   - Realistic Heart
   - Realistic Liver
   - Click to auto-fill the prompt

3. **Generate Button**
   - Validates inputs
   - Sends request to Flux server
   - Shows progress spinner
   - Applies texture to model

### Sidebar: Configuration

1. **Server Status**
   - Displays Flux server URL
   - "Check Server Status" button
   - Shows server health information

2. **Generation Settings**
   - **Texture Size**: 512, 1024, or 2048 pixels
   - **Guidance Scale**: 1.0 - 10.0 (how closely to follow prompt)
   - **Inference Steps**: 20 - 100 (quality vs speed)
   - **Seed**: -1 for random, or specific number for reproducibility

### Results Section

After successful generation:

1. **Generated Texture**
   - Preview of the texture image
   - High-resolution display

2. **Model Preview**
   - Placeholder for 3D viewer (future update)

3. **Download Files**
   - Download OBJ: The model with texture references
   - Download Texture: The PNG texture image
   - Download MTL: Material definition file

## Generation Settings Explained

### Texture Size

| Size | Quality | Speed | Use Case |
|------|---------|-------|----------|
| 512 | Low | Fast (20-30s) | Quick previews, testing prompts |
| 1024 | High | Medium (30-60s) | **Recommended for most uses** |
| 2048 | Very High | Slow (60-120s) | Final production, print quality |

### Guidance Scale

| Range | Effect | Best For |
|-------|--------|----------|
| 1.0 - 2.0 | More creative, less literal | Abstract textures |
| 3.0 - 5.0 | **Balanced** | **Medical/realistic textures** |
| 6.0 - 10.0 | Very literal, strict adherence | Exact specifications |

### Inference Steps

| Range | Quality | Time |
|-------|---------|------|
| 20 - 30 | Good | Fast |
| 40 - 60 | **Excellent** | **Medium** |
| 70 - 100 | Best | Slow |

### Seed

- **-1**: Random (different result each time)
- **0-999999**: Fixed seed (reproducible results)

## Prompt Engineering Guide

### Anatomy of a Good Prompt

```
[realism level] + [subject] + [specific features] + [quality indicators]
```

### Examples by Organ

#### Colon
```
hyper photo realistic colon tissue with haustra and natural surface details, medical grade quality, 8k resolution
```

**Key elements:**
- "haustra" = characteristic colon structure
- "medical grade" = clinical accuracy
- "8k resolution" = high detail

#### Heart
```
hyper photo realistic cardiac muscle tissue with blood vessels, medical photography, 8k resolution
```

**Key elements:**
- "cardiac muscle tissue" = specific tissue type
- "blood vessels" = anatomical detail
- "medical photography" = professional quality

#### Liver
```
hyper photo realistic liver tissue with smooth surface and blood vessels, medical grade, 8k resolution
```

**Key elements:**
- "smooth surface" = liver's characteristic appearance
- "blood vessels" = vascular detail

#### Aorta/Arteries
```
hyper photo realistic arterial wall tissue with smooth surface, medical imaging quality, 8k resolution
```

#### Bones/Hip
```
hyper photo realistic bone surface with natural texture and porous structure, medical CT scan quality, 8k resolution
```

### Prompt Best Practices

✅ **DO:**
- Start with "hyper photo realistic"
- Include "medical grade" or "medical photography"
- Mention specific anatomical features
- Add "8k resolution" or "4k resolution"
- Keep prompts focused and clear

❌ **DON'T:**
- Use vague terms like "nice" or "good"
- Include multiple unrelated elements
- Add artistic styles for medical models
- Use negative prompts (not supported)
- Make prompts too long (>100 words)

### Advanced Prompting

**For more vascular appearance:**
```
hyper photo realistic colon tissue with prominent blood vessels and capillaries, medical angiography, 8k resolution
```

**For specific coloration:**
```
hyper photo realistic liver tissue with natural reddish-brown color and subtle variations, medical photography, 8k resolution
```

**For specific surface features:**
```
hyper photo realistic heart tissue with visible muscle fiber striations and natural sheen, medical microscopy, 8k resolution
```

## Workflow Examples

### Example 1: Basic Colon Texture

1. Start servers (backend + frontend)
2. Upload `output/obj/colon.obj`
3. Use default prompt or click "Realistic Colon" example
4. Keep default settings (1024, 3.5 guidance, 50 steps)
5. Click "Generate Texture"
6. Wait ~45 seconds
7. Download all three files

### Example 2: High-Quality Heart

1. Upload `output/obj/heart.obj`
2. Click "Realistic Heart" example
3. Increase texture size to 2048
4. Increase steps to 75
5. Set guidance to 4.0
6. Click "Generate Texture"
7. Wait ~90 seconds
8. Download files

### Example 3: Testing Different Styles

1. Upload any OBJ file
2. Generate with prompt A (seed: 12345)
3. Note the result
4. Generate with prompt B (seed: 12345)
5. Compare results
6. Use different seed (67890) to get variation

## File Output Explained

### Generated Files

1. **textured_model.obj**
   - Original model geometry
   - UV coordinates (generated if needed)
   - References to MTL file
   - Can be opened in any 3D software

2. **textured_model.mtl**
   - Material definition
   - Texture file reference
   - Surface properties (Ka, Kd, Ks)
   - Requires texture.png to be in same folder

3. **texture.png**
   - Generated texture image
   - Size: 512, 1024, or 2048 pixels
   - PNG format (lossless)
   - Can be edited in Photoshop/GIMP

### Using Generated Files

#### In Blender

1. File → Import → Wavefront (.obj)
2. Select `textured_model.obj`
3. ✅ Import MTL
4. Ensure `texture.png` is in same folder
5. Switch to Material Preview mode
6. Texture should appear automatically

#### In MeshLab

1. File → Import Mesh
2. Select `textured_model.obj`
3. Render → Render Mode → Textured
4. View the textured model

#### In Three.js / Web

```javascript
const loader = new OBJLoader();
const mtlLoader = new MTLLoader();

mtlLoader.load('textured_model.mtl', (materials) => {
  materials.preload();
  loader.setMaterials(materials);
  loader.load('textured_model.obj', (object) => {
    scene.add(object);
  });
});
```

## Troubleshooting

### "Please upload an OBJ file first!"

**Solution:** Upload a file before clicking Generate

### "Server is not available"

**Cause:** Flux backend not running

**Solution:**
```bash
cd backend
./run_server.sh
```

### Generation takes too long

**Solutions:**
- Reduce texture size to 512
- Reduce inference steps to 30
- Check GPU usage: `nvidia-smi`

### Texture quality is poor

**Solutions:**
- Improve prompt specificity
- Increase texture size to 2048
- Increase inference steps to 75
- Adjust guidance scale (3-7)

### "No UV coordinates" warning

**Info:** This is normal. The app will generate basic UV mapping.

**For better results:** Use models with pre-existing UVs

### Downloaded OBJ doesn't show texture

**Checklist:**
- ✅ All three files in same folder
- ✅ MTL file references correct texture name
- ✅ 3D software supports MTL import
- ✅ Texture mode enabled in viewer

### Generation fails with error

**Check:**
1. Backend logs for errors
2. Prompt length (<512 characters recommended)
3. Server has enough VRAM
4. Network connection to backend

## Performance Tips

### For Speed

- Use texture size 512
- Use 30-40 inference steps
- Lower guidance scale (2-3)
- Use fixed seed to skip random generation

### For Quality

- Use texture size 2048
- Use 75-100 inference steps
- Medium guidance scale (4-5)
- Generate multiple variants and choose best

### For Reproducibility

- Always set a specific seed
- Document your exact settings
- Save prompt for future use
- Keep settings consistent across models

## Technical Details

### UV Mapping

If your OBJ lacks UV coordinates, the app generates them using **spherical projection**:

1. Centers the model
2. Calculates spherical coordinates (θ, φ)
3. Maps to 2D texture space (u, v)
4. Applies to all vertices

**Limitations:**
- May cause distortion on complex shapes
- Best for roughly spherical/cylindrical models
- For best results, pre-generate UVs in modeling software

### Texture Application

1. Loads original OBJ
2. Checks for UV coordinates
3. Generates UVs if missing
4. Creates MTL material file
5. Updates OBJ to reference MTL
6. Copies texture to output directory

### File Structure

```
temp/obj_texture_generator/
├── uploaded_file.obj          # Your upload
├── generated_texture.png      # Flux output
└── output/
    ├── textured_model.obj     # Final OBJ
    ├── textured_model.mtl     # Material
    └── texture.png            # Texture copy
```

## Best Practices

### 1. Start Simple

- Begin with default settings
- Test with small models first
- Use 1024 texture size initially

### 2. Iterate

- Generate → Review → Adjust prompt
- Try different guidance scales
- Compare multiple generations

### 3. Organize

- Save successful prompts
- Document settings that work
- Keep generated files organized

### 4. Optimize

- Find your quality/speed sweet spot
- Use appropriate texture sizes
- Batch similar generations

## FAQ

**Q: Can I use custom models?**
A: Yes! Any valid OBJ file works.

**Q: What if my model is too large?**
A: The app handles large models, but generation time is based on texture size, not model complexity.

**Q: Can I edit the texture afterwards?**
A: Yes! The PNG can be edited in any image editor.

**Q: Does it work offline?**
A: The backend needs to download the model once (24GB), then works offline.

**Q: Can I run multiple generations simultaneously?**
A: Currently no, but you can run multiple frontend instances.

**Q: What file formats are supported?**
A: Input: OBJ only. Output: OBJ + MTL + PNG.

---

For more information, see:
- `/frontend/README.md` - Technical documentation
- `/QUICKSTART.md` - Setup guide  
- `/backend/README.md` - Backend information

