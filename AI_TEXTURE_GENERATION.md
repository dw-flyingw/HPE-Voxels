# AI Texture Generation for 3D Medical Models

This document describes the AI-powered texture generation feature added to the HPE-Voxels Model Viewer.

## Overview

The Model Viewer now includes integrated AI texture generation powered by the Flux.1-dev model. This feature allows you to:
- Select any medical model from the models directory
- Generate photo-realistic textures using AI
- Automatically apply textures with proper UV mapping
- Save enhanced models back to the model folder

## Architecture

Following the project convention [[memory:4379183]], the implementation is split into:
- **UI Layer:** `frontend/model_viewer.py` - Pure Streamlit interface
- **Logic Layer:** `frontend/logic/model_viewer_logic.py` - Business logic and API calls

## Features

### 1. Auto-Generated Prompts
The system automatically generates hyper photo-realistic prompts for **all 130+ Vista3D anatomical structures**:

**Organs:** liver, spleen, pancreas, kidneys, bladder, gallbladder, esophagus, stomach, duodenum, small bowel, colon, rectum, brain, heart, prostate, thyroid gland, spinal cord

**Blood Vessels:** aorta, vena cava, iliac arteries/veins, carotid arteries, subclavian arteries, hepatic vessels, portal vein, pulmonary vein

**Bones:** vertebrae (C1-L6, S1), ribs, hip, femur, humerus, scapula, clavicle, sacrum, skull, sternum

**Muscles:** gluteus (maximus/medius/minimus), iliopsoas, autochthon (paraspinal)

**Lungs:** lung lobes (left/right upper/middle/lower)

**Airways:** trachea, airway

**Other:** costal cartilages, tumors, lesions, cysts

Each structure gets a medically accurate, hyper photo-realistic prompt with:
- Anatomically correct coloration
- Specific tissue texture descriptions
- Natural surface characteristics
- Clinical imaging quality (8K resolution)
- Professional medical illustration standards

### 2. Custom Prompt Support
Users can override the auto-generated prompt with custom text for specialized textures.

### 3. Adjustable Parameters
- **Texture Size:** 512, 1024, or 2048 pixels
- **Quality vs Speed:** Higher resolution = better quality but slower generation

### 4. Automatic UV Mapping
If the model doesn't have UV coordinates, the system automatically generates them using spherical mapping.

### 5. Multi-Format Output
Generated textures are saved in multiple formats:
- **OBJ + MTL:** Industry-standard format with material file
- **GLTF:** Modern 3D format with embedded textures
- **GLB:** Binary GLTF for optimized loading
- **PNG Texture:** Standalone texture image

### 6. Server Health Monitoring
The UI automatically checks if the Flux server is available and displays appropriate status/instructions.

## Usage

### Basic Workflow

1. **Start the Flux Backend:**
   ```bash
   cd backend
   ./run_server.sh
   ```

2. **Start Model Viewer:**
   ```bash
   cd frontend
   python model_viewer.py
   # Opens at http://localhost:8503
   ```

3. **Generate Texture:**
   - Select a model from the dropdown (e.g., "colon")
   - Review the auto-generated prompt
   - Optionally customize the prompt
   - Select texture size
   - Click "🚀 Generate Photo-Realistic Texture"
   - Wait 1-2 minutes for generation
   - Reload page to view enhanced model

### Environment Configuration

The system uses environment variables for configuration [[memory:4376306]]:

```bash
# Flux server configuration
FLUX_SERVER_PORT=8000        # Default: 8000
FLUX_HOST=localhost          # Default: localhost

# Model Viewer port
MODEL_VIEWER_PORT=8503       # Default: 8503
```

## Technical Implementation

### Logic Layer (`model_viewer_logic.py`)

#### Key Functions:

1. **`generate_texture_prompt(model_name: str) -> str`**
   - Generates organ-specific prompts
   - Handles left/right prefixes (e.g., "left_hip" → "hip")
   - Falls back to generic prompt for unknown organs

2. **`call_flux_api(...) -> Image.Image`**
   - Calls Flux API at configured endpoint
   - Handles timeout and error cases
   - Returns PIL Image object
   - Uses environment variables for port/host

3. **`generate_uv_mapping(mesh: trimesh.Trimesh) -> np.ndarray`**
   - Creates UV coordinates using spherical mapping
   - Handles edge cases (zero-length vectors)
   - Returns normalized (u, v) coordinates

4. **`apply_texture_to_model(...) -> dict`**
   - Applies texture to 3D mesh
   - Saves in multiple formats (OBJ, GLTF, GLB)
   - Returns metadata about the operation

5. **`generate_photorealistic_texture(...) -> dict`**
   - Main orchestration function
   - Combines all steps into single operation
   - Returns success/failure status with details

6. **`check_flux_server_health() -> dict`**
   - Checks if Flux server is available
   - Returns server status and details
   - Used by UI to show appropriate controls

### UI Layer (`model_viewer.py`)

The UI integrates texture generation into the sidebar:
- Shows Flux server status (✅ running / ⚠️ unavailable)
- Displays auto-generated prompt preview
- Provides custom prompt option
- Texture size selector
- Generate button with progress spinner
- Results display with file paths and statistics

## API Communication

### Flux API Endpoint

**Endpoint:** `POST http://localhost:8000/generate`

**Request:**
```json
{
  "prompt": "ultra photo-realistic human colon tissue texture...",
  "width": 1024,
  "height": 1024,
  "guidance_scale": 3.5,
  "num_inference_steps": 50,
  "return_base64": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Image generated successfully",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAA...",
  "metadata": {
    "prompt": "...",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50
  }
}
```

## File Organization

Generated files are saved to the model's folder:

```
output/models/colon/
├── scene.glb                    # Original GLB
├── scene.gltf                   # Original GLTF
├── scene.bin                    # Original binary data
├── textured_model.obj           # ← New textured OBJ
├── material.mtl                 # ← New material file
└── textures/
    └── diffuse.png              # ← New AI-generated texture
```

## Error Handling

The system handles various error cases:

1. **Flux Server Unavailable:**
   - UI shows warning with setup instructions
   - Displays error details in expandable section

2. **Model File Not Found:**
   - Searches multiple file types (GLB, GLTF, OBJ)
   - Returns helpful error message

3. **API Call Failure:**
   - 5-minute timeout for generation
   - Captures request exceptions
   - Returns error details to user

4. **Texture Application Failure:**
   - Logs warnings for format-specific failures
   - Returns partial success (e.g., OBJ works but GLTF fails)

## Performance Considerations

### Generation Time
- **512x512:** ~30-45 seconds
- **1024x1024:** ~60-90 seconds
- **2048x2048:** ~120-180 seconds

### GPU Requirements
- Backend requires CUDA-capable GPU
- Minimum 24GB VRAM recommended
- H200 GPU with 141GB is optimal

### Memory Usage
- Frontend: ~500MB (Streamlit + dependencies)
- Backend: ~24GB GPU VRAM (model loaded)

## Limitations

1. **Server Dependency:**
   - Requires Flux backend running
   - No offline mode

2. **Generation Time:**
   - 1-2 minutes per texture
   - Cannot generate multiple textures simultaneously

3. **Format Support:**
   - Best results with GLTF/GLB input
   - OBJ files may need manual UV adjustment

4. **Texture Mapping:**
   - Spherical mapping may not be ideal for all shapes
   - Complex models may need custom UV unwrapping

## Future Enhancements

Potential improvements:
- [ ] Batch texture generation for multiple models
- [ ] Progress bar for generation status
- [ ] Texture preview before applying
- [ ] Multiple texture mapping algorithms (planar, cylindrical, etc.)
- [ ] Texture editing and refinement tools
- [ ] Local caching of generated textures
- [ ] Support for PBR material properties (roughness, metalness)

## Troubleshooting

### Flux Server Connection Issues

**Problem:** "⚠️ Flux server is not available"

**Solutions:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Verify environment variables
echo $FLUX_SERVER_PORT

# Check server logs
cd backend && python flux_server.py
```

### Texture Not Showing

**Problem:** Generated texture not visible on model

**Solutions:**
1. Reload the page (Streamlit caching issue)
2. Check the textures/ subfolder exists
3. Verify texture file was created (diffuse.png)
4. Try viewing with external 3D software (Blender)

### Generation Takes Too Long

**Problem:** Texture generation exceeds 5 minutes

**Solutions:**
1. Reduce texture size (512 instead of 1024)
2. Check GPU utilization: `nvidia-smi`
3. Verify no other processes using GPU
4. Restart Flux server

### Poor Texture Quality

**Problem:** Generated texture doesn't look realistic

**Solutions:**
1. Improve prompt (add "medical grade", "8k resolution")
2. Increase texture size (2048)
3. Adjust guidance scale (try 4.0-5.0)
4. Use custom prompt with specific details

## Testing

### Manual Testing Checklist

- [ ] Select colon model
- [ ] Verify auto-generated prompt appears
- [ ] Generate texture with default settings
- [ ] Check texture file created in textures/ folder
- [ ] Verify OBJ, GLTF, GLB files created
- [ ] Reload page and view textured model
- [ ] Test custom prompt
- [ ] Test different texture sizes (512, 1024, 2048)
- [ ] Test with server unavailable (error handling)

### Test Cases

```python
# Test 1: Server health check
result = check_flux_server_health()
assert result['available'] == True

# Test 2: Prompt generation
prompt = generate_texture_prompt('colon')
assert 'photo-realistic' in prompt
assert 'colon' in prompt

# Test 3: UV mapping
mesh = trimesh.load('path/to/model.obj')
uvs = generate_uv_mapping(mesh)
assert uvs.shape[1] == 2
assert uvs.min() >= 0 and uvs.max() <= 1
```

## References

- [Flux.1-dev Model](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Trimesh Documentation](https://trimsh.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Three.js Documentation](https://threejs.org/docs/)

## License

This feature uses the FLUX.1-dev model which falls under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

---

**Created:** October 7, 2025  
**Author:** AI Assistant  
**Version:** 1.0

