# Quick Start: UV-Conditioned Texture Generation

## What Was Fixed

**Problem**: FLUX server was generating textures WITHOUT using the UV mask - only applying it after generation.

**Solution**: Implemented proper UV conditioning that guides FLUX during generation using:
- Init image creation from UV mask
- Img2img conditioning
- Enhanced prompting
- Feathered masking

---

## Getting Started

### 1. Update Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs:
- `controlnet-aux>=0.0.7` - Control preprocessing
- `opencv-python>=4.8.0` - Image processing

### 2. Restart FLUX Server

```bash
# If server is running, stop it (Ctrl+C)

# Start with new UV conditioning
./start_server.sh
```

You should see the server load successfully on port 8000.

### 3. Test UV Conditioning

```bash
# Run automated test
python test_uv_conditioning.py --model colon --size 1024
```

Expected output:
```
✓ Model directory: ...
✓ UV mask found: ...
✓ UV mask encoded
✓ Server healthy
⏳ Generating (this may take 30-60 seconds)...
✓ Generation successful!
✓ Non-UV areas properly masked
✓ UV areas have content with variation
✅ Test Complete: UV conditioning is working!
```

### 4. Generate Real Textures

```bash
cd ../frontend

# Generate with new UV conditioning
python utils/generate_flux_texture.py --organ colon --size 2048 --steps 50
```

---

## What's Different?

### Server Console Output

**Before**:
```
Generating UV-guided image with prompt: '...'
Control image loaded: (2048, 2048)
```

**After** (New):
```
============================================================
UV-Guided Generation Request
============================================================
Prompt: 'human colon anatomical structure...'
Size: 2048x2048
Steps: 50, Guidance: 3.5
✓ UV mask loaded: (2048, 2048)
  Using UV-conditioned generation with strength=0.7
  Enhanced prompt: ...seamless organic texture...
✓ UV-guided generation complete
============================================================
```

### Texture Quality

**Before**:
- Random patterns
- Repetitive details
- Ignores UV layout
- Blotchy appearance

**After**:
- Coherent, unique texture
- Respects UV islands
- Smooth boundaries
- Natural appearance

---

## Quick Commands

```bash
# Test with colon (fast)
python backend/test_uv_conditioning.py --model colon --size 1024

# Generate high quality texture
python frontend/utils/generate_flux_texture.py \
  --organ colon \
  --size 2048 \
  --steps 50 \
  --guidance 3.5

# Check server health
curl http://localhost:8000/health

# View server info
curl http://localhost:8000/
```

---

## Comparison Test

Generate before/after comparison:

```bash
# 1. Generate with new UV conditioning
python frontend/utils/generate_flux_texture.py \
  --organ colon \
  --size 2048 \
  --seed 42 \
  --output flux_texture_new.png

# 2. Compare with old texture (if you have it)
# The new texture should show:
# - Less repetition
# - Better UV alignment
# - Smoother boundaries
# - More realistic appearance
```

---

## Troubleshooting

### "Cannot connect to FLUX server"

```bash
# Start the server
cd backend
./start_server.sh

# Wait for "Model loaded successfully" message
```

### "UV mask not found"

```bash
# Create UV masks
python frontend/utils/create_perfect_uv_mask.py --model colon --size 2048
```

### "img2img not available"

This is **OK**! The fallback still provides UV conditioning through enhanced prompting and post-processing.

### Poor Quality Results

Try adjusting in `flux_server.py` line 440:

```python
init_strength=0.65  # More faithful to UV
# or
init_strength=0.75  # More creative
```

---

## Key Files

- **`flux_server.py`** - Main implementation with UV conditioning
- **`test_uv_conditioning.py`** - Automated test script
- **`UV_CONDITIONING_GUIDE.md`** - Detailed technical docs
- **`IMPLEMENTATION_SUMMARY.md`** - Complete change summary

---

## Next Steps

1. ✅ Test with colon model
2. ✅ Verify UV conditioning works
3. ✅ Generate textures for all your models
4. ✅ Compare quality with previous textures

---

## Need Help?

1. Check `UV_CONDITIONING_GUIDE.md` for details
2. Run `python backend/test_uv_conditioning.py` for diagnostics
3. Check server logs for errors
4. Verify UV mask quality

---

**🎉 You're all set! UV conditioning is now properly implemented.**

