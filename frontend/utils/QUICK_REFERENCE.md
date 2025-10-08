# FLUX Texture Generation - Quick Reference

## One-Command Usage

```bash
# Navigate to utils directory
cd /Users/dave/AI/HPE/HPE-Voxels/frontend/utils

# Generate texture (defaults: 1024x1024, 50 steps)
python generate_flux_texture.py --organ ORGAN_NAME
```

## Common Commands

| Command | Description |
|---------|-------------|
| `python generate_flux_texture.py --organ colon` | Generate texture for colon |
| `python generate_flux_texture.py --organ heart --size 2048 --steps 100` | High quality |
| `python generate_flux_texture.py --organ liver --seed 42` | Reproducible |
| `./generate_all_textures.sh` | Batch generate all models |
| `SIZE=2048 ./generate_all_textures.sh` | Batch high quality |

## Required Setup

1. **Start FLUX Server** (one-time per session)
   ```bash
   cd ../../backend
   python flux_server.py
   ```

2. **Ensure Models Have UV Coordinates**
   ```bash
   python pipeline.py  # If not done yet
   ```

3. **Create UV Masks**
   ```bash
   python create_uv_mask.py  # If not done yet
   ```

## Options Quick Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--organ` | *required* | Organ name (e.g., colon, heart) |
| `--size` | 1024 | Texture size (512, 1024, 2048) |
| `--steps` | 50 | Inference steps (20-150) |
| `--guidance` | 3.5 | Guidance scale (2.0-5.0) |
| `--seed` | random | Random seed |
| `--server` | localhost:8000 | FLUX server address |
| `--no-uv-guidance` | false | Disable UV guidance |
| `--no-mask` | false | Don't apply UV mask |

## Quality Presets

### Fast Preview (512px, 20 steps) - ~10 seconds
```bash
python generate_flux_texture.py --organ colon --size 512 --steps 20
```

### Standard (1024px, 50 steps) - ~30 seconds
```bash
python generate_flux_texture.py --organ colon
```

### High Quality (2048px, 100 steps) - ~2 minutes
```bash
python generate_flux_texture.py --organ colon --size 2048 --steps 100 --guidance 4.0
```

## File Locations

| File | Location |
|------|----------|
| **Script** | `frontend/utils/generate_flux_texture.py` |
| **Config** | `frontend/.env` → `FLUX_SERVER=localhost:8000` |
| **Prompts** | `vista3d_prompts.json` (project root) |
| **UV Masks** | `output/models/ORGAN/uv_mask.png` |
| **Output** | `output/models/ORGAN/textures/flux_texture.png` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot connect to FLUX server` | Start server: `cd backend && python flux_server.py` |
| `No UV coordinates found` | Run: `python add_uv_unwrap.py -i ../../output/obj --in-place` |
| `UV mask not found` | Run: `python create_uv_mask.py --organ ORGAN_NAME` |
| `Model directory not found` | Check organ name matches folder in `output/models/` |

## Example: Generate Colon Texture

```bash
# 1. Start FLUX server (in terminal 1)
cd /Users/dave/AI/HPE/HPE-Voxels/backend
python flux_server.py

# 2. Generate texture (in terminal 2)
cd /Users/dave/AI/HPE/HPE-Voxels/frontend/utils
python generate_flux_texture.py --organ colon

# 3. View result
cd ..
python model_viewer.py  # Select "colon" from dropdown
```

## Available Organs

Common organs with pre-defined prompts:
- `colon`, `heart`, `liver`, `aorta`
- `kidney` (left/right), `lung` (all lobes)
- `brain`, `stomach`, `bladder`, `spleen`
- `pancreas`, `gallbladder`, `esophagus`
- And many more in `vista3d_prompts.json`

## Full Documentation

- **Detailed Guide:** `FLUX_TEXTURE_GENERATION.md`
- **Setup Summary:** `SETUP_SUMMARY.md`
- **All Utils:** `README.md`

---

**Need Help?** Check `FLUX_TEXTURE_GENERATION.md` for detailed examples and troubleshooting.

