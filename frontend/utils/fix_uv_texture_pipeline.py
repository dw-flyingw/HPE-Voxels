#!/usr/bin/env python3
"""
fix_uv_texture_pipeline.py

Complete pipeline to fix UV unwrapping and texture issues for existing models.
This script:
1. Re-unwraps OBJ files with optimized xatlas settings
2. Creates accurate UV masks with triangle rasterization
3. Generates non-repeating textures using FLUX
4. Updates GLTF/GLB files with correct UV mappings

Usage:
    # Fix single model
    python fix_uv_texture_pipeline.py --model colon
    
    # Fix all models
    python fix_uv_texture_pipeline.py --all
    
    # Fix UVs only (skip texture generation)
    python fix_uv_texture_pipeline.py --model heart --skip-texture
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import trimesh
import xatlas
from PIL import Image, ImageDraw
import requests
import base64
from dotenv import load_dotenv


def check_dependencies() -> Tuple[bool, list]:
    """Check if all required dependencies are available."""
    missing = []
    
    try:
        import trimesh
    except ImportError:
        missing.append("trimesh")
    
    try:
        import xatlas
    except ImportError:
        missing.append("xatlas")
    
    try:
        import PIL
    except ImportError:
        missing.append("Pillow")
    
    try:
        import pygltflib
    except ImportError:
        missing.append("pygltflib")
    
    return len(missing) == 0, missing


def step1_rewrap_obj(obj_path: Path, verbose: bool = True) -> Tuple[trimesh.Trimesh, dict]:
    """
    Step 1: Re-unwrap OBJ file with optimized settings.
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: Optimized UV Unwrapping")
        print("="*70)
    
    # Load mesh
    mesh = trimesh.load(str(obj_path), process=False)
    if hasattr(mesh, 'geometry'):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    if verbose:
        print(f"  Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Apply xatlas unwrapping
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    
    # Create new mesh with UV seams
    new_vertices = vertices[vmapping]
    new_faces = indices.reshape(-1, 3)
    
    unwrapped_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )
    
    # Normalize UVs to 0-1 range with small padding
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range = np.where(uv_range < 1e-6, 1.0, uv_range)
    
    uvs_normalized = (uvs - uv_min) / uv_range
    padding = 0.01
    scale = 1.0 - 2 * padding
    uvs_normalized = uvs_normalized * scale + padding
    
    unwrapped_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs_normalized)
    
    # Statistics
    stats = {
        'original_vertices': len(vertices),
        'unwrapped_vertices': len(new_vertices),
        'seam_vertices': len(vmapping) - len(np.unique(vmapping)),
        'faces': len(new_faces),
        'uv_min': uv_min.tolist(),
        'uv_max': uv_max.tolist(),
        'normalized': True
    }
    
    if verbose:
        print(f"  ✓ Unwrapped: {len(new_vertices)} vertices")
        print(f"  ✓ UV range normalized to [0.01, 0.99]")
        print(f"  ✓ Seam vertices: {stats['seam_vertices']}")
    
    return unwrapped_mesh, stats


def step2_create_uv_mask(mesh: trimesh.Trimesh, size: int = 1024, verbose: bool = True) -> Image.Image:
    """
    Step 2: Create accurate UV mask by rasterizing triangles.
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: Creating UV Mask")
        print("="*70)
    
    uvs = mesh.visual.uv
    faces = mesh.faces
    
    # Create RGBA image with supersampling
    supersample = 2
    render_size = size * supersample
    mask_img = Image.new('RGBA', (render_size, render_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_img)
    
    # Convert UV to image coordinates
    uv_coords = uvs.copy()
    uv_coords[:, 1] = 1.0 - uv_coords[:, 1]  # Flip Y
    uv_coords = np.clip(uv_coords, 0.0, 1.0) * (render_size - 1)
    
    if verbose:
        print(f"  Rasterizing {len(faces)} triangles at {render_size}x{render_size}...")
    
    # Draw triangles
    for i, face_indices in enumerate(faces):
        tri_uvs = uv_coords[face_indices]
        pixels = [(int(uv[0]), int(uv[1])) for uv in tri_uvs]
        draw.polygon(pixels, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
    
    # Downsample
    mask_img = mask_img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Calculate coverage
    mask_array = np.array(mask_img)
    coverage = np.sum(mask_array[:, :, 3] > 0) / (size * size) * 100
    
    if verbose:
        print(f"  ✓ UV mask created: {size}x{size}")
        print(f"  ✓ Coverage: {coverage:.1f}%")
    
    return mask_img


def step3_save_model_files(
    mesh: trimesh.Trimesh,
    model_dir: Path,
    obj_path: Path,
    uv_mask: Image.Image,
    verbose: bool = True
) -> bool:
    """
    Step 3: Save updated OBJ, GLTF, GLB, and UV mask.
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: Saving Model Files")
        print("="*70)
    
    try:
        # Create model directory
        model_dir.mkdir(exist_ok=True, parents=True)
        textures_dir = model_dir / 'textures'
        textures_dir.mkdir(exist_ok=True)
        
        # Save unwrapped OBJ
        mesh.export(str(obj_path), file_type='obj')
        if verbose:
            print(f"  ✓ Saved OBJ: {obj_path}")
        
        # Save GLTF and GLB
        glb_path = model_dir / 'scene.glb'
        gltf_path = model_dir / 'scene.gltf'
        bin_path = model_dir / 'scene.bin'
        
        mesh.export(str(glb_path), file_type='glb')
        mesh.export(str(gltf_path), file_type='gltf')
        
        if verbose:
            print(f"  ✓ Saved GLB: {glb_path}")
            print(f"  ✓ Saved GLTF: {gltf_path}")
        
        # Fix GLTF material to ensure full brightness
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        for material in gltf_data.get('materials', []):
            if 'pbrMetallicRoughness' in material:
                material['pbrMetallicRoughness']['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
        
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        # Save UV masks
        uv_mask.save(str(model_dir / 'uv_mask_rgba.png'))
        uv_mask.convert('RGB').save(str(model_dir / 'uv_mask.png'))
        
        if verbose:
            print(f"  ✓ Saved UV masks")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving files: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_generate_texture(
    organ_name: str,
    model_dir: Path,
    uv_mask: Image.Image,
    size: int = 1024,
    flux_server: str = "localhost:8000",
    verbose: bool = True
) -> bool:
    """
    Step 4: Generate non-repeating texture with FLUX.
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: Generating Texture")
        print("="*70)
    
    # Load prompts
    try:
        script_dir = Path(__file__).parent
        prompts_path = script_dir.parent / 'conf' / 'vista3d_prompts.json'
        
        with open(prompts_path, 'r') as f:
            data = json.load(f)
        
        prompts = {}
        for item in data.get('prompts', []):
            name = item.get('name', '').lower().strip()
            prompt = item.get('prompt', '')
            if name and prompt:
                prompts[name] = prompt
        
        default_template = data.get('default_template', 
            "hyper photo-realistic human {structure} anatomical structure, medical photography")
        
        # Get prompt for this organ
        organ_key = organ_name.lower().strip().replace('_', ' ')
        if organ_key in prompts:
            prompt = prompts[organ_key]
        else:
            prompt = default_template.replace('{structure}', organ_name.replace('_', ' '))
        
        # Enhance prompt to prevent repeating patterns
        enhanced_prompt = f"{prompt}, unique non-repeating texture, continuous organic variations, seamless UV mapping"
        
        if verbose:
            print(f"  Prompt: {prompt[:80]}...")
        
    except Exception as e:
        print(f"  ✗ Error loading prompts: {e}")
        return False
    
    # Connect to FLUX server
    server_url = f"http://{flux_server}"
    
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"  ✗ FLUX server not healthy")
            return False
    except Exception as e:
        print(f"  ✗ Cannot connect to FLUX server: {e}")
        print(f"  Make sure server is running: cd backend && python flux_server.py")
        return False
    
    if verbose:
        print(f"  ✓ Connected to FLUX server")
    
    # Encode UV mask
    import io
    buffered = io.BytesIO()
    uv_mask.save(buffered, format='PNG')
    mask_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Generate texture
    payload = {
        'prompt': enhanced_prompt,
        'control_image': mask_b64,
        'control_type': 'uv_layout',
        'height': size,
        'width': size,
        'guidance_scale': 3.5,
        'num_inference_steps': 50,
        'seed': None,
        'return_base64': False
    }
    
    if verbose:
        print(f"  Generating {size}x{size} texture...")
    
    try:
        response = requests.post(f'{server_url}/generate_with_control', json=payload, timeout=300)
        
        if response.status_code == 200:
            texture = Image.open(io.BytesIO(response.content))
            
            textures_dir = model_dir / 'textures'
            texture.save(str(textures_dir / 'flux_texture.png'))
            texture.save(str(textures_dir / 'diffuse.png'))
            
            if verbose:
                print(f"  ✓ Texture saved")
            
            return True
        else:
            print(f"  ✗ Generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error during generation: {e}")
        return False


def process_model(
    model_name: str,
    obj_dir: Path = Path("output/obj"),
    models_dir: Path = Path("output/models"),
    texture_size: int = 1024,
    skip_texture: bool = False,
    flux_server: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """
    Complete pipeline to fix a single model.
    """
    if verbose:
        print("\n" + "🔧 " + "="*68 + " 🔧")
        print(f"   Fixing Model: {model_name.upper()}")
        print("🔧 " + "="*68 + " 🔧")
    
    obj_path = obj_dir / f"{model_name}.obj"
    model_dir = models_dir / model_name
    
    if not obj_path.exists():
        print(f"✗ OBJ file not found: {obj_path}")
        return False
    
    # Get FLUX server
    if flux_server is None:
        script_dir = Path(__file__).parent
        env_path = script_dir.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        flux_server = os.getenv('FLUX_SERVER', 'localhost:8000')
    
    try:
        # Step 1: Re-unwrap OBJ
        unwrapped_mesh, stats = step1_rewrap_obj(obj_path, verbose)
        
        # Step 2: Create UV mask
        uv_mask = step2_create_uv_mask(unwrapped_mesh, size=1024, verbose=verbose)
        
        # Step 3: Save files
        if not step3_save_model_files(unwrapped_mesh, model_dir, obj_path, uv_mask, verbose):
            return False
        
        # Step 4: Generate texture (optional)
        if not skip_texture:
            step4_generate_texture(model_name, model_dir, uv_mask, texture_size, flux_server, verbose)
        else:
            if verbose:
                print("\n[Skipped] Texture generation")
        
        if verbose:
            print("\n" + "✅ " + "="*68 + " ✅")
            print(f"   SUCCESS: {model_name} fixed!")
            print("✅ " + "="*68 + " ✅\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix UV unwrapping and texture issues for medical organ models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix single model (UVs + texture)
    python fix_uv_texture_pipeline.py --model colon
    
    # Fix all models
    python fix_uv_texture_pipeline.py --all
    
    # Fix UVs only (skip texture generation)
    python fix_uv_texture_pipeline.py --model heart --skip-texture
    
    # High resolution texture
    python fix_uv_texture_pipeline.py --model liver --texture-size 2048
    
    # Custom server
    python fix_uv_texture_pipeline.py --model colon --server 192.168.1.100:8000
        """
    )
    
    parser.add_argument('--model', type=str, help='Model name to fix')
    parser.add_argument('--all', action='store_true', help='Fix all models')
    parser.add_argument('--texture-size', type=int, default=1024, choices=[512, 1024, 2048],
                       help='Texture size (default: 1024)')
    parser.add_argument('--skip-texture', action='store_true', help='Skip texture generation')
    parser.add_argument('--server', type=str, help='FLUX server address (default: from .env)')
    parser.add_argument('--obj-dir', type=str, default='output/obj', help='OBJ files directory')
    parser.add_argument('--models-dir', type=str, default='output/models', help='Models directory')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"✗ Missing dependencies: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return 1
    
    obj_dir = Path(args.obj_dir)
    models_dir = Path(args.models_dir)
    verbose = not args.quiet
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "UV & Texture Fix Pipeline" + " "*27 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    if args.model:
        # Fix single model
        success = process_model(
            args.model,
            obj_dir,
            models_dir,
            args.texture_size,
            args.skip_texture,
            args.server,
            verbose
        )
        return 0 if success else 1
        
    elif args.all:
        # Fix all models
        obj_files = sorted(obj_dir.glob("*.obj"))
        if not obj_files:
            print(f"✗ No OBJ files found in {obj_dir}")
            return 1
        
        print(f"Processing {len(obj_files)} models...\n")
        
        successful = 0
        failed = 0
        
        for obj_file in obj_files:
            model_name = obj_file.stem
            if process_model(
                model_name,
                obj_dir,
                models_dir,
                args.texture_size,
                args.skip_texture,
                args.server,
                verbose
            ):
                successful += 1
            else:
                failed += 1
        
        print("\n" + "="*70)
        print(f"SUMMARY: {successful} successful, {failed} failed")
        print("="*70)
        
        return 0 if failed == 0 else 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

