#!/usr/bin/env python3
"""
Complete Model Texture Generation Pipeline
Automates: UV unwrapping → UV mask creation → FLUX texture generation → Inpainting

Usage:
    python generate_model_texture.py --model colon
    python generate_model_texture.py --model left_hip --size 2048
    python generate_model_texture.py --all  # Process all models
"""

import argparse
import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Tuple
import trimesh
import xatlas
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_dilation, distance_transform_edt
from dotenv import load_dotenv


def load_vista3d_prompts(prompts_file: str = "vista3d_prompts.json") -> dict:
    """Load Vista3D prompts from config file."""
    script_dir = Path(__file__).parent
    prompts_path = script_dir.parent / "conf" / prompts_file
    
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(prompts_path, 'r') as f:
        data = json.load(f)
    
    prompts = {}
    for item in data.get('prompts', []):
        name = item.get('name', '').lower().strip()
        prompt = item.get('prompt', '')
        if name and prompt:
            prompts[name] = prompt
    
    prompts['_default'] = data.get('default_template', 
        "hyper photo-realistic human {structure} anatomical structure, medical photography, "
        "anatomically accurate surface texture, natural clinical appearance, high detail, "
        "8K resolution, professional medical illustration")
    
    return prompts


def get_prompt_for_organ(organ_name: str, prompts: dict) -> str:
    """Get appropriate prompt for an organ."""
    organ_key = organ_name.lower().strip().replace('_', ' ')
    
    if organ_key in prompts:
        return prompts[organ_key]
    
    default_template = prompts.get('_default', '')
    if default_template:
        return default_template.replace('{structure}', organ_name.replace('_', ' '))
    
    return f"hyper photo-realistic human {organ_name.replace('_', ' ')} anatomical structure, medical photography, 8K resolution"


def step1_unwrap_uv(obj_path: Path, verbose: bool = True) -> Tuple[trimesh.Trimesh, int]:
    """Step 1: Apply xatlas UV unwrapping to mesh."""
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: UV Unwrapping with xatlas")
        print("="*70)
    
    # Load mesh
    mesh = trimesh.load(str(obj_path), process=False)
    if hasattr(mesh, 'geometry'):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    if verbose:
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Apply xatlas unwrapping
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    
    # Create new mesh with UV seams
    new_vertices = mesh.vertices[vmapping]
    new_faces = indices
    
    mesh_unwrapped = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )
    mesh_unwrapped.visual = trimesh.visual.TextureVisuals(uv=uvs)
    
    if verbose:
        print(f"✓ xatlas unwrapping complete")
        print(f"  New vertices: {len(new_vertices)} (includes UV seams)")
        print(f"  UV coordinates: {len(uvs)}")
    
    return mesh_unwrapped, len(uvs)


def step2_create_uv_mask(model_dir: Path, size: int = 1024, verbose: bool = True) -> float:
    """Step 2: Create RGBA UV mask from GLTF."""
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: Creating RGBA UV Mask")
        print("="*70)
    
    gltf_path = model_dir / 'scene.gltf'
    with open(gltf_path, 'r') as f:
        gltf_data = json.load(f)
    
    # Extract UVs
    uvs = []
    for mesh in gltf_data.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            attributes = primitive.get('attributes', {})
            
            if 'TEXCOORD_0' in attributes:
                texcoord_index = attributes['TEXCOORD_0']
                accessor = gltf_data['accessors'][texcoord_index]
                buffer_view_index = accessor['bufferView']
                buffer_view = gltf_data['bufferViews'][buffer_view_index]
                buffer_index = buffer_view['buffer']
                buffer_info = gltf_data['buffers'][buffer_index]
                
                buffer_path = model_dir / buffer_info['uri']
                with open(buffer_path, 'rb') as f:
                    buffer_data = f.read()
                
                byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                uv_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 8]
                uv_array = np.frombuffer(uv_data, dtype=np.float32).reshape(-1, 2)
                uvs.extend(uv_array)
    
    uvs = np.array(uvs)
    
    # Get indices
    indices_list = []
    for mesh in gltf_data.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            indices_accessor_idx = primitive.get('indices')
            if indices_accessor_idx is not None:
                accessor = gltf_data['accessors'][indices_accessor_idx]
                buffer_view_index = accessor['bufferView']
                buffer_view = gltf_data['bufferViews'][buffer_view_index]
                buffer_index = buffer_view['buffer']
                buffer_info = gltf_data['buffers'][buffer_index]
                
                buffer_path = model_dir / buffer_info['uri']
                with open(buffer_path, 'rb') as f:
                    buffer_data = f.read()
                
                byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                indices_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 4]
                indices = np.frombuffer(indices_data, dtype=np.uint32)
                indices_list.extend(indices)
    
    indices_list = np.array(indices_list)
    
    # Create RGBA UV mask
    uv_mask_rgba = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(uv_mask_rgba)
    
    for i in range(0, len(indices_list), 3):
        face_indices = indices_list[i:i+3]
        face_uvs = uvs[face_indices]
        pixels = []
        for uv in face_uvs:
            x = int(np.clip(uv[0], 0, 1) * (size - 1))
            y = int(np.clip(1.0 - uv[1], 0, 1) * (size - 1))
            pixels.append((x, y))
        if len(pixels) == 3:
            draw.polygon(pixels, fill=(255, 255, 255, 255))
    
    # Save mask
    uv_mask_rgba.save(str(model_dir / 'uv_mask_rgba.png'))
    
    # Calculate coverage
    mask_array = np.array(uv_mask_rgba)
    coverage = np.sum(mask_array[:, :, 3] > 0) / (size * size) * 100
    
    if verbose:
        print(f"✓ UV mask created: {size}x{size}")
        print(f"  Coverage: {coverage:.1f}%")
        print(f"  Saved to: {model_dir / 'uv_mask_rgba.png'}")
    
    return coverage


def step3_generate_texture(
    organ_name: str,
    model_dir: Path,
    size: int = 1024,
    flux_server: str = "localhost:8000",
    verbose: bool = True
) -> bool:
    """Step 3: Generate texture with FLUX using UV mask control."""
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: Generating Texture with FLUX")
        print("="*70)
    
    # Load prompts
    prompts = load_vista3d_prompts()
    prompt = get_prompt_for_organ(organ_name, prompts)
    
    # Enhanced prompt for non-repeating texture (concise to stay under 77 token limit)
    enhanced_prompt = f"""{prompt}, unique non-repeating seamless texture, organic detail"""
    
    if verbose:
        print(f"Prompt: {prompt[:80]}...")
    
    # Load UV mask
    uv_mask = Image.open(model_dir / 'uv_mask_rgba.png')
    
    # Encode as base64
    import io
    buffered = io.BytesIO()
    uv_mask.save(buffered, format='PNG')
    mask_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Generate with FLUX
    server_url = f'http://{flux_server}'
    
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
    
    try:
        response = requests.post(f'{server_url}/generate_with_control', json=payload, timeout=300)
        
        if response.status_code == 200:
            texture = Image.open(io.BytesIO(response.content))
            
            # Save texture
            textures_dir = model_dir / 'textures'
            textures_dir.mkdir(exist_ok=True)
            
            texture.save(str(textures_dir / 'flux_texture.png'))
            texture.save(str(textures_dir / 'diffuse.png'))
            
            if verbose:
                print(f"✓ Texture generated: {texture.size}")
                print(f"  Saved to: {textures_dir / 'diffuse.png'}")
            
            return True
        else:
            print(f"✗ Generation failed: {response.status_code}")
            print(f"  {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def step4_inpaint_texture(model_dir: Path, verbose: bool = True) -> float:
    """Step 4: Fill black areas with inpainting."""
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: Inpainting Black Areas")
        print("="*70)
    
    texture_path = model_dir / 'textures' / 'diffuse.png'
    texture = Image.open(texture_path)
    tex_array = np.array(texture)
    
    # Identify empty areas
    textured_mask = np.any(tex_array > 10, axis=2)
    empty_mask = ~textured_mask
    
    empty_percentage = np.sum(empty_mask) / empty_mask.size * 100
    
    if verbose:
        print(f"Empty pixels to fill: {np.sum(empty_mask)} ({empty_percentage:.1f}%)")
    
    if empty_percentage < 1:
        if verbose:
            print("✓ No significant empty areas - skipping inpainting")
        return 100.0
    
    # Inpaint using nearest neighbor
    dist, indices = distance_transform_edt(empty_mask, return_indices=True)
    
    filled_texture = tex_array.copy()
    empty_coords = np.where(empty_mask)
    for i in range(len(empty_coords[0])):
        y, x = empty_coords[0][i], empty_coords[1][i]
        nearest_y, nearest_x = indices[0][y, x], indices[1][y, x]
        filled_texture[y, x] = tex_array[nearest_y, nearest_x]
    
    # Smooth transitions
    filled_img = Image.fromarray(filled_texture.astype(np.uint8))
    blend_mask_array = binary_dilation(empty_mask, iterations=3) & ~empty_mask
    blend_mask = Image.fromarray((blend_mask_array * 255).astype(np.uint8))
    
    blurred = filled_img.filter(ImageFilter.GaussianBlur(radius=2))
    final_texture = Image.composite(blurred, filled_img, blend_mask)
    
    # Save
    final_texture.save(str(texture_path))
    final_texture.save(str(model_dir / 'textures' / 'flux_texture.png'))
    
    # Verify
    final_array = np.array(final_texture)
    final_coverage = np.sum(np.any(final_array > 10, axis=2)) / (final_array.shape[0] * final_array.shape[1]) * 100
    
    if verbose:
        print(f"✓ Inpainting complete")
        print(f"  Final coverage: {final_coverage:.1f}%")
    
    return final_coverage


def process_model(
    model_name: str,
    models_dir: Path = Path("output/models"),
    obj_dir: Path = Path("output/obj"),
    size: int = 1024,
    flux_server: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """Complete pipeline for a single model."""
    
    if verbose:
        print("\n" + "🎨 " + "="*68 + " 🎨")
        print(f"   Processing Model: {model_name.upper()}")
        print("🎨 " + "="*68 + " 🎨")
    
    # Setup paths
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
        # Step 1: UV Unwrapping
        mesh_unwrapped, uv_count = step1_unwrap_uv(obj_path, verbose)
        
        # Save unwrapped mesh
        mesh_unwrapped.export(str(obj_path), file_type='obj')
        
        # Create model directory
        model_dir.mkdir(exist_ok=True)
        (model_dir / 'textures').mkdir(exist_ok=True)
        
        # Save GLTF/GLB
        mesh_unwrapped.export(str(model_dir / 'scene.glb'), file_type='glb')
        mesh_unwrapped.export(str(model_dir / 'scene.gltf'), file_type='gltf')
        
        # Fix GLTF material baseColorFactor to full brightness
        import json
        gltf_path = model_dir / 'scene.gltf'
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        for material in gltf_data.get('materials', []):
            if 'pbrMetallicRoughness' in material:
                material['pbrMetallicRoughness']['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
        
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        if verbose:
            print(f"✓ Fixed GLTF material for full brightness")
        
        # Step 2: Create UV Mask
        coverage = step2_create_uv_mask(model_dir, size, verbose)
        
        # Step 3: Generate Texture
        if not step3_generate_texture(model_name, model_dir, size, flux_server, verbose):
            return False
        
        # Step 4: Inpaint
        final_coverage = step4_inpaint_texture(model_dir, verbose)
        
        if verbose:
            print("\n" + "✅ " + "="*68 + " ✅")
            print(f"   SUCCESS: {model_name} texture complete!")
            print(f"   Final coverage: {final_coverage:.1f}%")
            print("✅ " + "="*68 + " ✅\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete Model Texture Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single model
    python generate_model_texture.py --model colon
    
    # Process with custom size
    python generate_model_texture.py --model left_hip --size 2048
    
    # Process all models in directory
    python generate_model_texture.py --all
    
    # Custom server
    python generate_model_texture.py --model heart --server 192.168.1.100:8000
        """
    )
    
    parser.add_argument('--model', type=str, help='Model name to process')
    parser.add_argument('--all', action='store_true', help='Process all models in obj directory')
    parser.add_argument('--size', type=int, default=1024, choices=[512, 1024, 2048], help='Texture size (default: 1024)')
    parser.add_argument('--server', type=str, help='FLUX server address (default: from .env or localhost:8000)')
    parser.add_argument('--models-dir', type=str, default='output/models', help='Models directory')
    parser.add_argument('--obj-dir', type=str, default='output/obj', help='OBJ files directory')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode (less output)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    obj_dir = Path(args.obj_dir)
    verbose = not args.quiet
    
    if args.all:
        # Process all OBJ files
        obj_files = sorted(obj_dir.glob("*.obj"))
        if not obj_files:
            print(f"No OBJ files found in {obj_dir}")
            sys.exit(1)
        
        print(f"\n📦 Processing {len(obj_files)} models...")
        
        successful = 0
        failed = 0
        
        for obj_file in obj_files:
            model_name = obj_file.stem
            if process_model(model_name, models_dir, obj_dir, args.size, args.server, verbose):
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"SUMMARY: {successful} successful, {failed} failed")
        print(f"{'='*70}\n")
        
    elif args.model:
        # Process single model
        if process_model(args.model, models_dir, obj_dir, args.size, args.server, verbose):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

