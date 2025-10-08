#!/usr/bin/env python3
"""
create_seamless_texture.py

Create seamless, non-repeating textures that perfectly fit the UV layout.
This approach generates textures that are specifically designed for the mesh topology
without visible seams or repeating patterns.

Usage:
    python create_seamless_texture.py --model colon
    python create_seamless_texture.py --all
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from typing import Tuple, Optional
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist


def load_uv_data_from_gltf(gltf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load UV coordinates and face indices from GLTF."""
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        all_uvs = []
        all_indices = []
        
        for mesh in gltf_data.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                attributes = primitive.get('attributes', {})
                
                if 'TEXCOORD_0' in attributes:
                    # Get UV coordinates
                    texcoord_index = attributes['TEXCOORD_0']
                    accessor = gltf_data['accessors'][texcoord_index]
                    buffer_view_index = accessor['bufferView']
                    buffer_view = gltf_data['bufferViews'][buffer_view_index]
                    buffer_index = buffer_view['buffer']
                    buffer_info = gltf_data['buffers'][buffer_index]
                    
                    buffer_path = os.path.join(model_folder, buffer_info['uri'])
                    with open(buffer_path, 'rb') as f:
                        buffer_data = f.read()
                    
                    byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                    uv_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 8]
                    uv_array = np.frombuffer(uv_data, dtype=np.float32).reshape(-1, 2)
                    
                    # Get face indices
                    if 'indices' in primitive:
                        indices_accessor_idx = primitive['indices']
                        accessor = gltf_data['accessors'][indices_accessor_idx]
                        buffer_view_index = accessor['bufferView']
                        buffer_view = gltf_data['bufferViews'][buffer_view_index]
                        buffer_index = buffer_view['buffer']
                        buffer_info = gltf_data['buffers'][buffer_index]
                        
                        buffer_path = os.path.join(model_folder, buffer_info['uri'])
                        with open(buffer_path, 'rb') as f:
                            buffer_data = f.read()
                        
                        byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                        indices_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 4]
                        indices = np.frombuffer(indices_data, dtype=np.uint32)
                        
                        offset = len(all_uvs)
                        all_uvs.extend(uv_array)
                        all_indices.extend((indices.reshape(-1, 3) + offset).tolist())
        
        return np.array(all_uvs), np.array(all_indices, dtype=np.uint32)
    
    except Exception as e:
        print(f"✗ Error loading UV data: {e}")
        return np.array([]), np.array([])


def create_uv_distance_map(uvs: np.ndarray, indices: np.ndarray, size: int = 1024) -> np.ndarray:
    """
    Create a distance map that shows how far each pixel is from UV boundaries.
    This helps create seamless textures by blending near seams.
    """
    # Create base mask
    mask = np.zeros((size, size), dtype=np.float32)
    draw = ImageDraw.Draw(Image.fromarray(mask))
    
    # Convert UV to image coordinates
    uv_coords = uvs.copy()
    uv_coords[:, 1] = 1.0 - uv_coords[:, 1]  # Flip Y
    uv_coords = np.clip(uv_coords, 0.0, 1.0) * (size - 1)
    
    # Draw triangles
    for face_indices in indices.reshape(-1, 3):
        tri_uvs = uv_coords[face_indices]
        pixels = [(int(uv[0]), int(uv[1])) for uv in tri_uvs]
        draw.polygon(pixels, fill=1.0, outline=1.0)
    
    # Create distance map from boundaries
    from scipy.ndimage import distance_transform_edt
    distance_map = distance_transform_edt(mask)
    
    # Normalize to 0-1 range
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    
    return distance_map


def create_organ_specific_texture(
    organ_name: str, 
    uv_mask: np.ndarray,
    distance_map: np.ndarray,
    size: int = 1024
) -> Image.Image:
    """
    Create an organ-specific texture using procedural generation that respects UV layout.
    """
    print(f"  Creating {organ_name} texture...")
    
    # Base colors for different organs
    organ_colors = {
        'colon': {
            'base': (180, 120, 100),      # Beige-brown
            'variation': (40, 30, 20),    # Darker variation
            'highlight': (220, 180, 150), # Light highlight
            'detail': (140, 80, 60)       # Texture detail
        },
        'heart': {
            'base': (180, 80, 80),        # Red-brown
            'variation': (60, 20, 20),    # Darker red
            'highlight': (220, 140, 140), # Light red
            'detail': (120, 40, 40)       # Dark red detail
        },
        'liver': {
            'base': (140, 100, 80),       # Liver brown
            'variation': (50, 35, 25),    # Darker variation
            'highlight': (190, 160, 130), # Light brown
            'detail': (100, 70, 50)       # Dark detail
        },
        'lung': {
            'base': (200, 180, 180),      # Pink-gray
            'variation': (60, 50, 50),    # Darker variation
            'highlight': (240, 220, 220), # Light pink
            'detail': (160, 140, 140)     # Dark detail
        }
    }
    
    # Get colors for this organ
    colors = organ_colors.get(organ_name.lower(), organ_colors['colon'])
    
    # Create base texture
    texture = np.zeros((size, size, 3), dtype=np.float32)
    
    # Layer 1: Base color with subtle noise
    base_noise = np.random.rand(size, size, 3) * 0.1 - 0.05  # -0.05 to 0.05
    texture[:, :, 0] = colors['base'][0] / 255.0 + base_noise[:, :, 0]
    texture[:, :, 1] = colors['base'][1] / 255.0 + base_noise[:, :, 1]
    texture[:, :, 2] = colors['base'][2] / 255.0 + base_noise[:, :, 2]
    
    # Layer 2: Organic variation patterns
    for i in range(3):  # Multiple scales
        scale = 2 ** (i + 3)  # 8, 16, 32
        noise = np.random.rand(size // scale + 1, size // scale + 1, 3)
        noise = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)[:size, :size]
        
        # Blend with distance map for seamless edges
        blend_factor = 0.3 * (distance_map ** 0.5)  # More blending near edges
        texture = texture * (1 - blend_factor[:, :, np.newaxis]) + noise * blend_factor[:, :, np.newaxis]
    
    # Layer 3: Fine detail texture
    detail_noise = np.random.rand(size, size, 3) * 0.15 - 0.075
    detail_colors = np.array(colors['detail']) / 255.0
    detail_layer = detail_noise * detail_colors[np.newaxis, np.newaxis, :]
    
    # Apply detail with distance-based blending
    detail_blend = 0.4 * distance_map[:, :, np.newaxis]
    texture = texture * (1 - detail_blend) + detail_layer * detail_blend
    
    # Layer 4: Highlight variations
    highlight_mask = (distance_map > 0.3) & (distance_map < 0.8)  # Mid-range areas
    highlight_noise = np.random.rand(size, size, 3) * 0.1 - 0.05
    highlight_colors = np.array(colors['highlight']) / 255.0
    highlight_layer = highlight_noise * highlight_colors[np.newaxis, np.newaxis, :]
    
    highlight_blend = highlight_mask.astype(np.float32)[:, :, np.newaxis] * 0.3
    texture = texture * (1 - highlight_blend) + highlight_layer * highlight_blend
    
    # Apply UV mask
    texture = texture * uv_mask[:, :, np.newaxis]
    
    # Smooth transitions
    texture = gaussian_filter(texture, sigma=1.0, mode='nearest')
    
    # Clamp to valid range
    texture = np.clip(texture, 0.0, 1.0)
    
    # Convert to PIL Image
    texture_uint8 = (texture * 255).astype(np.uint8)
    return Image.fromarray(texture_uint8)


def create_seamless_texture_for_model(
    model_name: str,
    models_dir: Path = Path("output/models"),
    size: int = 1024,
    overwrite: bool = False
) -> bool:
    """
    Create a seamless texture for a specific model.
    """
    print(f"\n{'='*70}")
    print(f"Creating Seamless Texture: {model_name}")
    print(f"{'='*70}")
    
    model_dir = models_dir / model_name
    
    # Check if texture exists
    texture_path = model_dir / 'textures' / 'seamless_texture.png'
    if texture_path.exists() and not overwrite:
        print(f"✓ Seamless texture already exists")
        return True
    
    try:
        # Load UV data
        gltf_path = model_dir / 'scene.gltf'
        if not gltf_path.exists():
            print(f"✗ GLTF file not found: {gltf_path}")
            return False
        
        print("Loading UV data...")
        uvs, indices = load_uv_data_from_gltf(str(gltf_path))
        
        if len(uvs) == 0 or len(indices) == 0:
            print(f"✗ No UV data found")
            return False
        
        print(f"✓ Loaded {len(uvs)} UVs and {len(indices)} triangles")
        
        # Create UV mask
        print("Creating UV mask...")
        uv_mask = np.zeros((size, size), dtype=np.float32)
        draw = ImageDraw.Draw(Image.fromarray(uv_mask))
        
        uv_coords = uvs.copy()
        uv_coords[:, 1] = 1.0 - uv_coords[:, 1]
        uv_coords = np.clip(uv_coords, 0.0, 1.0) * (size - 1)
        
        for face_indices in indices.reshape(-1, 3):
            tri_uvs = uv_coords[face_indices]
            pixels = [(int(uv[0]), int(uv[1])) for uv in tri_uvs]
            draw.polygon(pixels, fill=1.0, outline=1.0)
        
        print(f"✓ UV mask created")
        
        # Create distance map for seamless blending
        print("Creating distance map...")
        distance_map = create_uv_distance_map(uvs, indices, size)
        print(f"✓ Distance map created")
        
        # Generate organ-specific texture
        print("Generating seamless texture...")
        texture = create_organ_specific_texture(model_name, uv_mask, distance_map, size)
        
        # Save texture
        textures_dir = model_dir / 'textures'
        textures_dir.mkdir(exist_ok=True)
        
        texture.save(str(texture_path))
        texture.save(str(model_dir / 'textures' / 'diffuse.png'))  # Override existing
        
        print(f"✓ Saved seamless texture: {texture_path}")
        
        # Update GLTF to use new texture
        print("Updating GLTF...")
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        # Ensure material uses the texture
        for material in gltf_data.get('materials', []):
            if 'pbrMetallicRoughness' in material:
                material['pbrMetallicRoughness']['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
        
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        print(f"✓ GLTF updated")
        
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS: {model_name} seamless texture created!")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create seamless, non-repeating textures for 3D models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create seamless texture for single model
    python create_seamless_texture.py --model colon
    
    # Create for all models
    python create_seamless_texture.py --all
    
    # High resolution
    python create_seamless_texture.py --model heart --size 2048
    
    # Overwrite existing
    python create_seamless_texture.py --model liver --overwrite
        """
    )
    
    parser.add_argument('--model', type=str, help='Model name to process')
    parser.add_argument('--all', action='store_true', help='Process all models')
    parser.add_argument('--size', type=int, default=1024, choices=[512, 1024, 2048],
                       help='Texture size (default: 1024)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing textures')
    parser.add_argument('--models-dir', type=str, default='output/models',
                       help='Models directory (default: output/models)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return 1
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "Seamless Texture Generator" + " "*28 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    if args.model:
        # Process single model
        success = create_seamless_texture_for_model(
            args.model, 
            models_dir, 
            args.size, 
            args.overwrite
        )
        return 0 if success else 1
        
    elif args.all:
        # Process all models
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            print(f"✗ No model directories found in {models_dir}")
            return 1
        
        print(f"Processing {len(model_dirs)} models...\n")
        
        successful = 0
        failed = 0
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            if create_seamless_texture_for_model(
                model_name, 
                models_dir, 
                args.size, 
                args.overwrite
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
