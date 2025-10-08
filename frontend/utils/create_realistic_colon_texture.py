#!/usr/bin/env python3
"""
create_realistic_colon_texture.py

Create a realistic, visible colon texture that properly maps to the UV layout.
This creates a texture that's actually visible and looks like colon tissue.

Usage:
    python create_realistic_colon_texture.py --model colon
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from typing import Tuple
import trimesh


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


def create_uv_mask(uvs: np.ndarray, indices: np.ndarray, size: int = 1024) -> np.ndarray:
    """Create UV mask by rasterizing triangles."""
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
    
    return mask


def create_realistic_colon_texture(size: int = 1024) -> Image.Image:
    """
    Create a realistic colon texture with proper colors and visible details.
    """
    print(f"  Creating realistic colon texture ({size}x{size})...")
    
    # Create base image
    img = Image.new('RGB', (size, size), (200, 180, 160))  # Light beige background
    
    # Base colon colors
    base_color = (180, 140, 120)  # Beige-brown
    mucosa_color = (220, 190, 170)  # Light pink-beige
    detail_color = (140, 100, 80)  # Darker brown
    
    # Fill with base color
    img_array = np.array(img)
    img_array[:, :, 0] = base_color[0]
    img_array[:, :, 1] = base_color[1] 
    img_array[:, :, 2] = base_color[2]
    
    # Add organic variation patterns
    np.random.seed(42)  # For reproducible results
    
    # Layer 1: Large-scale mucosal patterns
    for i in range(8):
        # Create irregular shapes
        center_x = np.random.randint(0, size)
        center_y = np.random.randint(0, size)
        radius = np.random.randint(50, 150)
        
        # Create gradient circle
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = distance < radius
        gradient = np.exp(-distance / (radius * 0.3))
        
        # Blend mucosa color
        blend_factor = gradient * 0.3
        img_array[mask, 0] = img_array[mask, 0] * (1 - blend_factor[mask]) + mucosa_color[0] * blend_factor[mask]
        img_array[mask, 1] = img_array[mask, 1] * (1 - blend_factor[mask]) + mucosa_color[1] * blend_factor[mask]
        img_array[mask, 2] = img_array[mask, 2] * (1 - blend_factor[mask]) + mucosa_color[2] * blend_factor[mask]
    
    # Layer 2: Fine detail texture
    detail_noise = np.random.rand(size, size, 3) * 0.2 - 0.1  # -0.1 to 0.1
    img_array = img_array + detail_noise * 50  # Scale to color range
    
    # Layer 3: Vascular patterns (simulate blood vessels)
    for i in range(20):
        # Create line patterns
        start_x = np.random.randint(0, size)
        start_y = np.random.randint(0, size)
        end_x = start_x + np.random.randint(-100, 100)
        end_y = start_y + np.random.randint(-100, 100)
        
        # Draw line with varying intensity
        draw = ImageDraw.Draw(Image.fromarray(img_array.astype(np.uint8)))
        color = (int(detail_color[0]), int(detail_color[1]), int(detail_color[2]))
        width = np.random.randint(1, 3)
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
        img_array = np.array(draw._image)
    
    # Layer 4: Add some highlights for realism
    highlight_noise = np.random.rand(size, size) * 0.1
    highlight_mask = highlight_noise > 0.95  # 5% of pixels
    
    img_array[highlight_mask, 0] = np.minimum(255, img_array[highlight_mask, 0] + 30)
    img_array[highlight_mask, 1] = np.minimum(255, img_array[highlight_mask, 1] + 30)
    img_array[highlight_mask, 2] = np.minimum(255, img_array[highlight_mask, 2] + 30)
    
    # Apply slight blur for organic feel
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    return img


def apply_uv_mask_to_texture(texture: Image.Image, uv_mask: np.ndarray) -> Image.Image:
    """Apply UV mask to texture, making unmapped areas transparent/black."""
    texture_array = np.array(texture)
    mask_3d = uv_mask[:, :, np.newaxis]
    
    # Apply mask - keep texture where mask is 1, make black where mask is 0
    masked_texture = texture_array * mask_3d
    
    return Image.fromarray(masked_texture.astype(np.uint8))


def create_colon_texture_for_model(
    model_name: str,
    models_dir: Path = Path("output/models"),
    size: int = 1024,
    overwrite: bool = False
) -> bool:
    """
    Create a realistic colon texture for a specific model.
    """
    print(f"\n{'='*70}")
    print(f"Creating Realistic Colon Texture: {model_name}")
    print(f"{'='*70}")
    
    model_dir = models_dir / model_name
    
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
        uv_mask = create_uv_mask(uvs, indices, size)
        print(f"✓ UV mask created")
        
        # Generate realistic texture
        print("Generating realistic texture...")
        texture = create_realistic_colon_texture(size)
        
        # Apply UV mask
        print("Applying UV mask...")
        masked_texture = apply_uv_mask_to_texture(texture, uv_mask)
        
        # Save texture
        textures_dir = model_dir / 'textures'
        textures_dir.mkdir(exist_ok=True)
        
        # Save as diffuse.png (main texture)
        texture_path = textures_dir / 'diffuse.png'
        masked_texture.save(str(texture_path))
        print(f"✓ Saved texture: {texture_path}")
        
        # Also save a copy without mask for reference
        texture.save(str(textures_dir / 'base_texture.png'))
        
        # Update GLTF to ensure proper material setup
        print("Updating GLTF material...")
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        # Ensure material uses the texture with proper settings
        for material in gltf_data.get('materials', []):
            if 'pbrMetallicRoughness' in material:
                # Set to full brightness to ensure texture is visible
                material['pbrMetallicRoughness']['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
                material['pbrMetallicRoughness']['metallicFactor'] = 0.0
                material['pbrMetallicRoughness']['roughnessFactor'] = 0.8
        
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        print(f"✓ GLTF material updated")
        
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS: {model_name} realistic texture created!")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create realistic, visible colon texture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create realistic colon texture
    python create_realistic_colon_texture.py --model colon
    
    # High resolution
    python create_realistic_colon_texture.py --model colon --size 2048
    
    # Overwrite existing
    python create_realistic_colon_texture.py --model colon --overwrite
        """
    )
    
    parser.add_argument('--model', type=str, default='colon', help='Model name to process')
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
    print("║" + " "*15 + "Realistic Colon Texture Generator" + " "*25 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    success = create_colon_texture_for_model(
        args.model, 
        models_dir, 
        args.size, 
        args.overwrite
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
