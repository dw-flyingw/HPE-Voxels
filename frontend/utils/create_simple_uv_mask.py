#!/usr/bin/env python3
"""
create_simple_uv_mask.py

Create a simple, correct UV mask directly from GLTF data without complex processing.
This ensures the mask perfectly matches the object geometry.

Usage:
    python create_simple_uv_mask.py --model colon
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def create_simple_uv_mask_from_gltf(gltf_path: str, size: int = 2048) -> Image.Image:
    """
    Create a simple UV mask directly from GLTF data.
    """
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        
        # Create mask image
        mask = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        total_triangles = 0
        
        for mesh in gltf_data.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                attributes = primitive.get('attributes', {})
                
                if 'TEXCOORD_0' not in attributes or 'indices' not in primitive:
                    continue
                
                # Load UV coordinates
                uv_index = attributes['TEXCOORD_0']
                uv_accessor = gltf_data['accessors'][uv_index]
                uv_buffer_view = gltf_data['bufferViews'][uv_accessor['bufferView']]
                uv_buffer = gltf_data['buffers'][uv_buffer_view['buffer']]
                
                uv_path = os.path.join(model_folder, uv_buffer['uri'])
                with open(uv_path, 'rb') as f:
                    uv_data = f.read()
                
                uv_offset = uv_buffer_view.get('byteOffset', 0) + uv_accessor.get('byteOffset', 0)
                texcoord_data = uv_data[uv_offset:uv_offset + uv_accessor['count'] * 8]
                uvs = np.frombuffer(texcoord_data, dtype=np.float32).reshape(-1, 2)
                
                # Load face indices
                indices_accessor_idx = primitive['indices']
                indices_accessor = gltf_data['accessors'][indices_accessor_idx]
                indices_buffer_view = gltf_data['bufferViews'][indices_accessor['bufferView']]
                indices_buffer = gltf_data['buffers'][indices_buffer_view['buffer']]
                
                indices_path = os.path.join(model_folder, indices_buffer['uri'])
                with open(indices_path, 'rb') as f:
                    indices_data = f.read()
                
                indices_offset = indices_buffer_view.get('byteOffset', 0) + indices_accessor.get('byteOffset', 0)
                face_data = indices_data[indices_offset:indices_offset + indices_accessor['count'] * 4]
                faces = np.frombuffer(face_data, dtype=np.uint32).reshape(-1, 3)
                
                print(f"    Processing {len(faces)} triangles...")
                
                # Draw each triangle
                for i, face in enumerate(faces):
                    if i % 10000 == 0 and i > 0:
                        print(f"      Progress: {i}/{len(faces)} triangles...")
                    
                    try:
                        # Get UV coordinates for this triangle
                        tri_uvs = uvs[face]
                        
                        # Convert to image coordinates (flip Y)
                        pixels = []
                        for uv in tri_uvs:
                            x = int(uv[0] * size)
                            y = int((1.0 - uv[1]) * size)  # Flip Y
                            x = max(0, min(size-1, x))
                            y = max(0, min(size-1, y))
                            pixels.append((x, y))
                        
                        # Draw triangle
                        draw.polygon(pixels, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
                        total_triangles += 1
                        
                    except Exception as e:
                        if i < 10:  # Only print first few errors
                            print(f"      Warning: Failed to draw triangle {i}: {e}")
        
        print(f"    Drew {total_triangles} triangles")
        return mask
        
    except Exception as e:
        print(f"    Error creating UV mask: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_simple_uv_mask_for_model(
    model_name: str,
    models_dir: Path = Path("output/models"),
    size: int = 2048,
    overwrite: bool = False
) -> bool:
    """
    Create simple UV mask for a specific model.
    """
    print(f"\n{'='*70}")
    print(f"Creating Simple UV Mask: {model_name}")
    print(f"{'='*70}")
    
    model_dir = models_dir / model_name
    
    try:
        # Find GLTF file
        gltf_path = model_dir / 'scene.gltf'
        if not gltf_path.exists():
            print(f"✗ GLTF file not found: {gltf_path}")
            return False
        
        print(f"Found GLTF: {gltf_path}")
        
        # Create UV mask
        print("Creating simple UV mask...")
        mask = create_simple_uv_mask_from_gltf(str(gltf_path), size)
        
        if mask is None:
            print(f"✗ Failed to create UV mask")
            return False
        
        # Save mask
        mask_path = model_dir / 'uv_mask_simple.png'
        mask.save(str(mask_path))
        print(f"✓ Saved simple UV mask: uv_mask_simple.png")
        
        # Also save RGB version
        rgb_mask = mask.convert('RGB')
        rgb_path = model_dir / 'uv_mask_simple_rgb.png'
        rgb_mask.save(str(rgb_path))
        print(f"✓ Saved RGB version: uv_mask_simple_rgb.png")
        
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS: {model_name} simple UV mask created!")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create simple UV mask directly from GLTF data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create simple UV mask for colon
    python create_simple_uv_mask.py --model colon
    
    # High resolution
    python create_simple_uv_mask.py --model colon --size 4096
    
    # Overwrite existing
    python create_simple_uv_mask.py --model colon --overwrite
        """
    )
    
    parser.add_argument('--model', type=str, default='colon', help='Model name to process')
    parser.add_argument('--size', type=int, default=2048, choices=[1024, 2048, 4096],
                       help='Mask size (default: 2048)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
    parser.add_argument('--models-dir', type=str, default='output/models',
                       help='Models directory (default: output/models)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return 1
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*17 + "Simple UV Mask Creator" + " "*36 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    success = create_simple_uv_mask_for_model(
        args.model, 
        models_dir, 
        args.size, 
        args.overwrite
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
