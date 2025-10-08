#!/usr/bin/env python3
"""
create_uv_mask.py

A utility to create UV unwrap masks for FLUX.1 texture generation.
These masks show where the model's UV coordinates are mapped in texture space,
which can be used to guide AI texture generation.

Requirements
------------
- numpy
- Pillow
- scipy

Example
-------
    # Process all models
    python create_uv_mask.py
    
    # Process specific organ
    python create_uv_mask.py --organ colon
    
    # Custom size and overwrite existing
    python create_uv_mask.py --size 2048 --overwrite
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from typing import Tuple, Optional


def extract_gltf_uv_coordinates(gltf_path: str) -> np.ndarray:
    """
    Extract UV coordinates directly from GLTF file.
    
    Args:
        gltf_path: Path to GLTF file
    
    Returns:
        np.ndarray: UV coordinates (N x 2) or empty array if not found
    """
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        uvs = []
        
        # Look for TEXCOORD_0 accessor in meshes
        if 'meshes' in gltf_data:
            for mesh in gltf_data['meshes']:
                for primitive in mesh.get('primitives', []):
                    attributes = primitive.get('attributes', {})
                    
                    # Look for TEXCOORD_0 attribute
                    if 'TEXCOORD_0' in attributes:
                        texcoord_index = attributes['TEXCOORD_0']
                        accessor = gltf_data['accessors'][texcoord_index]
                        
                        # Load UV data from buffer
                        buffer_view_index = accessor['bufferView']
                        buffer_view = gltf_data['bufferViews'][buffer_view_index]
                        buffer_index = buffer_view['buffer']
                        buffer_info = gltf_data['buffers'][buffer_index]
                        
                        # Read buffer data
                        buffer_path = os.path.join(model_folder, buffer_info['uri'])
                        if not os.path.exists(buffer_path):
                            print(f"    ⚠️  Warning: Buffer file not found: {buffer_info['uri']}")
                            continue
                            
                        with open(buffer_path, 'rb') as f:
                            buffer_data = f.read()
                        
                        # Extract UV coordinates
                        byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                        uv_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 8]  # 2 floats * 4 bytes each
                        
                        # Convert to numpy array (assuming FLOAT32)
                        uv_array = np.frombuffer(uv_data, dtype=np.float32)
                        uv_array = uv_array.reshape(-1, 2)  # Each UV pair has 2 components
                        
                        uvs.extend(uv_array)
        
        return np.array(uvs) if uvs else np.array([])
    
    except Exception as e:
        print(f"    ✗ Error extracting UVs from GLTF: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def create_uv_mask(uvs: np.ndarray, size: int = 1024, mask_type: str = 'binary') -> Image.Image:
    """
    Create a UV unwrap mask for FLUX.1 texture generation.
    
    Args:
        uvs: UV coordinates array (N x 2)
        size: Output image size
        mask_type: Type of mask - 'binary', 'soft', or 'filled'
    
    Returns:
        PIL Image containing the UV mask
    """
    # Create base image (black background)
    mask = np.zeros((size, size), dtype=np.float32)
    
    if len(uvs) == 0:
        print("    ⚠️  No UV coordinates found, creating empty mask")
        return Image.fromarray((mask * 255).astype(np.uint8))
    
    print(f"    - Creating {mask_type} UV mask with {len(uvs)} coordinates...")
    
    # Convert UV coordinates to image coordinates
    # UV coordinates are in [0,1] range, with (0,0) at bottom-left
    # Image coordinates have (0,0) at top-left, so we need to flip Y
    uv_coords = uvs.copy()
    uv_coords[:, 1] = 1.0 - uv_coords[:, 1]  # Flip Y coordinate
    uv_coords = uv_coords * size
    uv_coords = np.clip(uv_coords, 0, size - 1).astype(int)
    
    # Mark UV coordinates on the mask
    for u, v in uv_coords:
        # Draw a small filled area around each UV point
        radius = 2
        y_start = max(0, v - radius)
        y_end = min(size, v + radius + 1)
        x_start = max(0, u - radius)
        x_end = min(size, u + radius + 1)
        
        mask[y_start:y_end, x_start:x_end] = 1.0
    
    if mask_type == 'filled':
        # Fill the regions between UV coordinates to create a solid mask
        from scipy import ndimage
        
        # Dilate the mask to fill gaps
        print("    - Filling UV regions...")
        mask = ndimage.binary_dilation(mask, iterations=15).astype(np.float32)
        
        # Apply morphological closing to fill holes
        mask = ndimage.binary_closing(mask, iterations=10).astype(np.float32)
        
    elif mask_type == 'soft':
        # Create a soft mask with gradient falloff
        from scipy import ndimage
        
        print("    - Creating soft edges...")
        # Dilate first to connect nearby points
        mask = ndimage.binary_dilation(mask, iterations=8).astype(np.float32)
        
        # Apply Gaussian blur for soft edges
        mask = ndimage.gaussian_filter(mask, sigma=3.0)
        
    # Convert to PIL Image
    mask_uint8 = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8).convert('L')
    
    return img


def create_detailed_uv_mask(uvs: np.ndarray, size: int = 1024) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Create multiple UV mask variants for different FLUX.1 use cases.
    
    Args:
        uvs: UV coordinates array (N x 2)
        size: Output image size
    
    Returns:
        Tuple of (binary_mask, soft_mask, filled_mask) PIL Images
    """
    binary_mask = create_uv_mask(uvs, size, mask_type='binary')
    soft_mask = create_uv_mask(uvs, size, mask_type='soft')
    filled_mask = create_uv_mask(uvs, size, mask_type='filled')
    
    return binary_mask, soft_mask, filled_mask


def process_model_for_uv_mask(
    model_dir: str, 
    size: int = 1024, 
    overwrite: bool = False,
    create_variants: bool = False
) -> bool:
    """
    Process a single model to create UV unwrap mask(s).
    
    Args:
        model_dir: Path to model directory
        size: Texture size
        overwrite: Whether to overwrite existing masks
        create_variants: Whether to create all mask variants
    
    Returns:
        True if successful
    """
    model_name = os.path.basename(model_dir)
    
    print(f"\n[*] Processing '{model_name}' for UV mask...")
    
    # Check if mask already exists
    mask_path = os.path.join(model_dir, "uv_mask.png")
    if os.path.exists(mask_path) and not overwrite:
        print(f"    ✓ UV mask already exists: {mask_path}")
        if not create_variants:
            return True
    
    try:
        # Find GLTF file
        gltf_path = os.path.join(model_dir, "scene.gltf")
        if not os.path.exists(gltf_path):
            print(f"    ✗ No scene.gltf found in {model_dir}")
            return False
        
        # Extract UV coordinates
        print("    - Extracting UV coordinates from GLTF...")
        uvs = extract_gltf_uv_coordinates(gltf_path)
        
        if len(uvs) == 0:
            print("    ⚠️  No UV coordinates found")
        else:
            print(f"    ✓ Found {len(uvs)} UV coordinates")
            
            # Calculate UV coverage statistics
            uv_min = uvs.min(axis=0)
            uv_max = uvs.max(axis=0)
            uv_range = uv_max - uv_min
            print(f"    - UV range: X[{uv_min[0]:.3f}, {uv_max[0]:.3f}] Y[{uv_min[1]:.3f}, {uv_max[1]:.3f}]")
            print(f"    - UV coverage: {uv_range[0]*100:.1f}% x {uv_range[1]*100:.1f}%")
        
        if create_variants:
            # Create all three mask variants
            print("    - Creating mask variants...")
            binary_mask, soft_mask, filled_mask = create_detailed_uv_mask(uvs, size)
            
            # Save all variants
            binary_path = os.path.join(model_dir, "uv_mask_binary.png")
            soft_path = os.path.join(model_dir, "uv_mask_soft.png")
            filled_path = os.path.join(model_dir, "uv_mask_filled.png")
            
            binary_mask.save(binary_path)
            soft_mask.save(soft_path)
            filled_mask.save(filled_path)
            
            print(f"    ✓ Saved binary mask: {binary_path}")
            print(f"    ✓ Saved soft mask: {soft_path}")
            print(f"    ✓ Saved filled mask: {filled_path}")
            
            # Also save the filled one as default
            filled_mask.save(mask_path)
            print(f"    ✓ Saved default mask: {mask_path}")
        else:
            # Create filled mask (best for FLUX.1 use)
            mask = create_uv_mask(uvs, size, mask_type='filled')
            
            # Save mask
            mask.save(mask_path)
            print(f"    ✓ Saved UV mask: {mask_path}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all models or specific organ."""
    parser = argparse.ArgumentParser(
        description="Create UV unwrap masks for FLUX.1 texture generation"
    )
    parser.add_argument(
        "--organ", 
        help="Specific organ to process (e.g., 'colon', 'heart')"
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=1024, 
        help="Mask size in pixels (default: 1024)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing masks"
    )
    parser.add_argument(
        "--variants", 
        action="store_true", 
        help="Create all mask variants (binary, soft, filled)"
    )
    parser.add_argument(
        "--models-dir",
        default="output/models",
        help="Directory containing model folders (default: output/models)"
    )
    
    args = parser.parse_args()
    
    models_dir = args.models_dir
    
    if not os.path.exists(models_dir):
        print(f"✗ Error: Models directory '{models_dir}' not found")
        return 1
    
    print("=" * 60)
    print("UV Mask Generator for FLUX.1 Texture Generation")
    print("=" * 60)
    
    if args.organ:
        # Process specific organ
        organ_dir = os.path.join(models_dir, args.organ)
        if not os.path.exists(organ_dir):
            print(f"✗ Error: Organ directory '{organ_dir}' not found")
            return 1
        
        success = process_model_for_uv_mask(
            organ_dir, 
            args.size, 
            args.overwrite,
            args.variants
        )
        return 0 if success else 1
    else:
        # Process all models
        model_dirs = [
            d for d in os.listdir(models_dir) 
            if os.path.isdir(os.path.join(models_dir, d))
        ]
        
        if not model_dirs:
            print(f"✗ No model directories found in '{models_dir}'")
            return 1
        
        print(f"\nFound {len(model_dirs)} model directories to process")
        if args.variants:
            print("Creating all mask variants (binary, soft, filled)")
        
        successful = 0
        failed = 0
        
        for model_dir in sorted(model_dirs):
            model_path = os.path.join(models_dir, model_dir)
            if process_model_for_uv_mask(
                model_path, 
                args.size, 
                args.overwrite,
                args.variants
            ):
                successful += 1
            else:
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"Summary: {successful} successful, {failed} failed")
        print("=" * 60)
        
        if successful > 0:
            print(f"\n✓ UV masks saved in each model directory:")
            print(f"  - uv_mask.png (default filled mask for FLUX.1)")
            if args.variants:
                print(f"  - uv_mask_binary.png (sparse point mask)")
                print(f"  - uv_mask_soft.png (soft gradient mask)")
                print(f"  - uv_mask_filled.png (filled region mask)")
        
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

