#!/usr/bin/env python3
"""
create_uv_mask_improved.py

Improved UV mask generator that properly rasterizes UV triangles.
This creates accurate masks for FLUX texture generation without gaps or repeating patterns.

Usage:
    python create_uv_mask_improved.py --organ colon
    python create_uv_mask_improved.py --all --size 2048
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Tuple, Optional


def extract_uv_data_from_gltf(gltf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract UV coordinates and face indices from GLTF file.
    
    Args:
        gltf_path: Path to GLTF file
    
    Returns:
        Tuple of (uvs array [N x 2], indices array [M x 3])
    """
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        all_uvs = []
        all_indices = []
        
        # Process each mesh
        for mesh in gltf_data.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                attributes = primitive.get('attributes', {})
                
                # Extract UV coordinates
                if 'TEXCOORD_0' in attributes:
                    texcoord_index = attributes['TEXCOORD_0']
                    accessor = gltf_data['accessors'][texcoord_index]
                    buffer_view_index = accessor['bufferView']
                    buffer_view = gltf_data['bufferViews'][buffer_view_index]
                    buffer_index = buffer_view['buffer']
                    buffer_info = gltf_data['buffers'][buffer_index]
                    
                    buffer_path = os.path.join(model_folder, buffer_info['uri'])
                    if not os.path.exists(buffer_path):
                        print(f"⚠ Warning: Buffer file not found: {buffer_info['uri']}")
                        continue
                    
                    with open(buffer_path, 'rb') as f:
                        buffer_data = f.read()
                    
                    byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                    uv_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 8]
                    uv_array = np.frombuffer(uv_data, dtype=np.float32).reshape(-1, 2)
                    
                    # Extract face indices
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
                        
                        # Handle different index types
                        component_type = accessor['componentType']
                        if component_type == 5125:  # UNSIGNED_INT
                            dtype = np.uint32
                            bytes_per_index = 4
                        elif component_type == 5123:  # UNSIGNED_SHORT
                            dtype = np.uint16
                            bytes_per_index = 2
                        else:
                            print(f"⚠ Warning: Unsupported index type: {component_type}")
                            continue
                        
                        indices_data = buffer_data[byte_offset:byte_offset + accessor['count'] * bytes_per_index]
                        indices = np.frombuffer(indices_data, dtype=dtype)
                        
                        # Store with offset for current UV batch
                        offset = len(all_uvs)
                        all_uvs.extend(uv_array)
                        all_indices.extend((indices.reshape(-1, 3) + offset).tolist())
        
        uvs = np.array(all_uvs)
        indices = np.array(all_indices, dtype=np.uint32)
        
        return uvs, indices
    
    except Exception as e:
        print(f"✗ Error extracting UV data from GLTF: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([])


def normalize_uvs(uvs: np.ndarray, padding: float = 0.02) -> Tuple[np.ndarray, dict]:
    """
    Normalize UV coordinates to use full 0-1 range with optional padding.
    
    Args:
        uvs: UV coordinates array [N x 2]
        padding: Padding around edges (0.02 = 2% on each side)
    
    Returns:
        Tuple of (normalized UVs, transform info dict)
    """
    if len(uvs) == 0:
        return uvs, {}
    
    # Get current bounds
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    
    # Avoid division by zero
    uv_range = np.where(uv_range < 1e-6, 1.0, uv_range)
    
    # Normalize to 0-1 range
    uvs_normalized = (uvs - uv_min) / uv_range
    
    # Apply padding
    if padding > 0:
        scale = 1.0 - 2 * padding
        uvs_normalized = uvs_normalized * scale + padding
    
    transform_info = {
        'original_min': uv_min.tolist(),
        'original_max': uv_max.tolist(),
        'original_range': uv_range.tolist(),
        'padding': padding,
        'normalized': True
    }
    
    return uvs_normalized, transform_info


def create_uv_mask_from_triangles(
    uvs: np.ndarray,
    indices: np.ndarray,
    size: int = 1024,
    antialias: bool = True,
    supersample: int = 2
) -> Image.Image:
    """
    Create UV mask by properly rasterizing UV triangles.
    
    Args:
        uvs: UV coordinates array [N x 2]
        indices: Triangle indices array [M x 3]
        size: Output image size
        antialias: Whether to use supersampling for antialiasing
        supersample: Supersampling factor (2 = 2x, 4 = 4x)
    
    Returns:
        PIL Image containing the UV mask (RGBA)
    """
    if len(uvs) == 0 or len(indices) == 0:
        print("⚠ Warning: No UV data, creating empty mask")
        return Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    # Use supersampling for antialiasing
    render_size = size * supersample if antialias else size
    
    # Create RGBA image for mask
    mask_img = Image.new('RGBA', (render_size, render_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_img)
    
    print(f"  Rasterizing {len(indices)} triangles...")
    
    # Convert UV coordinates to image space
    # UV (0,0) is bottom-left, image (0,0) is top-left
    uv_coords = uvs.copy()
    uv_coords[:, 1] = 1.0 - uv_coords[:, 1]  # Flip Y
    uv_coords = np.clip(uv_coords, 0.0, 1.0) * (render_size - 1)
    
    # Draw each triangle
    triangles_drawn = 0
    for i, face_indices in enumerate(indices):
        if i % 10000 == 0 and i > 0:
            print(f"  Progress: {i}/{len(indices)} triangles...")
        
        try:
            # Get triangle UV coordinates
            tri_uvs = uv_coords[face_indices]
            
            # Convert to pixel coordinates
            pixels = [(int(uv[0]), int(uv[1])) for uv in tri_uvs]
            
            # Draw filled triangle
            draw.polygon(pixels, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
            triangles_drawn += 1
            
        except Exception as e:
            if i < 10:  # Only print first few errors
                print(f"  ⚠ Warning: Failed to draw triangle {i}: {e}")
    
    print(f"  Successfully drew {triangles_drawn}/{len(indices)} triangles")
    
    # Downsample if we used supersampling
    if antialias and supersample > 1:
        mask_img = mask_img.resize((size, size), Image.Resampling.LANCZOS)
    
    return mask_img


def create_dilated_mask(mask_rgba: Image.Image, dilation_pixels: int = 4) -> Image.Image:
    """
    Create dilated version of mask to help with texture bleeding at seams.
    
    Args:
        mask_rgba: Original RGBA mask
        dilation_pixels: Number of pixels to dilate
    
    Returns:
        Dilated RGBA mask
    """
    from scipy.ndimage import binary_dilation
    
    # Convert to numpy
    mask_array = np.array(mask_rgba)
    alpha_channel = mask_array[:, :, 3] > 127
    
    # Dilate the alpha channel
    structure = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(alpha_channel, structure=structure, iterations=dilation_pixels)
    
    # Create new image
    dilated_array = mask_array.copy()
    dilated_array[:, :, 3] = (dilated * 255).astype(np.uint8)
    
    return Image.fromarray(dilated_array)


def analyze_uv_coverage(uvs: np.ndarray, indices: np.ndarray, size: int = 1024) -> dict:
    """
    Analyze UV layout to detect issues.
    
    Args:
        uvs: UV coordinates array
        indices: Triangle indices
        size: Texture size for coverage calculation
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    if len(uvs) == 0:
        return {'error': 'No UV data'}
    
    # UV bounds
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    
    analysis['uv_min'] = uv_min.tolist()
    analysis['uv_max'] = uv_max.tolist()
    analysis['uv_range'] = uv_range.tolist()
    
    # Check for UVs outside 0-1 range
    outside_range = np.sum((uvs < 0.0) | (uvs > 1.0))
    analysis['uvs_outside_01_range'] = int(outside_range)
    analysis['uvs_outside_01_percent'] = float(outside_range) / len(uvs) * 100
    
    # UV space utilization
    analysis['u_utilization_percent'] = float(uv_range[0] * 100)
    analysis['v_utilization_percent'] = float(uv_range[1] * 100)
    analysis['avg_utilization_percent'] = float(np.mean(uv_range) * 100)
    
    # Check for potential repeating UVs (multiple vertices at same UV)
    unique_uvs = np.unique(uvs, axis=0)
    analysis['total_uvs'] = len(uvs)
    analysis['unique_uvs'] = len(unique_uvs)
    analysis['duplicate_uv_percent'] = float((len(uvs) - len(unique_uvs)) / len(uvs) * 100)
    
    return analysis


def process_model_for_improved_uv_mask(
    model_dir: str,
    size: int = 1024,
    overwrite: bool = False,
    normalize: bool = True,
    supersample: int = 2,
    create_variants: bool = False
) -> bool:
    """
    Process a model to create improved UV masks.
    
    Args:
        model_dir: Path to model directory
        size: Texture size
        overwrite: Whether to overwrite existing masks
        normalize: Whether to normalize UVs to 0-1 range
        supersample: Supersampling factor for antialiasing
        create_variants: Create additional mask variants
    
    Returns:
        True if successful
    """
    model_name = os.path.basename(model_dir)
    
    print(f"\n{'='*70}")
    print(f"Processing: {model_name}")
    print(f"{'='*70}")
    
    # Check if mask exists
    mask_path = os.path.join(model_dir, "uv_mask_rgba.png")
    if os.path.exists(mask_path) and not overwrite:
        print(f"✓ UV mask already exists (use --overwrite to regenerate)")
        return True
    
    try:
        # Find GLTF file
        gltf_path = os.path.join(model_dir, "scene.gltf")
        if not os.path.exists(gltf_path):
            print(f"✗ No scene.gltf found")
            return False
        
        # Extract UV data
        print("Extracting UV data from GLTF...")
        uvs, indices = extract_uv_data_from_gltf(gltf_path)
        
        if len(uvs) == 0 or len(indices) == 0:
            print("✗ No UV data found in GLTF")
            return False
        
        print(f"✓ Extracted {len(uvs)} UVs and {len(indices)} triangles")
        
        # Analyze UV layout
        print("\nAnalyzing UV layout...")
        analysis = analyze_uv_coverage(uvs, indices, size)
        
        print(f"  UV Range: U[{analysis['uv_min'][0]:.3f}, {analysis['uv_max'][0]:.3f}] "
              f"V[{analysis['uv_min'][1]:.3f}, {analysis['uv_max'][1]:.3f}]")
        print(f"  Utilization: U={analysis['u_utilization_percent']:.1f}% "
              f"V={analysis['v_utilization_percent']:.1f}%")
        
        if analysis['uvs_outside_01_percent'] > 0:
            print(f"  ⚠ Warning: {analysis['uvs_outside_01_percent']:.1f}% of UVs outside 0-1 range")
        
        # Normalize UVs if requested
        if normalize and (analysis['avg_utilization_percent'] < 95.0 or 
                         analysis['uvs_outside_01_percent'] > 0):
            print("\nNormalizing UVs to full 0-1 range...")
            uvs_normalized, transform_info = normalize_uvs(uvs, padding=0.02)
            
            # Save transform info
            transform_path = os.path.join(model_dir, "uv_transform.json")
            with open(transform_path, 'w') as f:
                json.dump(transform_info, f, indent=2)
            
            print(f"✓ UVs normalized (transform saved to uv_transform.json)")
            uvs = uvs_normalized
        
        # Create UV mask from triangles
        print(f"\nCreating UV mask ({size}x{size}, {supersample}x supersampling)...")
        mask_rgba = create_uv_mask_from_triangles(uvs, indices, size, antialias=True, supersample=supersample)
        
        # Calculate coverage
        mask_array = np.array(mask_rgba)
        coverage = np.sum(mask_array[:, :, 3] > 0) / (size * size) * 100
        print(f"✓ UV mask coverage: {coverage:.1f}%")
        
        # Save main mask
        mask_rgba.save(mask_path)
        print(f"✓ Saved: {mask_path}")
        
        # Also save as RGB version for compatibility
        mask_rgb = mask_rgba.convert('RGB')
        mask_rgb.save(os.path.join(model_dir, "uv_mask.png"))
        
        # Create dilated version for texture bleeding prevention
        if create_variants:
            print("\nCreating mask variants...")
            
            dilated_mask = create_dilated_mask(mask_rgba, dilation_pixels=4)
            dilated_path = os.path.join(model_dir, "uv_mask_dilated.png")
            dilated_mask.save(dilated_path)
            print(f"✓ Saved dilated mask: {dilated_path}")
        
        # Save analysis
        analysis_path = os.path.join(model_dir, "uv_analysis.json")
        analysis['coverage_percent'] = float(coverage)
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Saved analysis: {analysis_path}")
        
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS: {model_name}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create improved UV masks with proper triangle rasterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single model
    python create_uv_mask_improved.py --organ colon
    
    # Process all models with high resolution
    python create_uv_mask_improved.py --all --size 2048
    
    # Process with variants and overwrite existing
    python create_uv_mask_improved.py --organ heart --variants --overwrite
    
    # Disable UV normalization
    python create_uv_mask_improved.py --organ liver --no-normalize
        """
    )
    
    parser.add_argument('--organ', type=str, help='Specific organ to process')
    parser.add_argument('--all', action='store_true', help='Process all models')
    parser.add_argument('--size', type=int, default=1024, choices=[512, 1024, 2048, 4096],
                       help='Mask size (default: 1024)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Skip UV normalization')
    parser.add_argument('--supersample', type=int, default=2, choices=[1, 2, 4],
                       help='Supersampling factor (default: 2)')
    parser.add_argument('--variants', action='store_true', help='Create mask variants')
    parser.add_argument('--models-dir', type=str, default='output/models',
                       help='Models directory (default: output/models)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"✗ Error: Models directory not found: {models_dir}")
        return 1
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "Improved UV Mask Generator" + " "*27 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    if args.organ:
        # Process single model
        model_dir = models_dir / args.organ
        if not model_dir.exists():
            print(f"✗ Error: Model directory not found: {model_dir}")
            return 1
        
        success = process_model_for_improved_uv_mask(
            str(model_dir),
            size=args.size,
            overwrite=args.overwrite,
            normalize=args.normalize,
            supersample=args.supersample,
            create_variants=args.variants
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
            if process_model_for_improved_uv_mask(
                str(model_dir),
                size=args.size,
                overwrite=args.overwrite,
                normalize=args.normalize,
                supersample=args.supersample,
                create_variants=args.variants
            ):
                successful += 1
            else:
                failed += 1
        
        print("\n" + "="*70)
        print(f"SUMMARY: {successful}/{len(model_dirs)} successful")
        if failed > 0:
            print(f"Failed: {failed}")
        print("="*70)
        
        return 0 if failed == 0 else 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

