#!/usr/bin/env python3
"""
create_perfect_uv_mask.py

Create a perfect UV mask that exactly matches the object geometry.
This ensures the texture maps seamlessly without any blotchy or misaligned areas.

Usage:
    python create_perfect_uv_mask.py --model colon
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Tuple
import trimesh


def load_uv_data_from_gltf(gltf_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load UV coordinates, face indices, and vertex positions from GLTF."""
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        all_uvs = []
        all_indices = []
        all_vertices = []
        
        for mesh in gltf_data.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                attributes = primitive.get('attributes', {})
                
                # Get vertex positions
                if 'POSITION' in attributes:
                    position_index = attributes['POSITION']
                    accessor = gltf_data['accessors'][position_index]
                    buffer_view_index = accessor['bufferView']
                    buffer_view = gltf_data['bufferViews'][buffer_view_index]
                    buffer_index = buffer_view['buffer']
                    buffer_info = gltf_data['buffers'][buffer_index]
                    
                    buffer_path = os.path.join(model_folder, buffer_info['uri'])
                    with open(buffer_path, 'rb') as f:
                        buffer_data = f.read()
                    
                    byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                    vertex_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 12]  # 3 floats * 4 bytes
                    vertex_array = np.frombuffer(vertex_data, dtype=np.float32).reshape(-1, 3)
                    all_vertices.extend(vertex_array)
                
                # Get UV coordinates
                if 'TEXCOORD_0' in attributes:
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
                    uv_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 8]  # 2 floats * 4 bytes
                    uv_array = np.frombuffer(uv_data, dtype=np.float32).reshape(-1, 2)
                    all_uvs.extend(uv_array)
                    
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
                        indices_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 4]  # uint32
                        indices = np.frombuffer(indices_data, dtype=np.uint32)
                        
                        offset = len(all_uvs)
                        all_indices.extend((indices.reshape(-1, 3) + offset).tolist())
        
        return np.array(all_vertices), np.array(all_uvs), np.array(all_indices, dtype=np.uint32)
    
    except Exception as e:
        print(f"✗ Error loading UV data: {e}")
        return np.array([]), np.array([]), np.array([])


def create_perfect_uv_mask(
    vertices: np.ndarray, 
    uvs: np.ndarray, 
    indices: np.ndarray, 
    size: int = 2048,
    anti_aliasing: bool = True
) -> Image.Image:
    """
    Create a perfect UV mask that exactly matches the object geometry.
    """
    print(f"  Creating perfect UV mask ({size}x{size})...")
    
    # Create high-resolution mask with anti-aliasing
    if anti_aliasing:
        render_size = size * 2  # 2x supersampling
    else:
        render_size = size
    
    # Create RGBA mask
    mask = np.zeros((render_size, render_size), dtype=np.float32)
    draw = ImageDraw.Draw(Image.fromarray(mask))
    
    # Convert UV coordinates to image coordinates
    uv_coords = uvs.copy()
    uv_coords[:, 1] = 1.0 - uv_coords[:, 1]  # Flip Y coordinate
    uv_coords = np.clip(uv_coords, 0.0, 1.0) * (render_size - 1)
    
    print(f"    Rasterizing {len(indices)} triangles at {render_size}x{render_size}...")
    
    # Draw each triangle with proper anti-aliasing
    for i, face_indices in enumerate(indices.reshape(-1, 3)):
        if i % 10000 == 0 and i > 0:
            print(f"    Progress: {i}/{len(indices)} triangles...")
        
        try:
            # Get triangle UV coordinates
            tri_uvs = uv_coords[face_indices]
            
            # Convert to pixel coordinates with sub-pixel precision
            pixels = [(float(uv[0]), float(uv[1])) for uv in tri_uvs]
            
            # Draw filled triangle
            draw.polygon(pixels, fill=1.0, outline=1.0)
            
        except Exception as e:
            if i < 10:  # Only print first few errors
                print(f"    ⚠ Warning: Failed to draw triangle {i}: {e}")
    
    # Downsample with anti-aliasing if requested
    if anti_aliasing:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((size, size), Image.Resampling.LANCZOS)
        mask = np.array(mask_img).astype(np.float32) / 255.0
    
    # Convert to RGBA for better texture blending
    rgba_mask = np.zeros((size, size, 4), dtype=np.uint8)
    rgba_mask[:, :, 3] = (mask * 255).astype(np.uint8)  # Alpha channel
    rgba_mask[:, :, :3] = 255  # RGB channels (white)
    
    return Image.fromarray(rgba_mask, 'RGBA')


def create_distance_based_mask(
    vertices: np.ndarray,
    uvs: np.ndarray, 
    indices: np.ndarray,
    size: int = 2048
) -> Image.Image:
    """
    Create a mask with distance-based blending to eliminate seams.
    """
    print(f"  Creating distance-based mask for seamless blending...")
    
    # First create the base mask
    base_mask = create_perfect_uv_mask(vertices, uvs, indices, size, anti_aliasing=True)
    mask_array = np.array(base_mask)[:, :, 3].astype(np.float32) / 255.0
    
    # Create distance transform from boundaries
    from scipy.ndimage import distance_transform_edt
    
    # Invert mask for distance transform
    inverted_mask = 1.0 - mask_array
    
    # Calculate distance from boundaries
    distance = distance_transform_edt(inverted_mask)
    
    # Normalize distance
    if distance.max() > 0:
        distance = distance / distance.max()
    
    # Create gradient mask for seamless blending
    # Areas near boundaries get lower alpha for blending
    gradient_mask = np.zeros((size, size, 4), dtype=np.uint8)
    
    # Create smooth falloff near edges
    edge_distance = 8  # pixels from edge
    blend_factor = np.clip(distance / edge_distance, 0, 1)
    
    # Apply gradient to alpha channel
    gradient_mask[:, :, 3] = (blend_factor * 255).astype(np.uint8)
    gradient_mask[:, :, :3] = 255  # White RGB
    
    return Image.fromarray(gradient_mask, 'RGBA')


def analyze_uv_coverage(uvs: np.ndarray, indices: np.ndarray, size: int = 2048) -> dict:
    """Analyze UV layout for optimization."""
    print(f"  Analyzing UV coverage...")
    
    # Create temporary mask for analysis
    temp_mask = create_perfect_uv_mask(np.array([]), uvs, indices, size, anti_aliasing=False)
    mask_array = np.array(temp_mask)[:, :, 3]
    
    # Calculate statistics
    total_pixels = size * size
    covered_pixels = np.sum(mask_array > 0)
    coverage_percent = (covered_pixels / total_pixels) * 100
    
    # Analyze UV distribution
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    
    analysis = {
        'total_pixels': int(total_pixels),
        'covered_pixels': int(covered_pixels),
        'coverage_percent': float(coverage_percent),
        'uv_min': uv_min.tolist(),
        'uv_max': uv_max.tolist(),
        'uv_range': uv_range.tolist(),
        'uv_utilization_u': float(uv_range[0] * 100),
        'uv_utilization_v': float(uv_range[1] * 100),
        'avg_utilization': float(np.mean(uv_range) * 100)
    }
    
    print(f"    Coverage: {coverage_percent:.1f}%")
    print(f"    UV utilization: U={analysis['uv_utilization_u']:.1f}%, V={analysis['uv_utilization_v']:.1f}%")
    
    return analysis


def create_perfect_uv_mask_for_model(
    model_name: str,
    models_dir: Path = Path("output/models"),
    size: int = 2048,
    overwrite: bool = False
) -> bool:
    """
    Create perfect UV masks for a specific model.
    """
    print(f"\n{'='*70}")
    print(f"Creating Perfect UV Mask: {model_name}")
    print(f"{'='*70}")
    
    model_dir = models_dir / model_name
    
    try:
        # Load UV and vertex data
        gltf_path = model_dir / 'scene.gltf'
        if not gltf_path.exists():
            print(f"✗ GLTF file not found: {gltf_path}")
            return False
        
        print("Loading UV and vertex data...")
        vertices, uvs, indices = load_uv_data_from_gltf(str(gltf_path))
        
        if len(uvs) == 0 or len(indices) == 0:
            print(f"✗ No UV data found")
            return False
        
        print(f"✓ Loaded {len(vertices)} vertices, {len(uvs)} UVs, {len(indices)} triangles")
        
        # Analyze UV coverage
        analysis = analyze_uv_coverage(uvs, indices, size)
        
        # Create perfect UV mask
        print("Creating perfect UV mask...")
        perfect_mask = create_perfect_uv_mask(vertices, uvs, indices, size, anti_aliasing=True)
        
        # Create distance-based mask for seamless blending
        print("Creating seamless blending mask...")
        seamless_mask = create_distance_based_mask(vertices, uvs, indices, size)
        
        # Save masks
        print("Saving UV masks...")
        
        # Save perfect mask
        perfect_mask.save(str(model_dir / 'uv_mask_perfect.png'))
        perfect_mask.convert('RGB').save(str(model_dir / 'uv_mask_perfect_rgb.png'))
        print(f"✓ Saved perfect UV mask: uv_mask_perfect.png")
        
        # Save seamless mask
        seamless_mask.save(str(model_dir / 'uv_mask_seamless.png'))
        seamless_mask.convert('RGB').save(str(model_dir / 'uv_mask_seamless_rgb.png'))
        print(f"✓ Saved seamless UV mask: uv_mask_seamless.png")
        
        # Save analysis
        analysis_path = model_dir / 'uv_perfect_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Saved analysis: uv_perfect_analysis.json")
        
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS: {model_name} perfect UV masks created!")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create perfect UV masks that exactly match object geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create perfect UV mask for colon
    python create_perfect_uv_mask.py --model colon
    
    # High resolution
    python create_perfect_uv_mask.py --model colon --size 4096
    
    # Overwrite existing
    python create_perfect_uv_mask.py --model colon --overwrite
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
    print("║" + " "*15 + "Perfect UV Mask Generator" + " "*30 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    success = create_perfect_uv_mask_for_model(
        args.model, 
        models_dir, 
        args.size, 
        args.overwrite
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
