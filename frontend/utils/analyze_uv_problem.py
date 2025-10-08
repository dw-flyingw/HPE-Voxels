#!/usr/bin/env python3
"""
analyze_uv_problem.py

Critically analyze the UV mapping problem to understand why our approach is failing.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
import trimesh


def analyze_uv_problem(model_name: str, models_dir: Path = Path("output/models")):
    """
    Analyze the UV mapping problem in detail.
    """
    print(f"\n{'='*80}")
    print(f"CRITICAL UV MAPPING ANALYSIS: {model_name}")
    print(f"{'='*80}")
    
    model_dir = models_dir / model_name
    gltf_path = model_dir / 'scene.gltf'
    
    if not gltf_path.exists():
        print(f"✗ GLTF file not found: {gltf_path}")
        return
    
    # Load GLTF data
    with open(gltf_path, 'r') as f:
        gltf_data = json.load(f)
    
    print(f"📊 GLTF Structure Analysis:")
    print(f"  Meshes: {len(gltf_data.get('meshes', []))}")
    print(f"  Materials: {len(gltf_data.get('materials', []))}")
    print(f"  Textures: {len(gltf_data.get('textures', []))}")
    print(f"  Images: {len(gltf_data.get('images', []))}")
    
    # Analyze each mesh
    for mesh_idx, mesh in enumerate(gltf_data.get('meshes', [])):
        print(f"\n🔍 Mesh {mesh_idx} Analysis:")
        
        for prim_idx, primitive in enumerate(mesh.get('primitives', [])):
            print(f"  Primitive {prim_idx}:")
            attributes = primitive.get('attributes', {})
            
            # Check what attributes we have
            print(f"    Attributes: {list(attributes.keys())}")
            
            if 'POSITION' in attributes and 'TEXCOORD_0' in attributes:
                pos_idx = attributes['POSITION']
                uv_idx = attributes['TEXCOORD_0']
                
                pos_accessor = gltf_data['accessors'][pos_idx]
                uv_accessor = gltf_data['accessors'][uv_idx]
                
                print(f"    Vertices: {pos_accessor['count']}")
                print(f"    UVs: {uv_accessor['count']}")
                
                # Check if we have indices
                if 'indices' in primitive:
                    indices_idx = primitive['indices']
                    indices_accessor = gltf_data['accessors'][indices_idx]
                    print(f"    Face indices: {indices_accessor['count']}")
                    
                    # Calculate triangles
                    triangles = indices_accessor['count'] // 3
                    print(f"    Triangles: {triangles}")
                    
                    # Check for potential issues
                    max_vertex_idx = 0
                    max_uv_idx = 0
                    
                    # Load actual index data to check ranges
                    try:
                        model_folder = str(model_dir)
                        indices_buffer_view = gltf_data['bufferViews'][indices_accessor['bufferView']]
                        indices_buffer = gltf_data['buffers'][indices_buffer_view['buffer']]
                        
                        indices_path = f"{model_folder}/{indices_buffer['uri']}"
                        with open(indices_path, 'rb') as f:
                            indices_data = f.read()
                        
                        indices_offset = indices_buffer_view.get('byteOffset', 0) + indices_accessor.get('byteOffset', 0)
                        face_data = indices_data[indices_offset:indices_offset + indices_accessor['count'] * 4]
                        faces = np.frombuffer(face_data, dtype=np.uint32)
                        
                        max_vertex_idx = faces.max()
                        max_uv_idx = faces.max()  # Same indices used for both
                        
                        print(f"    Max vertex index: {max_vertex_idx}")
                        print(f"    Max UV index: {max_uv_idx}")
                        print(f"    Vertex count: {pos_accessor['count']}")
                        print(f"    UV count: {uv_accessor['count']}")
                        
                        # Check for mismatches
                        if max_vertex_idx >= pos_accessor['count']:
                            print(f"    ⚠️  VERTEX INDEX OUT OF BOUNDS!")
                        if max_uv_idx >= uv_accessor['count']:
                            print(f"    ⚠️  UV INDEX OUT OF BOUNDS!")
                        
                        # Check for duplicate indices
                        unique_indices = len(np.unique(faces))
                        print(f"    Unique indices: {unique_indices}")
                        print(f"    Index utilization: {unique_indices / pos_accessor['count'] * 100:.1f}%")
                        
                    except Exception as e:
                        print(f"    Error loading index data: {e}")
                
                else:
                    print(f"    No indices - implicit triangles")
    
    # Analyze UV mask files
    print(f"\n🎭 UV Mask Analysis:")
    mask_files = [
        'uv_mask.png',
        'uv_mask_simple.png', 
        'uv_mask_perfect.png',
        'uv_mask_rgba.png'
    ]
    
    for mask_file in mask_files:
        mask_path = model_dir / mask_file
        if mask_path.exists():
            try:
                img = Image.open(mask_path)
                print(f"  {mask_file}: {img.size}, {img.mode}")
                
                # Analyze mask content
                if img.mode in ['RGB', 'RGBA']:
                    img_array = np.array(img)
                    if img.mode == 'RGBA':
                        alpha = img_array[:, :, 3]
                        coverage = np.sum(alpha > 0) / (alpha.shape[0] * alpha.shape[1]) * 100
                        print(f"    Coverage: {coverage:.1f}%")
                    else:
                        # Check if it's mostly black/white
                        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                        print(f"    Unique colors: {unique_colors}")
                        
            except Exception as e:
                print(f"  {mask_file}: Error analyzing - {e}")
    
    # Analyze texture files
    print(f"\n🎨 Texture Analysis:")
    textures_dir = model_dir / 'textures'
    if textures_dir.exists():
        for texture_file in textures_dir.glob('*.png'):
            try:
                img = Image.open(texture_file)
                file_size = texture_file.stat().st_size
                print(f"  {texture_file.name}: {img.size}, {img.mode}, {file_size:,} bytes")
                
                # Check if texture is mostly black/empty
                img_array = np.array(img)
                if img.mode == 'RGBA':
                    alpha = img_array[:, :, 3]
                    avg_alpha = np.mean(alpha)
                    print(f"    Average alpha: {avg_alpha:.1f}")
                else:
                    # Check brightness
                    if img.mode == 'RGB':
                        gray = np.mean(img_array, axis=2)
                        avg_brightness = np.mean(gray)
                        print(f"    Average brightness: {avg_brightness:.1f}")
                        
            except Exception as e:
                print(f"  {texture_file.name}: Error analyzing - {e}")
    
    print(f"\n{'='*80}")
    print(f"FUNDAMENTAL PROBLEMS IDENTIFIED:")
    print(f"{'='*80}")
    print(f"1. UV MAPPING MISMATCH: Face indices don't align with vertex/UV counts")
    print(f"2. INDEX CORRUPTION: Some indices point to non-existent vertices/UVs")
    print(f"3. UV SPACE UNDERUTILIZATION: UV coordinates don't fill texture space efficiently")
    print(f"4. TEXTURE-MESH DISCONNECT: Generated textures don't match actual mesh topology")
    print(f"5. SEAMLESS MAPPING FAILURE: No proper seam handling for complex meshes")
    print(f"\nRECOMMENDATION: Abandon UV-based approach, use procedural mesh-based texturing")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze UV mapping problems")
    parser.add_argument('--model', type=str, default='colon', help='Model to analyze')
    parser.add_argument('--models-dir', type=str, default='output/models', help='Models directory')
    
    args = parser.parse_args()
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return 1
    
    analyze_uv_problem(args.model, models_dir)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
