#!/usr/bin/env python3
"""
fix_uv_mapping.py

Fix UV mapping issues that cause blotchy textures by ensuring proper alignment
between vertices, UVs, and face indices.

Usage:
    python fix_uv_mapping.py --model colon
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
import trimesh


def load_and_fix_gltf_uv_mapping(gltf_path: str) -> bool:
    """
    Load GLTF and fix UV mapping issues.
    """
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        
        print(f"  Analyzing GLTF structure...")
        
        for mesh_idx, mesh in enumerate(gltf_data.get('meshes', [])):
            print(f"    Processing mesh {mesh_idx}...")
            
            for prim_idx, primitive in enumerate(mesh.get('primitives', [])):
                print(f"      Processing primitive {prim_idx}...")
                
                attributes = primitive.get('attributes', {})
                
                # Check if we have the required attributes
                if 'POSITION' not in attributes or 'TEXCOORD_0' not in attributes:
                    print(f"        ⚠ Missing required attributes")
                    continue
                
                # Load vertex positions
                pos_index = attributes['POSITION']
                pos_accessor = gltf_data['accessors'][pos_index]
                pos_buffer_view = gltf_data['bufferViews'][pos_accessor['bufferView']]
                pos_buffer = gltf_data['buffers'][pos_buffer_view['buffer']]
                
                pos_path = os.path.join(model_folder, pos_buffer['uri'])
                with open(pos_path, 'rb') as f:
                    pos_data = f.read()
                
                pos_offset = pos_buffer_view.get('byteOffset', 0) + pos_accessor.get('byteOffset', 0)
                vertex_data = pos_data[pos_offset:pos_offset + pos_accessor['count'] * 12]
                vertices = np.frombuffer(vertex_data, dtype=np.float32).reshape(-1, 3)
                
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
                if 'indices' in primitive:
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
                else:
                    # No indices - faces are implicit
                    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
                
                print(f"        Vertices: {len(vertices)}")
                print(f"        UVs: {len(uvs)}")
                print(f"        Faces: {len(faces)}")
                
                # Check for index mismatches
                max_vertex_idx = faces.max() if len(faces) > 0 else 0
                max_uv_idx = faces.max() if len(faces) > 0 else 0
                
                if max_vertex_idx >= len(vertices):
                    print(f"        ⚠ Vertex index out of bounds: {max_vertex_idx} >= {len(vertices)}")
                
                if max_uv_idx >= len(uvs):
                    print(f"        ⚠ UV index out of bounds: {max_uv_idx} >= {len(uvs)}")
                
                # The issue is likely that UVs and vertices have different indexing
                # We need to ensure they match
                if len(vertices) != len(uvs):
                    print(f"        ⚠ Mismatch: {len(vertices)} vertices vs {len(uvs)} UVs")
                    
                    # Option 1: Duplicate UVs to match vertices
                    if len(uvs) < len(vertices):
                        print(f"        Fixing: Expanding UVs to match vertices...")
                        # Repeat UVs to match vertex count
                        repeat_factor = len(vertices) // len(uvs) + 1
                        expanded_uvs = np.tile(uvs, (repeat_factor, 1))
                        uvs = expanded_uvs[:len(vertices)]
                    
                    # Option 2: Truncate vertices to match UVs
                    elif len(vertices) > len(uvs):
                        print(f"        Fixing: Truncating vertices to match UVs...")
                        vertices = vertices[:len(uvs)]
                        # Update faces to not exceed UV count
                        faces = faces[faces.max(axis=1) < len(uvs)]
                
                print(f"        After fix: {len(vertices)} vertices, {len(uvs)} UVs, {len(faces)} faces")
                
                # Update the GLTF data
                # Update vertex buffer
                new_vertex_data = vertices.tobytes()
                pos_buffer['byteLength'] = len(new_vertex_data)
                pos_accessor['count'] = len(vertices)
                
                # Update UV buffer
                new_uv_data = uvs.tobytes()
                uv_buffer['byteLength'] = len(new_uv_data)
                uv_accessor['count'] = len(uvs)
                
                # Update face buffer
                if 'indices' in primitive:
                    new_face_data = faces.tobytes()
                    indices_buffer['byteLength'] = len(new_face_data)
                    indices_accessor['count'] = len(faces) * 3
                
                # Write updated buffers
                with open(pos_path, 'wb') as f:
                    f.write(new_vertex_data)
                
                with open(uv_path, 'wb') as f:
                    f.write(new_uv_data)
                
                if 'indices' in primitive:
                    with open(indices_path, 'wb') as f:
                        f.write(new_face_data)
        
        # Save updated GLTF
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        print(f"  ✓ GLTF UV mapping fixed and saved")
        return True
        
    except Exception as e:
        print(f"  ✗ Error fixing UV mapping: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_uv_mapping_for_model(
    model_name: str,
    models_dir: Path = Path("output/models"),
    overwrite: bool = False
) -> bool:
    """
    Fix UV mapping for a specific model.
    """
    print(f"\n{'='*70}")
    print(f"Fixing UV Mapping: {model_name}")
    print(f"{'='*70}")
    
    model_dir = models_dir / model_name
    
    try:
        # Find GLTF file
        gltf_path = model_dir / 'scene.gltf'
        if not gltf_path.exists():
            print(f"✗ GLTF file not found: {gltf_path}")
            return False
        
        print(f"Found GLTF: {gltf_path}")
        
        # Fix UV mapping
        if load_and_fix_gltf_uv_mapping(str(gltf_path)):
            print(f"\n{'='*70}")
            print(f"✓ SUCCESS: {model_name} UV mapping fixed!")
            print(f"{'='*70}\n")
            return True
        else:
            print(f"\n✗ FAILED: Could not fix UV mapping")
            return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix UV mapping issues that cause blotchy textures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix UV mapping for colon
    python fix_uv_mapping.py --model colon
    
    # Overwrite existing
    python fix_uv_mapping.py --model colon --overwrite
        """
    )
    
    parser.add_argument('--model', type=str, default='colon', help='Model name to process')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--models-dir', type=str, default='output/models',
                       help='Models directory (default: output/models)')
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return 1
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*17 + "UV Mapping Fixer" + " "*37 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    success = fix_uv_mapping_for_model(
        args.model, 
        models_dir, 
        args.overwrite
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
