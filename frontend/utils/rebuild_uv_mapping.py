#!/usr/bin/env python3
"""
rebuild_uv_mapping.py

Completely rebuild UV mapping from scratch to fix blotchy textures.
This creates a clean, properly aligned UV mapping.

Usage:
    python rebuild_uv_mapping.py --model colon
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
import trimesh
import xatlas


def rebuild_uv_mapping_from_obj(obj_path: str, gltf_path: str) -> bool:
    """
    Rebuild UV mapping from the original OBJ file using xatlas.
    """
    try:
        print(f"  Loading OBJ file: {obj_path}")
        mesh = trimesh.load(obj_path, process=False)
        
        if hasattr(mesh, 'geometry'):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        
        print(f"    Vertices: {len(mesh.vertices)}")
        print(f"    Faces: {len(mesh.faces)}")
        
        # Re-unwrap with xatlas
        print(f"  Re-unwrapping with xatlas...")
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
        
        print(f"    Unwrapped vertices: {len(vmapping)}")
        print(f"    UV coordinates: {len(uvs)}")
        print(f"    New faces: {len(indices)}")
        
        # Create new mesh with proper UV mapping
        new_vertices = vertices[vmapping]
        new_faces = indices.reshape(-1, 3)
        
        # Normalize UVs to 0-1 range with padding
        uv_min = uvs.min(axis=0)
        uv_max = uvs.max(axis=0)
        uv_range = uv_max - uv_min
        uv_range = np.where(uv_range < 1e-6, 1.0, uv_range)
        
        uvs_normalized = (uvs - uv_min) / uv_range
        padding = 0.01
        scale = 1.0 - 2 * padding
        uvs_normalized = uvs_normalized * scale + padding
        
        print(f"    UV range normalized to [{uvs_normalized.min():.3f}, {uvs_normalized.max():.3f}]")
        
        # Create trimesh with UVs
        mesh_with_uvs = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False
        )
        mesh_with_uvs.visual = trimesh.visual.TextureVisuals(uv=uvs_normalized)
        
        # Export to GLTF
        print(f"  Exporting to GLTF: {gltf_path}")
        mesh_with_uvs.export(str(gltf_path), file_type='gltf')
        
        print(f"  ✓ UV mapping rebuilt successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Error rebuilding UV mapping: {e}")
        import traceback
        traceback.print_exc()
        return False


def rebuild_uv_for_model(
    model_name: str,
    obj_dir: Path = Path("output/obj"),
    models_dir: Path = Path("output/models"),
    overwrite: bool = False
) -> bool:
    """
    Rebuild UV mapping for a specific model.
    """
    print(f"\n{'='*70}")
    print(f"Rebuilding UV Mapping: {model_name}")
    print(f"{'='*70}")
    
    obj_path = obj_dir / f"{model_name}.obj"
    model_dir = models_dir / model_name
    
    if not obj_path.exists():
        print(f"✗ OBJ file not found: {obj_path}")
        return False
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False
    
    try:
        gltf_path = model_dir / 'scene.gltf'
        
        # Rebuild UV mapping
        if rebuild_uv_mapping_from_obj(str(obj_path), str(gltf_path)):
            print(f"\n{'='*70}")
            print(f"✓ SUCCESS: {model_name} UV mapping rebuilt!")
            print(f"{'='*70}\n")
            return True
        else:
            print(f"\n✗ FAILED: Could not rebuild UV mapping")
            return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild UV mapping from scratch to fix blotchy textures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Rebuild UV mapping for colon
    python rebuild_uv_mapping.py --model colon
    
    # Overwrite existing
    python rebuild_uv_mapping.py --model colon --overwrite
        """
    )
    
    parser.add_argument('--model', type=str, default='colon', help='Model name to process')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--obj-dir', type=str, default='output/obj', help='OBJ files directory')
    parser.add_argument('--models-dir', type=str, default='output/models',
                       help='Models directory (default: output/models)')
    
    args = parser.parse_args()
    
    obj_dir = Path(args.obj_dir)
    models_dir = Path(args.models_dir)
    
    if not obj_dir.exists():
        print(f"✗ OBJ directory not found: {obj_dir}")
        return 1
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return 1
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*17 + "UV Mapping Rebuilder" + " "*36 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    success = rebuild_uv_for_model(
        args.model, 
        obj_dir,
        models_dir, 
        args.overwrite
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
