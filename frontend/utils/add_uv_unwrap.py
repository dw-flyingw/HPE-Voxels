#!/usr/bin/env python3
"""
add_uv_unwrap.py

Add UV coordinates to OBJ files that don't have them.
Uses xatlas for optimal UV unwrapping with spherical mapping as fallback.

Requirements
------------
- trimesh
- numpy
- xatlas (optional, for optimal unwrapping)

Example
-------
    # Process all OBJ files
    python add_uv_unwrap.py -i ./output/obj -o ./output/obj_unwrapped
    
    # Process specific file
    python add_uv_unwrap.py -i ./output/obj/heart.obj -o ./output/obj_unwrapped/heart.obj
    
    # Use specific unwrapping method
    python add_uv_unwrap.py -i ./output/obj -m spherical
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import numpy as np
import trimesh
from typing import Optional, Tuple


def check_xatlas_available() -> bool:
    """Check if xatlas is available for optimal UV unwrapping."""
    try:
        import xatlas
        return True
    except ImportError:
        return False


def unwrap_with_xatlas(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Unwrap mesh using xatlas for optimal UV coordinates.
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Tuple of (mesh with UVs, UV coordinates array)
    """
    try:
        import xatlas
        
        print("    - Using xatlas for optimal UV unwrapping...")
        
        # Prepare mesh data for xatlas
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        
        # Create xatlas atlas
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
        
        # xatlas may duplicate vertices for proper UV seams
        # Create new mesh with remapped vertices
        new_vertices = vertices[vmapping]
        new_faces = indices.reshape(-1, 3)
        
        # Create new mesh
        unwrapped_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False
        )
        
        # Set UV coordinates
        unwrapped_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        
        print(f"    ✓ Generated {len(uvs)} UV coordinates using xatlas")
        return unwrapped_mesh, uvs
        
    except Exception as e:
        print(f"    ⚠️  xatlas unwrapping failed: {e}")
        raise


def unwrap_spherical(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Generate spherical UV mapping for mesh.
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Tuple of (mesh with UVs, UV coordinates array)
    """
    print("    - Using spherical UV mapping...")
    
    vertices = mesh.vertices
    
    # Center the mesh
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    # Calculate spherical coordinates
    r = np.sqrt(np.sum(centered**2, axis=1))
    r = np.where(r == 0, 1e-10, r)  # Avoid division by zero
    
    # Theta (azimuth): angle in XY plane
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    # Normalize to [0, 1]
    u = (theta + np.pi) / (2 * np.pi)
    
    # Phi (elevation): angle from Z axis
    phi = np.arccos(np.clip(centered[:, 2] / r, -1, 1))
    # Normalize to [0, 1]
    v = phi / np.pi
    
    # Stack to create UV coordinates
    uvs = np.column_stack([u, v])
    
    # Ensure UVs are in valid range
    uvs = np.clip(uvs, 0, 1)
    
    # Create mesh with UV coordinates
    mesh_with_uvs = mesh.copy()
    mesh_with_uvs.visual = trimesh.visual.TextureVisuals(uv=uvs)
    
    print(f"    ✓ Generated {len(uvs)} UV coordinates using spherical mapping")
    return mesh_with_uvs, uvs


def unwrap_cylindrical(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Generate cylindrical UV mapping for mesh (good for elongated objects).
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Tuple of (mesh with UVs, UV coordinates array)
    """
    print("    - Using cylindrical UV mapping...")
    
    vertices = mesh.vertices
    
    # Center the mesh
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    # Cylindrical mapping
    # U coordinate from angle around Y axis
    u = 0.5 + np.arctan2(centered[:, 2], centered[:, 0]) / (2 * np.pi)
    
    # V coordinate from height (Y axis)
    y_min, y_max = centered[:, 1].min(), centered[:, 1].max()
    if y_max > y_min:
        v = (centered[:, 1] - y_min) / (y_max - y_min)
    else:
        v = np.zeros(len(centered))
    
    # Stack to create UV coordinates
    uvs = np.column_stack([u, v])
    uvs = np.clip(uvs, 0, 1)
    
    # Create mesh with UV coordinates
    mesh_with_uvs = mesh.copy()
    mesh_with_uvs.visual = trimesh.visual.TextureVisuals(uv=uvs)
    
    print(f"    ✓ Generated {len(uvs)} UV coordinates using cylindrical mapping")
    return mesh_with_uvs, uvs


def unwrap_smart(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Smart UV unwrapping - tries xatlas first, falls back to spherical.
    
    Args:
        mesh: Input trimesh object
        
    Returns:
        Tuple of (mesh with UVs, UV coordinates array)
    """
    if check_xatlas_available():
        try:
            return unwrap_with_xatlas(mesh)
        except Exception:
            print("    - Falling back to spherical mapping...")
            return unwrap_spherical(mesh)
    else:
        print("    ⚠️  xatlas not available, using spherical mapping")
        print("    💡 Install xatlas for better UV unwrapping: pip install xatlas")
        return unwrap_spherical(mesh)


def add_uv_to_obj(
    input_path: str,
    output_path: str,
    method: str = 'smart',
    overwrite: bool = False
) -> bool:
    """
    Add UV coordinates to an OBJ file.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file
        method: UV mapping method ('smart', 'xatlas', 'spherical', 'cylindrical')
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"    ⚠️  Output file exists, skipping: {output_path}")
        return True
    
    try:
        # Load mesh
        print(f"    - Loading mesh from {os.path.basename(input_path)}...")
        mesh = trimesh.load(input_path, process=False)
        
        # Handle Scene (multiple meshes)
        if hasattr(mesh, 'geometry'):
            meshes = list(mesh.geometry.values())
            if len(meshes) == 0:
                print(f"    ✗ No meshes found in scene")
                return False
            # Concatenate all meshes
            mesh = trimesh.util.concatenate(meshes)
        
        # Check if mesh already has UVs
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            print(f"    ✓ Mesh already has {len(mesh.visual.uv)} UV coordinates")
            # Still save to output path
            mesh.export(output_path, file_type='obj')
            return True
        
        # Generate UV coordinates based on method
        if method == 'smart':
            unwrapped_mesh, uvs = unwrap_smart(mesh)
        elif method == 'xatlas':
            if not check_xatlas_available():
                print("    ✗ xatlas not available. Install with: pip install xatlas")
                return False
            unwrapped_mesh, uvs = unwrap_with_xatlas(mesh)
        elif method == 'spherical':
            unwrapped_mesh, uvs = unwrap_spherical(mesh)
        elif method == 'cylindrical':
            unwrapped_mesh, uvs = unwrap_cylindrical(mesh)
        else:
            print(f"    ✗ Unknown UV mapping method: {method}")
            return False
        
        # Export with UV coordinates
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        unwrapped_mesh.export(output_path, file_type='obj')
        
        print(f"    ✓ Saved OBJ with UV coordinates to {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process {os.path.basename(input_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(
    input_dir: str,
    output_dir: str,
    method: str = 'smart',
    overwrite: bool = False,
    in_place: bool = False
) -> Tuple[int, int]:
    """
    Process all OBJ files in a directory.
    
    Args:
        input_dir: Input directory with OBJ files
        output_dir: Output directory for processed files
        method: UV mapping method
        overwrite: Whether to overwrite existing files
        in_place: Whether to modify files in place
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    obj_files = glob.glob(os.path.join(input_dir, "*.obj"))
    
    if not obj_files:
        print(f"✗ No OBJ files found in {input_dir}")
        return 0, 0
    
    print(f"\nFound {len(obj_files)} OBJ file(s) to process")
    if in_place:
        print("⚠️  Processing files IN PLACE (will modify original files)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for input_path in sorted(obj_files):
        filename = os.path.basename(input_path)
        print(f"\n[*] Processing: {filename}")
        
        if in_place:
            output_path = input_path
        else:
            output_path = os.path.join(output_dir, filename)
        
        if add_uv_to_obj(input_path, output_path, method, overwrite):
            successful += 1
        else:
            failed += 1
    
    return successful, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add UV coordinates to OBJ files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
UV Mapping Methods:
  smart        - Try xatlas first, fall back to spherical (recommended)
  xatlas       - Optimal UV unwrapping (requires xatlas package)
  spherical    - Spherical projection (good for organic shapes)
  cylindrical  - Cylindrical projection (good for tubes/elongated objects)

Examples:
  # Process directory with smart method
  python add_uv_unwrap.py -i ./output/obj -o ./output/obj_unwrapped
  
  # Process in-place (modify original files)
  python add_uv_unwrap.py -i ./output/obj --in-place
  
  # Use specific method
  python add_uv_unwrap.py -i ./output/obj -m spherical
  
  # Single file
  python add_uv_unwrap.py -i ./model.obj -o ./model_unwrapped.obj
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input OBJ file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output OBJ file or directory (default: input_dir + '_unwrapped')"
    )
    parser.add_argument(
        "-m", "--method",
        choices=['smart', 'xatlas', 'spherical', 'cylindrical'],
        default='smart',
        help="UV mapping method (default: smart)"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify files in place (overwrite originals)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("UV Unwrapping Utility")
    print("="*70)
    
    # Check if xatlas is available
    if args.method in ['smart', 'xatlas']:
        has_xatlas = check_xatlas_available()
        if has_xatlas:
            print("✓ xatlas available for optimal UV unwrapping")
        else:
            print("⚠️  xatlas not available (will use spherical mapping)")
            print("   Install with: pip install xatlas")
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Error: Input path does not exist: {args.input}")
        return 1
    
    # Handle output path
    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        if input_path.is_dir():
            output_path = Path(str(input_path) + "_unwrapped")
        else:
            output_path = input_path.parent / (input_path.stem + "_unwrapped" + input_path.suffix)
    
    # Process
    if input_path.is_file():
        # Single file
        print(f"\nProcessing single file: {input_path.name}")
        success = add_uv_to_obj(
            str(input_path),
            str(output_path),
            args.method,
            args.overwrite or args.in_place
        )
        print(f"\n{'✓ Success' if success else '✗ Failed'}")
        return 0 if success else 1
    else:
        # Directory
        successful, failed = process_directory(
            str(input_path),
            str(output_path),
            args.method,
            args.overwrite or args.in_place,
            args.in_place
        )
        
        print("\n" + "="*70)
        print(f"Summary: {successful} successful, {failed} failed")
        print("="*70)
        
        if not args.in_place:
            print(f"\nOutput directory: {output_path}")
        
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

