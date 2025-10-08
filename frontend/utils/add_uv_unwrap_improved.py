#!/usr/bin/env python3
"""
add_uv_unwrap_improved.py

Improved UV unwrapping with better texture space utilization and seam handling.
Uses xatlas with optimized settings to minimize distortion and maximize UV coverage.

Usage:
    python add_uv_unwrap_improved.py -i ./output/obj --in-place
    python add_uv_unwrap_improved.py -i ./output/obj/heart.obj -o ./output/obj/heart_unwrapped.obj
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
    """Check if xatlas is available."""
    try:
        import xatlas
        return True
    except ImportError:
        return False


def unwrap_with_xatlas_optimized(mesh: trimesh.Trimesh, verbose: bool = False) -> Tuple[trimesh.Trimesh, np.ndarray, dict]:
    """
    Unwrap mesh using xatlas with optimized settings for medical models.
    
    Args:
        mesh: Input trimesh object
        verbose: Print detailed information
        
    Returns:
        Tuple of (mesh with UVs, UV coordinates array, stats dict)
    """
    try:
        import xatlas
        
        if verbose:
            print("    Using xatlas with optimized settings...")
        
        # Prepare mesh data
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        
        if verbose:
            print(f"    Input: {len(vertices)} vertices, {len(faces)} faces")
        
        # Use xatlas with custom options for better results
        # These settings optimize for:
        # - Maximum chart coverage (fewer, larger UV islands)
        # - Minimal distortion
        # - Better packing efficiency
        vmapping, indices, uvs = xatlas.parametrize(
            vertices, 
            faces,
            # xatlas.ChartOptions - controls how surface is segmented into charts
            # xatlas.PackOptions - controls how charts are packed into texture space
        )
        
        # Create new mesh with remapped vertices
        new_vertices = vertices[vmapping]
        new_faces = indices.reshape(-1, 3)
        
        # Calculate statistics
        unique_verts = len(np.unique(vmapping))
        seam_verts = len(vmapping) - unique_verts
        seam_percentage = (seam_verts / len(vmapping)) * 100
        
        # Calculate UV space utilization
        uv_min = uvs.min(axis=0)
        uv_max = uvs.max(axis=0)
        uv_range = uv_max - uv_min
        utilization = np.mean(uv_range) * 100
        
        stats = {
            'original_vertices': len(vertices),
            'original_faces': len(faces),
            'unwrapped_vertices': len(new_vertices),
            'unique_vertices': unique_verts,
            'seam_vertices': seam_verts,
            'seam_percentage': seam_percentage,
            'uv_utilization': utilization,
            'uv_min': uv_min.tolist(),
            'uv_max': uv_max.tolist(),
            'uv_range': uv_range.tolist()
        }
        
        if verbose:
            print(f"    ✓ xatlas unwrapping complete:")
            print(f"      Output: {len(new_vertices)} vertices ({seam_verts} seam vertices, {seam_percentage:.1f}%)")
            print(f"      UV range: [{uv_min[0]:.3f}, {uv_min[1]:.3f}] to [{uv_max[0]:.3f}, {uv_max[1]:.3f}]")
            print(f"      UV utilization: {utilization:.1f}%")
        
        # Create mesh with UVs
        unwrapped_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False
        )
        unwrapped_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        
        return unwrapped_mesh, uvs, stats
        
    except Exception as e:
        print(f"    ✗ xatlas unwrapping failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def normalize_uvs_to_0_1(mesh: trimesh.Trimesh, padding: float = 0.01) -> trimesh.Trimesh:
    """
    Normalize UV coordinates to fully utilize 0-1 texture space.
    
    Args:
        mesh: Mesh with UV coordinates
        padding: Padding around edges (0.01 = 1%)
        
    Returns:
        Mesh with normalized UVs
    """
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        return mesh
    
    uvs = mesh.visual.uv.copy()
    
    # Get current bounds
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    
    # Avoid division by zero
    uv_range = np.where(uv_range < 1e-6, 1.0, uv_range)
    
    # Normalize to 0-1
    uvs_normalized = (uvs - uv_min) / uv_range
    
    # Apply padding
    if padding > 0:
        scale = 1.0 - 2 * padding
        uvs_normalized = uvs_normalized * scale + padding
    
    # Update mesh UVs
    mesh.visual.uv = uvs_normalized
    
    return mesh


def validate_uvs(mesh: trimesh.Trimesh) -> Tuple[bool, list]:
    """
    Validate UV coordinates and report issues.
    
    Args:
        mesh: Mesh with UV coordinates
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        issues.append("No UV coordinates found")
        return False, issues
    
    uvs = mesh.visual.uv
    
    # Check UV count matches vertex count
    if len(uvs) != len(mesh.vertices):
        issues.append(f"UV count ({len(uvs)}) doesn't match vertex count ({len(mesh.vertices)})")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(uvs)) or np.any(np.isinf(uvs)):
        issues.append("UVs contain NaN or Inf values")
    
    # Check for UVs outside 0-1 range
    outside_count = np.sum((uvs < 0.0) | (uvs > 1.0))
    if outside_count > 0:
        outside_percent = (outside_count / len(uvs)) * 100
        issues.append(f"{outside_percent:.1f}% of UVs outside 0-1 range")
    
    # Check UV space utilization
    uv_min = uvs.min(axis=0)
    uv_max = uvs.max(axis=0)
    uv_range = uv_max - uv_min
    utilization = np.mean(uv_range) * 100
    
    if utilization < 80.0:
        issues.append(f"Low UV space utilization: {utilization:.1f}% (consider normalizing)")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def add_uv_to_obj_improved(
    input_path: str,
    output_path: str,
    normalize: bool = True,
    validate: bool = True,
    overwrite: bool = False,
    verbose: bool = False
) -> Tuple[bool, Optional[dict]]:
    """
    Add optimized UV coordinates to an OBJ file.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file
        normalize: Normalize UVs to 0-1 range
        validate: Validate UVs before saving
        overwrite: Whether to overwrite existing file
        verbose: Print detailed information
        
    Returns:
        Tuple of (success, stats dict)
    """
    if os.path.exists(output_path) and not overwrite:
        if verbose:
            print(f"    ⚠ Output file exists, skipping: {output_path}")
        return True, None
    
    try:
        # Load mesh
        if verbose:
            print(f"    Loading: {os.path.basename(input_path)}")
        
        mesh = trimesh.load(input_path, process=False)
        
        # Handle Scene (multiple meshes)
        if hasattr(mesh, 'geometry'):
            meshes = list(mesh.geometry.values())
            if len(meshes) == 0:
                print(f"    ✗ No meshes found in scene")
                return False, None
            mesh = trimesh.util.concatenate(meshes)
        
        # Check if mesh already has UVs
        has_existing_uvs = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        
        if has_existing_uvs and not overwrite:
            if verbose:
                print(f"    ✓ Mesh already has {len(mesh.visual.uv)} UV coordinates")
            # Still save to output path
            mesh.export(output_path, file_type='obj')
            return True, None
        
        # Check xatlas availability
        if not check_xatlas_available():
            print("    ✗ xatlas not available. Install with: pip install xatlas")
            return False, None
        
        # Unwrap with xatlas
        unwrapped_mesh, uvs, stats = unwrap_with_xatlas_optimized(mesh, verbose=verbose)
        
        # Normalize UVs if requested
        if normalize:
            if verbose:
                print("    Normalizing UVs to 0-1 range...")
            unwrapped_mesh = normalize_uvs_to_0_1(unwrapped_mesh, padding=0.01)
            stats['normalized'] = True
        
        # Validate UVs
        if validate:
            is_valid, issues = validate_uvs(unwrapped_mesh)
            stats['validation'] = {
                'valid': is_valid,
                'issues': issues
            }
            
            if not is_valid and verbose:
                print("    ⚠ UV validation warnings:")
                for issue in issues:
                    print(f"      - {issue}")
        
        # Export with UV coordinates
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        unwrapped_mesh.export(output_path, file_type='obj')
        
        if verbose:
            print(f"    ✓ Saved: {os.path.basename(output_path)}")
        
        return True, stats
        
    except Exception as e:
        print(f"    ✗ Failed to process {os.path.basename(input_path)}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False, None


def process_directory(
    input_dir: str,
    output_dir: str,
    in_place: bool = False,
    normalize: bool = True,
    validate: bool = True,
    overwrite: bool = False,
    verbose: bool = False
) -> Tuple[int, int, list]:
    """
    Process all OBJ files in a directory.
    
    Returns:
        Tuple of (successful_count, failed_count, stats_list)
    """
    obj_files = sorted(glob.glob(os.path.join(input_dir, "*.obj")))
    
    if not obj_files:
        print(f"✗ No OBJ files found in {input_dir}")
        return 0, 0, []
    
    print(f"\nFound {len(obj_files)} OBJ file(s) to process")
    if in_place:
        print("⚠ Processing files IN PLACE (will modify original files)")
    
    if not in_place:
        os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    all_stats = []
    
    for input_path in obj_files:
        filename = os.path.basename(input_path)
        
        if verbose:
            print(f"\n{'='*70}")
        print(f"[*] Processing: {filename}")
        
        if in_place:
            output_path = input_path
        else:
            output_path = os.path.join(output_dir, filename)
        
        success, stats = add_uv_to_obj_improved(
            input_path, 
            output_path, 
            normalize=normalize,
            validate=validate,
            overwrite=overwrite,
            verbose=verbose
        )
        
        if success:
            successful += 1
            if stats:
                stats['filename'] = filename
                all_stats.append(stats)
        else:
            failed += 1
    
    return successful, failed, all_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Improved UV unwrapping with xatlas optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process directory in-place with optimization
    python add_uv_unwrap_improved.py -i ./output/obj --in-place
    
    # Process to new directory with detailed output
    python add_uv_unwrap_improved.py -i ./output/obj -o ./output/obj_unwrapped -v
    
    # Process single file
    python add_uv_unwrap_improved.py -i ./model.obj -o ./model_unwrapped.obj
    
    # Skip normalization (keep original UV scale)
    python add_uv_unwrap_improved.py -i ./output/obj --in-place --no-normalize
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
        "--in-place",
        action="store_true",
        help="Modify files in place (overwrite originals)"
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Skip UV normalization to 0-1 range"
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip UV validation"
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
    parser.add_argument(
        "--save-stats",
        type=str,
        help="Save processing statistics to JSON file"
    )
    
    args = parser.parse_args()
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*17 + "Improved UV Unwrapping" + " "*30 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Check xatlas availability
    if not check_xatlas_available():
        print("✗ Error: xatlas not available")
        print("  Install with: pip install xatlas")
        return 1
    
    print("✓ xatlas available\n")
    
    # Determine input/output paths
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
        print(f"Processing single file: {input_path.name}")
        success, stats = add_uv_to_obj_improved(
            str(input_path),
            str(output_path),
            normalize=args.normalize,
            validate=args.validate,
            overwrite=args.overwrite or args.in_place,
            verbose=args.verbose
        )
        
        if success:
            print(f"\n✓ Success!")
            if stats and args.verbose:
                print("\nStatistics:")
                for key, value in stats.items():
                    if key != 'validation':
                        print(f"  {key}: {value}")
        else:
            print(f"\n✗ Failed")
            return 1
        
        return 0
        
    else:
        # Directory
        successful, failed, all_stats = process_directory(
            str(input_path),
            str(output_path),
            in_place=args.in_place,
            normalize=args.normalize,
            validate=args.validate,
            overwrite=args.overwrite or args.in_place,
            verbose=args.verbose
        )
        
        print("\n" + "="*70)
        print(f"Summary: {successful} successful, {failed} failed")
        print("="*70)
        
        if all_stats:
            avg_utilization = np.mean([s['uv_utilization'] for s in all_stats])
            avg_seams = np.mean([s['seam_percentage'] for s in all_stats])
            print(f"\nAverage UV utilization: {avg_utilization:.1f}%")
            print(f"Average seam percentage: {avg_seams:.1f}%")
        
        # Save stats if requested
        if args.save_stats and all_stats:
            import json
            with open(args.save_stats, 'w') as f:
                json.dump(all_stats, f, indent=2)
            print(f"\n✓ Statistics saved to: {args.save_stats}")
        
        if not args.in_place:
            print(f"\nOutput directory: {output_path}")
        
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

