#!/usr/bin/env python3
"""
nifti2obj.py

A utility to convert .nii.gz files into high-quality .obj surface meshes.
If the input is 'all.nii.gz', it generates a multi-label mesh where each
label is colored and named according to the provided JSON color map.

Requirements
------------
- nibabel
- numpy
- scikit-image
- open3d
- trimesh

Example
-------
    python nifti2obj.py -i ./input -o ./output/obj
"""

import argparse
import os
import sys
import glob
import json
import nibabel as nib
import numpy as np
from skimage import measure
import open3d as o3d
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def _normalize_label_name(name: str) -> str:
    """Normalize label names for robust matching between filenames and JSON names."""
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = s.replace("-", "_").replace(" ", "_")
    # Collapse multiple underscores and trim
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def detect_boundary_holes(mesh):
    """Detect boundary holes in a mesh and return boundary vertices and edges."""
    if mesh.is_watertight:
        return None, None, []
    
    # Find boundary edges (edges that appear only once)
    edges = mesh.edges_unique
    edge_counts = mesh.edges_unique_inverse
    unique_counts = np.bincount(edge_counts)
    boundary_edge_mask = unique_counts == 1
    boundary_edges = edges[boundary_edge_mask]
    
    if len(boundary_edges) == 0:
        return None, None, []
    
    # Get boundary vertices
    boundary_vertices = np.unique(boundary_edges.flatten())
    boundary_coords = mesh.vertices[boundary_vertices]
    
    # Group boundary vertices into separate holes/components
    holes = []
    if len(boundary_vertices) > 0:
        # Use clustering to group boundary vertices into separate holes
        try:
            from sklearn.cluster import DBSCAN
            
            # Use a more adaptive clustering approach
            # Try different eps values to find the best clustering
            best_clustering = None
            best_n_clusters = 0
            
            for eps_factor in [0.05, 0.1, 0.2, 0.5]:
                clustering = DBSCAN(eps=mesh.scale * eps_factor, min_samples=3).fit(boundary_coords)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                
                if n_clusters > best_n_clusters:
                    best_clustering = clustering
                    best_n_clusters = n_clusters
            
            if best_clustering is not None:
                labels = best_clustering.labels_
                
                # Group vertices by cluster
                for label in np.unique(labels):
                    if label == -1:  # Skip noise points
                        continue
                    hole_vertices = boundary_vertices[labels == label]
                    if len(hole_vertices) >= 3:  # Need at least 3 vertices for a hole
                        holes.append(hole_vertices)
        
        except ImportError:
            # Fallback: treat all boundary vertices as one hole
            print("    [!] sklearn not available, treating all boundary vertices as one hole")
            holes = [boundary_vertices]
    
    return boundary_vertices, boundary_edges, holes


def fill_hole_with_convex_hull(mesh, hole_vertices, method='convex'):
    """Fill a hole in the mesh using the specified method."""
    if len(hole_vertices) < 3:
        return mesh
    
    hole_coords = mesh.vertices[hole_vertices]
    
    if method == 'convex':
        try:
            # Use convex hull to fill the hole
            hull = ConvexHull(hole_coords)
            # Create faces for the hole (ensure correct winding)
            hole_faces = hull.simplices
            
            # Check face orientation by computing normals
            face_centers = []
            face_normals = []
            for face in hole_faces:
                v0, v1, v2 = hole_coords[face]
                center = (v0 + v1 + v2) / 3
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)
                face_centers.append(center)
                face_normals.append(normal)
            
            # Determine if we need to flip the faces
            # Get the center of the hole
            hole_center = np.mean(hole_coords, axis=0)
            
            # Get the center of the original mesh
            mesh_center = np.mean(mesh.vertices, axis=0)
            
            # Compute average outward normal from hole center
            outward_vector = hole_center - mesh_center
            
            # Check if normals point in the right direction
            avg_normal = np.mean(face_normals, axis=0)
            if np.dot(avg_normal, outward_vector) < 0:
                # Flip faces
                hole_faces = hole_faces[:, [0, 2, 1]]
            
            # Map back to original vertex indices
            hole_faces_mapped = hole_vertices[hole_faces]
            
            # Add new faces to the mesh
            new_faces = np.vstack([mesh.faces, hole_faces_mapped])
            new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_faces)
            
            return new_mesh
            
        except Exception as e:
            print(f"    [!] Convex hull hole filling failed: {e}", file=sys.stderr)
            return mesh
    
    elif method == 'planar':
        # Simple planar triangulation (works well for flat cuts)
        if len(hole_vertices) >= 3:
            # Use the first few vertices to create a planar triangulation
            # This is a simplified approach - could be improved with proper triangulation
            n_verts = len(hole_vertices)
            new_faces = []
            
            # Create a fan triangulation from the first vertex
            for i in range(1, n_verts - 1):
                face = [hole_vertices[0], hole_vertices[i], hole_vertices[i + 1]]
                new_faces.append(face)
            
            if new_faces:
                new_faces = np.array(new_faces)
                # Add new faces to the mesh
                updated_faces = np.vstack([mesh.faces, new_faces])
                new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=updated_faces)
                return new_mesh
    
    return mesh


def close_mesh_boundaries(mesh, method='convex', max_hole_size=1000):
    """Close boundary holes in a mesh to make it watertight."""
    if mesh.is_watertight:
        return mesh
    
    print(f"    - Attempting to close mesh boundaries...")
    
    # First try trimesh's built-in hole filling
    try:
        # Use trimesh's fill_holes method (returns boolean)
        holes_filled = mesh.fill_holes()
        if holes_filled:
            print(f"    - Successfully filled holes using trimesh's built-in method")
            if mesh.is_watertight:
                print(f"    - Mesh is now watertight!")
                return mesh
            else:
                print(f"    - Some holes filled but mesh still not watertight")
        else:
            print(f"    - Trimesh's built-in method couldn't fill holes")
    except Exception as e:
        print(f"    - Trimesh's built-in hole filling failed: {e}")
    
    # Fallback to custom hole filling
    boundary_vertices, boundary_edges, holes = detect_boundary_holes(mesh)
    
    if not holes:
        return mesh
    
    print(f"    - Found {len(holes)} boundary holes to fill")
    
    closed_mesh = mesh
    for i, hole_vertices in enumerate(holes):
        if len(hole_vertices) > max_hole_size:
            print(f"    - Skipping hole {i+1} (too large: {len(hole_vertices)} vertices)")
            continue
            
        print(f"    - Filling hole {i+1} with {len(hole_vertices)} vertices using {method} method")
        closed_mesh = fill_hole_with_convex_hull(closed_mesh, hole_vertices, method)
    
    # Final check and attempt to use trimesh's repair methods
    if not closed_mesh.is_watertight:
        print(f"    - Trying trimesh repair methods...")
        try:
            # Try to repair the mesh
            repaired_mesh = closed_mesh.fix_normals()
            
            if repaired_mesh.is_watertight:
                print(f"    - Successfully repaired mesh using trimesh repair methods")
                return repaired_mesh
            else:
                print(f"    - Mesh repair didn't make it watertight, but may have improved it")
                return repaired_mesh
        except Exception as e:
            print(f"    - Trimesh repair failed: {e}")
    
    return closed_mesh


def load_label_info(json_path):
    """Loads a map from label ID to its name and color."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {item['id']: {'name': item['name'], 'color': item['color']} for item in data}
    except FileNotFoundError:
        print(f"[!] Warning: Color map not found at '{json_path}'. Using random colors.", file=sys.stderr)
        return None


def nifti_to_mesh(nib_img: nib.Nifti1Image, threshold: float, smoothing: int, decimation: float, 
                  label_info: dict = None, base_name: str = None, textured_voxels: np.ndarray = None,
                  close_boundaries: bool = False, hole_filling_method: str = 'convex'):
    """Converts a single-label NIfTI image to a trimesh Mesh."""
    data = nib_img.get_fdata()
    spacing = nib_img.header.get_zooms()[:3]  # Get voxel spacing (x, y, z)
    
    # Use marching cubes to generate isosurface
    verts, faces, normals, values = measure.marching_cubes(data, level=threshold, spacing=spacing)
    
    if len(verts) == 0:
        raise ValueError("No surface generated at the given threshold.")

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Extract largest connected component
    mesh = mesh.split(only_watertight=False)[0] if len(mesh.split(only_watertight=False)) > 0 else mesh
    
    # Apply decimation if requested
    if 0 < decimation < 1:
        try:
            target_faces = int(len(mesh.faces) * decimation)
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except ImportError:
            # Fallback to basic decimation if fast_simplification is not available
            print(f"    [!] fast_simplification not available, skipping decimation", file=sys.stderr)
    
    # Apply smoothing if requested
    if smoothing > 0:
        # Convert to open3d for smoothing, then back to trimesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        
        # Apply Laplacian smoothing
        o3d_mesh = o3d_mesh.filter_smooth_simple(number_of_iterations=smoothing)
        
        # Convert back to trimesh
        mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles))
    
    # Compute normals
    mesh.vertex_normals
    
    # Close boundaries if requested
    if close_boundaries and not mesh.is_watertight:
        print(f"    - Closing mesh boundaries using {hole_filling_method} method")
        mesh = close_mesh_boundaries(mesh, method=hole_filling_method)
    
    # Handle textured voxels
    if textured_voxels is not None and textured_voxels.shape[:3] == data.shape:
        # Convert world coordinates back to voxel indices
        # Note: This is a simplified approach - may need refinement based on actual coordinate systems
        voxel_indices = (verts / np.array(spacing)).astype(int)
        
        # Clamp indices to be within the bounds of the textured_voxels array
        max_indices = np.array(textured_voxels.shape[:3]) - 1
        voxel_indices = np.clip(voxel_indices, 0, max_indices)
        
        # Look up colors from the textured volume
        vertex_colors = textured_voxels[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
        
        # Store colors in mesh metadata for OBJ export
        mesh.visual.vertex_colors = vertex_colors.astype(np.uint8)
    
    # Fallback to uniform color from label_info if available
    elif label_info and base_name:
        # Find matching label by name
        color_applied = False
        normalized_base = _normalize_label_name(base_name)
        for label_id, info in label_info.items():
            # Compare base_name with label name (case-insensitive, handle spaces)
            label_name = _normalize_label_name(info['name'])
            if normalized_base == label_name:
                # Set per-vertex colors for proper OBJ export
                n_vertices = len(mesh.vertices)
                vertex_colors = np.tile(info['color'], (n_vertices, 1)).astype(np.uint8)
                mesh.visual.vertex_colors = vertex_colors
                color_applied = True
                break
        
        if not color_applied:
            print(f"    [!] No color mapping found for '{base_name}', using default color", file=sys.stderr)
            # Set default color
            n_vertices = len(mesh.vertices)
            default_color = [128, 128, 128]  # Gray
            vertex_colors = np.tile(default_color, (n_vertices, 1)).astype(np.uint8)
            mesh.visual.vertex_colors = vertex_colors
    
    return mesh


def nifti_to_multilabel_mesh(nib_img: nib.Nifti1Image, label_info: dict, smoothing: int, decimation: float, 
                            textured_voxels: np.ndarray = None, close_boundaries: bool = False, 
                            hole_filling_method: str = 'convex'):
    """Converts a multi-label NIfTI into a colored and named list of meshes."""
    data = nib_img.get_fdata()
    spacing = nib_img.header.get_zooms()[:3]  # Get voxel spacing (x, y, z)
    # Cast to int to avoid float comparison issues
    data_i = data.astype(np.int32, copy=False)
    labels = np.unique(data_i)
    labels = labels[labels > 0]  # Ignore background label 0

    meshes = []

    for label in labels:
        print(f"    - Processing label: {int(label)}")
        mask = np.zeros_like(data_i, dtype=np.uint8)
        mask[data_i == label] = 1
        
        # Use marching cubes to generate isosurface
        verts, faces, normals, values = measure.marching_cubes(mask, level=0.5, spacing=spacing)
        
        if len(verts) == 0:
            continue

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Extract largest connected component
        mesh = mesh.split(only_watertight=False)[0] if len(mesh.split(only_watertight=False)) > 0 else mesh
        
        # Apply decimation if requested
        if 0 < decimation < 1:
            try:
                target_faces = int(len(mesh.faces) * decimation)
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            except ImportError:
                # Fallback to basic decimation if fast_simplification is not available
                print(f"    [!] fast_simplification not available, using basic decimation", file=sys.stderr)
                # Simple face reduction by sampling every nth face
                step = int(1 / decimation)
                if step > 1:
                    mesh = mesh.subdivide_to_size(face_size=mesh.scale * 0.1)
        
        # Apply smoothing if requested
        if smoothing > 0:
            # Convert to open3d for smoothing, then back to trimesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()
            
            # Apply Laplacian smoothing
            o3d_mesh = o3d_mesh.filter_smooth_simple(number_of_iterations=smoothing)
            
            # Convert back to trimesh
            mesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles))
        
        # Compute normals
        mesh.vertex_normals
        
        # Close boundaries if requested
        if close_boundaries and not mesh.is_watertight:
            print(f"    - Closing mesh boundaries for label {int(label)} using {hole_filling_method} method")
            mesh = close_mesh_boundaries(mesh, method=hole_filling_method)
        
        # Store label name in metadata
        info = label_info.get(int(label))
        if info:
            mesh.metadata['label_name'] = info['name']
            mesh.metadata['label_id'] = int(label)
        else:
            mesh.metadata['label_name'] = f'label_{int(label)}'
            mesh.metadata['label_id'] = int(label)

        if textured_voxels is not None and textured_voxels.shape[:3] == data.shape:
            # Convert world coordinates back to voxel indices
            voxel_indices = (verts / np.array(spacing)).astype(int)

            # Clamp indices to be within the bounds of the textured_voxels array
            max_indices = np.array(textured_voxels.shape[:3]) - 1
            voxel_indices = np.clip(voxel_indices, 0, max_indices)

            # Look up colors from the textured volume
            vertex_colors = textured_voxels[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
            
            mesh.visual.vertex_colors = vertex_colors.astype(np.uint8)
        else:
            info = label_info.get(int(label))
            if info:
                # Set per-vertex colors for proper OBJ export
                n_vertices = len(mesh.vertices)
                vertex_colors = np.tile(info['color'], (n_vertices, 1)).astype(np.uint8)
                mesh.visual.vertex_colors = vertex_colors
            else:
                # Generate random color
                np.random.seed(int(label))  # Ensure consistent colors for same labels
                random_color = np.random.randint(0, 256, 3)
                n_vertices = len(mesh.vertices)
                vertex_colors = np.tile(random_color, (n_vertices, 1)).astype(np.uint8)
                mesh.visual.vertex_colors = vertex_colors

        meshes.append(mesh)

    return meshes


def process_directory(input_dir: str, output_dir: str, threshold: float,
                      smoothing: int, decimation: float, verbose: bool = False,
                      close_boundaries: bool = False, hole_filling_method: str = 'convex'):
    os.makedirs(output_dir, exist_ok=True)
    nifti_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    label_info = load_label_info('vista3d_label_colors.json')

    if not nifti_files:
        print(f"No NIfTI files found in {input_dir}.", file=sys.stderr)
        return

    for fpath in nifti_files:
        base_name = os.path.splitext(os.path.splitext(os.path.basename(fpath))[0])[0]
        out_path = os.path.join(output_dir, f"{base_name}.obj")
        textured_npy_path = os.path.join(
            os.path.dirname(output_dir), "npy", f"{base_name}.npy"
        )
        textured_voxels = None

        try:
            if os.path.exists(textured_npy_path):
                textured_voxels = np.load(textured_npy_path)
                if verbose:
                    print(f"[*] Loaded textured voxels from '{textured_npy_path}'")
            else:
                if verbose:
                    print(f"[!] No textured voxels found at '{textured_npy_path}', using default coloring.")

            if verbose:
                print(f"[*] Reading '{fpath}' …")
            nib_img = nib.load(fpath)
            
            if base_name == 'all' and label_info:
                if verbose:
                    print(f"[+] Converting multi-label file '{fpath}' to meshes…")
                meshes = nifti_to_multilabel_mesh(nib_img, label_info, smoothing, decimation, textured_voxels, close_boundaries, hole_filling_method)
                
                if verbose:
                    print(f"[+] Writing {len(meshes)} meshes to OBJ files…")
                
                # Export each mesh as a separate OBJ file
                for i, mesh in enumerate(meshes):
                    label_name = mesh.metadata.get('label_name', f'label_{i}')
                    # Sanitize filename
                    safe_name = "".join(c for c in label_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_name = safe_name.replace(' ', '_')
                    mesh_out_path = os.path.join(output_dir, f"{base_name}_{safe_name}.obj")
                    
                    # Export OBJ with materials
                    mesh.export(mesh_out_path, file_type='obj')
                    
                    if verbose:
                        print(f"    - Exported: {mesh_out_path}")
            else:
                if verbose:
                    print(f"[+] Converting single-label file '{fpath}' to mesh…")
                mesh = nifti_to_mesh(nib_img, threshold, smoothing, decimation, label_info, base_name, textured_voxels, close_boundaries, hole_filling_method)

                if verbose:
                    print(f"[+] Writing OBJ to '{out_path}' …")
                mesh.export(out_path, file_type='obj')

            if verbose:
                print(f"[✓] Done: {out_path}\n")
        except Exception as e:
            print(f"[!] Failed to process '{fpath}': {e}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert .nii.gz volumes into high-quality .obj surface meshes.")
    parser.add_argument("-i", "--input", required=True,
                        help="Directory with input .nii.gz files.")
    parser.add_argument("-o", "--output", required=True,
                        help="Directory to write resulting .obj files.")
    parser.add_argument("-t", "--threshold", type=float, default=0.1,
                        help="Iso-surface threshold for single-label files. Default: 0.1")
    parser.add_argument("-s", "--smoothing", type=int, default=10,
                        help="Number of smoothing iterations. Default: 10")
    parser.add_argument("-d", "--decimation", type=float, default=0.5,
                        help="Mesh decimation fraction (0.0 to 1.0). Default: 0.5")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output.")
    parser.add_argument("--close-boundaries", action="store_true",
                        help="Close boundary holes in clipped meshes to make them watertight.")
    parser.add_argument("--hole-filling", choices=['convex', 'planar'], default='convex',
                        help="Method for filling holes: 'convex' (convex hull) or 'planar' (simple triangulation). Default: convex")
    return parser.parse_args()


def main():
    args = parse_args()
    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        smoothing=args.smoothing,
        decimation=args.decimation,
        verbose=args.verbose,
        close_boundaries=args.close_boundaries,
        hole_filling_method=args.hole_filling
    )


if __name__ == "__main__":
    main()
