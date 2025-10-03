#!/usr/bin/env python3
"""
obj2glb.py

A utility to convert .obj files into .glb binary format.
GLB (GL Transmission Format Binary) embeds all mesh data and textures
into a single binary file, making it ideal for web viewing and distribution.

Requirements
------------
- trimesh
- pygltflib

Example
-------
    python obj2glb.py -i ./output/obj -o ./output/glb
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import trimesh
import pygltflib
from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene, Asset, Attributes
import numpy as np


def _write_glb_mesh(trimesh_obj, output_path):
    """Converts a trimesh object to a GLB file using pygltflib."""
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0", generator="trimesh-pygltflib")

    # Handle Scene (multiple meshes) or single Mesh
    meshes_to_process = []
    if isinstance(trimesh_obj, trimesh.Scene):
        for geometry in trimesh_obj.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes_to_process.append(geometry)
    elif isinstance(trimesh_obj, trimesh.Trimesh):
        meshes_to_process.append(trimesh_obj)
    else:
        raise ValueError("Unsupported trimesh object type for GLB export.")

    if not meshes_to_process:
        raise ValueError("No valid meshes found in the loaded object.")

    all_gltf_meshes = []
    all_binary_data = []
    all_buffer_views = []
    all_accessors = []
    
    buffer_byte_offset = 0

    for i, mesh in enumerate(meshes_to_process):
        # Get vertices and faces
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.flatten().astype(np.uint32)

        # Vertices buffer
        vertices_buffer_data = vertices.tobytes()
        all_binary_data.append(vertices_buffer_data)
        vertices_buffer_view = BufferView(
            buffer=0,
            byteOffset=buffer_byte_offset,
            byteLength=len(vertices_buffer_data),
            target=34962,  # ARRAY_BUFFER
        )
        all_buffer_views.append(vertices_buffer_view)
        vertices_accessor = Accessor(
            bufferView=len(all_buffer_views) - 1,
            byteOffset=0,
            componentType=5126,  # FLOAT
            count=len(vertices),
            type="VEC3",
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
        )
        all_accessors.append(vertices_accessor)
        buffer_byte_offset += len(vertices_buffer_data)

        # Faces buffer
        faces_buffer_data = faces.tobytes()
        all_binary_data.append(faces_buffer_data)
        faces_buffer_view = BufferView(
            buffer=0,
            byteOffset=buffer_byte_offset,
            byteLength=len(faces_buffer_data),
            target=34963,  # ELEMENT_ARRAY_BUFFER
        )
        all_buffer_views.append(faces_buffer_view)
        faces_accessor = Accessor(
            bufferView=len(all_buffer_views) - 1,
            byteOffset=0,
            componentType=5125,  # UNSIGNED_INT
            count=len(faces),
            type="SCALAR",
        )
        all_accessors.append(faces_accessor)
        buffer_byte_offset += len(faces_buffer_data)

        # Create primitive
        primitive = Primitive(
            attributes=Attributes(POSITION=len(all_accessors) - 2),  # -2 because we just added faces accessor
            indices=len(all_accessors) - 1
        )

        # Create GLTF mesh
        gltf_mesh = Mesh(primitives=[primitive])
        all_gltf_meshes.append(gltf_mesh)

    # Concatenate all binary data
    combined_binary_data = b''.join(all_binary_data)
    
    # Set up GLTF structure
    gltf.buffers.append(Buffer(byteLength=len(combined_binary_data)))
    gltf.bufferViews = all_buffer_views
    gltf.accessors = all_accessors
    gltf.meshes = all_gltf_meshes

    # Create nodes and scene
    gltf.nodes = []
    gltf.scenes.append(Scene(nodes=[]))
    for i, gltf_mesh in enumerate(all_gltf_meshes):
        node = Node(mesh=i)
        gltf.nodes.append(node)
        gltf.scenes[0].nodes.append(len(gltf.nodes) - 1)

    # Save GLB file
    gltf.set_binary_blob(combined_binary_data)
    gltf.save(output_path)


def process_directory(input_dir: str, output_dir: str, verbose: bool = False):
    """Processes a directory of OBJ files to generate GLB meshes."""
    os.makedirs(output_dir, exist_ok=True)
    obj_files = glob.glob(os.path.join(input_dir, "*.obj"))

    if not obj_files:
        print(f"No OBJ files found in {input_dir}.", file=sys.stderr)
        return

    print(f"Found {len(obj_files)} OBJ files to convert...")

    for fpath in obj_files:
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(output_dir, f"{base_name}.glb")

        try:
            if verbose:
                print(f"[*] Reading '{fpath}' …")
            
            # Load OBJ file with trimesh
            scene = trimesh.load(fpath, force='scene')
            
            if verbose:
                print(f"[+] Writing GLB to '{out_path}' …")
            
            _write_glb_mesh(scene, out_path)
            
            if verbose:
                file_size = os.path.getsize(out_path) / 1024  # KB
                print(f"[✓] Done: {out_path} ({file_size:.2f} KB)\n")
            else:
                print(f"[✓] Converted: {base_name}.obj → {base_name}.glb")

        except Exception as e:
            print(f"[!] Failed to process '{fpath}': {e}", file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OBJ files into .glb binary format.")
    parser.add_argument("-i", "--input", 
                        default="./output/obj",
                        help="Directory with input .obj files (default: ./output/obj)")
    parser.add_argument("-o", "--output", 
                        default="./output/glb",
                        help="Directory to write resulting .glb files (default: ./output/glb)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Converting OBJ files from '{args.input}' to '{args.output}'")
    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        verbose=args.verbose
    )
    print("Conversion complete!")


if __name__ == "__main__":
    main()
