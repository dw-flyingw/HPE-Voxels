#!/usr/bin/env python3
"""
obj2model.py

Convert OBJ files directly to model directories with GLTF/GLB files.
Combines the functionality of obj2glb.py and glb2model.py for a streamlined pipeline.

Requirements
------------
- trimesh
- pygltflib
- numpy
- Pillow

Example
-------
    python obj2model.py -i ./output/obj -o ./output/models
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import trimesh
import pygltflib
from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene, Asset, Attributes, Material, Texture, Image as GLTFImage, Sampler
import numpy as np
from PIL import Image


def create_placeholder_texture(organ_name: str, size: int = 512) -> Image.Image:
    """Create a placeholder texture with organ-appropriate color."""
    
    # Organ color mapping
    organ_colors = {
        'heart': (220, 20, 60),      # Crimson
        'liver': (139, 69, 19),      # Saddle brown
        'kidney': (165, 42, 42),     # Brown
        'lung': (255, 182, 193),     # Light pink
        'brain': (220, 220, 220),    # Gainsboro
        'spleen': (128, 0, 32),      # Dark red
        'pancreas': (255, 218, 185), # Peach
        'stomach': (245, 222, 179),  # Wheat
        'colon': (139, 69, 19),      # Saddle brown
        'intestine': (210, 180, 140), # Tan
        'bladder': (255, 215, 0),    # Gold
        'aorta': (255, 69, 0),       # Orange red
        'artery': (255, 140, 0),     # Dark orange
        'vein': (0, 0, 139),         # Dark blue
        'bone': (245, 245, 220),     # Beige
        'hip': (70, 130, 180),       # Steel blue
    }
    
    # Find matching color
    color = (180, 140, 120)  # Default
    organ_lower = organ_name.lower()
    for key, col in organ_colors.items():
        if key in organ_lower:
            color = col
            break
    
    # Create solid color image
    img = Image.new('RGB', (size, size), color)
    return img


def create_gltf_with_uv(mesh: trimesh.Trimesh, output_dir: str, base_name: str) -> bool:
    """
    Create GLTF/GLB files with UV coordinates directly from trimesh.
    
    Args:
        mesh: Trimesh object with UV coordinates
        output_dir: Output directory for the model
        base_name: Base name for the model
        
    Returns:
        True if successful
    """
    try:
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        textures_dir = os.path.join(output_dir, 'textures')
        os.makedirs(textures_dir, exist_ok=True)
        
        # Initialize GLTF
        gltf = GLTF2()
        gltf.asset = Asset(version="2.0", generator="obj2model-pipeline")
        
        # Handle Scene vs Mesh
        meshes_to_process = []
        if hasattr(mesh, 'geometry'):
            for geometry in mesh.geometry.values():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes_to_process.append(geometry)
        elif isinstance(mesh, trimesh.Trimesh):
            meshes_to_process.append(mesh)
        
        if not meshes_to_process:
            print(f"    ✗ No valid meshes found")
            return False
        
        # Combine all meshes if multiple
        if len(meshes_to_process) > 1:
            combined_mesh = trimesh.util.concatenate(meshes_to_process)
            meshes_to_process = [combined_mesh]
        
        all_binary_data = []
        all_buffer_views = []
        all_accessors = []
        all_gltf_meshes = []
        
        buffer_byte_offset = 0
        
        for mesh_obj in meshes_to_process:
            # Get vertices and faces
            vertices = mesh_obj.vertices.astype(np.float32)
            faces = mesh_obj.faces.flatten().astype(np.uint32)
            
            # Vertices buffer
            vertices_buffer_data = vertices.tobytes()
            all_binary_data.append(vertices_buffer_data)
            vertices_buffer_view = BufferView(
                buffer=0,
                byteOffset=buffer_byte_offset,
                byteLength=len(vertices_buffer_data),
                target=34962,
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
            vertices_accessor_index = len(all_accessors) - 1
            buffer_byte_offset += len(vertices_buffer_data)
            
            # UV coordinates (if available)
            uv_accessor_index = None
            has_uvs = hasattr(mesh_obj.visual, 'uv') and mesh_obj.visual.uv is not None and len(mesh_obj.visual.uv) > 0
            
            if has_uvs:
                uvs = mesh_obj.visual.uv.astype(np.float32)
                uvs_buffer_data = uvs.tobytes()
                all_binary_data.append(uvs_buffer_data)
                uvs_buffer_view = BufferView(
                    buffer=0,
                    byteOffset=buffer_byte_offset,
                    byteLength=len(uvs_buffer_data),
                    target=34962,
                )
                all_buffer_views.append(uvs_buffer_view)
                uvs_accessor = Accessor(
                    bufferView=len(all_buffer_views) - 1,
                    byteOffset=0,
                    componentType=5126,  # FLOAT
                    count=len(uvs),
                    type="VEC2",
                    max=uvs.max(axis=0).tolist(),
                    min=uvs.min(axis=0).tolist(),
                )
                all_accessors.append(uvs_accessor)
                uv_accessor_index = len(all_accessors) - 1
                buffer_byte_offset += len(uvs_buffer_data)
            
            # Faces buffer
            faces_buffer_data = faces.tobytes()
            all_binary_data.append(faces_buffer_data)
            faces_buffer_view = BufferView(
                buffer=0,
                byteOffset=buffer_byte_offset,
                byteLength=len(faces_buffer_data),
                target=34963,
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
            faces_accessor_index = len(all_accessors) - 1
            buffer_byte_offset += len(faces_buffer_data)
            
            # Create primitive
            attributes = Attributes(POSITION=vertices_accessor_index)
            if uv_accessor_index is not None:
                attributes.TEXCOORD_0 = uv_accessor_index
            
            primitive = Primitive(
                attributes=attributes,
                indices=faces_accessor_index,
                material=0  # Reference to material
            )
            
            gltf_mesh = Mesh(primitives=[primitive])
            all_gltf_meshes.append(gltf_mesh)
        
        # Create placeholder texture
        texture_path = os.path.join(textures_dir, 'diffuse.png')
        placeholder_img = create_placeholder_texture(base_name)
        placeholder_img.save(texture_path)
        
        # Combine all binary data
        combined_binary_data = b''.join(all_binary_data)
        
        # Set up GLTF structure
        gltf.buffers.append(Buffer(byteLength=len(combined_binary_data)))
        gltf.bufferViews = all_buffer_views
        gltf.accessors = all_accessors
        gltf.meshes = all_gltf_meshes
        
        # Create material with texture
        gltf.materials = [Material(
            pbrMetallicRoughness={
                'baseColorTexture': {'index': 0},
                'metallicFactor': 0.0,
                'roughnessFactor': 0.8
            },
            doubleSided=True
        )]
        
        # Add texture
        gltf.textures = [Texture(source=0, sampler=0)]
        gltf.images = [GLTFImage(uri='textures/diffuse.png')]
        gltf.samplers = [Sampler(
            magFilter=9729,  # LINEAR
            minFilter=9987,  # LINEAR_MIPMAP_LINEAR
            wrapS=10497,     # REPEAT
            wrapT=10497      # REPEAT
        )]
        
        # Create nodes and scene
        gltf.nodes = []
        gltf.scenes.append(Scene(nodes=[]))
        for i in range(len(all_gltf_meshes)):
            node = Node(mesh=i)
            gltf.nodes.append(node)
            gltf.scenes[0].nodes.append(len(gltf.nodes) - 1)
        
        # Save GLB file
        glb_path = os.path.join(output_dir, 'scene.glb')
        gltf.set_binary_blob(combined_binary_data)
        gltf.save(glb_path)
        
        # Save GLTF file (with separate binary)
        gltf_path = os.path.join(output_dir, 'scene.gltf')
        
        # Update buffer to reference external .bin file
        gltf.buffers[0].uri = 'scene.bin'
        
        # Save binary data separately
        bin_path = os.path.join(output_dir, 'scene.bin')
        with open(bin_path, 'wb') as f:
            f.write(combined_binary_data)
        
        # Save GLTF
        gltf.save(gltf_path)
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error creating GLTF: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_obj_to_model(
    obj_path: str,
    models_dir: str,
    verbose: bool = False
) -> bool:
    """
    Process a single OBJ file to create a model directory.
    
    Args:
        obj_path: Path to OBJ file
        models_dir: Base models directory
        verbose: Verbose output
        
    Returns:
        True if successful
    """
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    output_dir = os.path.join(models_dir, base_name)
    
    try:
        if verbose:
            print(f"\n[*] Processing '{base_name}'")
        else:
            print(f"[*] Processing '{base_name}'...", end=' ')
        
        # Load OBJ with trimesh
        mesh = trimesh.load(obj_path, force='scene', process=False)
        
        # Check for UV coordinates
        has_uvs = False
        if isinstance(mesh, trimesh.Trimesh):
            has_uvs = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        elif hasattr(mesh, 'geometry'):
            for geom in mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    if hasattr(geom.visual, 'uv') and geom.visual.uv is not None and len(geom.visual.uv) > 0:
                        has_uvs = True
                        break
        
        if verbose and has_uvs:
            print(f"    ✓ Found UV coordinates")
        elif verbose:
            print(f"    ⚠️  No UV coordinates found")
        
        # Create GLTF/GLB with UV support
        if create_gltf_with_uv(mesh, output_dir, base_name):
            # Count files
            num_files = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
            num_files += len([f for f in os.listdir(os.path.join(output_dir, 'textures')) if os.path.isfile(os.path.join(output_dir, 'textures', f))])
            
            if verbose:
                print(f"    ✓ Created model directory: {output_dir}")
                print(f"    ✓ Files: scene.gltf, scene.glb, scene.bin, textures/diffuse.png")
            else:
                print(f"✓")
            
            return True
        else:
            if not verbose:
                print(f"✗")
            return False
            
    except Exception as e:
        if not verbose:
            print(f"✗")
        print(f"    ✗ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def process_directory(input_dir: str, output_dir: str, verbose: bool = False):
    """Process all OBJ files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    obj_files = sorted(glob.glob(os.path.join(input_dir, "*.obj")))
    
    if not obj_files:
        print(f"✗ No OBJ files found in {input_dir}")
        return
    
    print(f"Found {len(obj_files)} OBJ file(s) to convert\n")
    
    successful = 0
    failed = 0
    
    for obj_path in obj_files:
        if process_obj_to_model(obj_path, output_dir, verbose):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Conversion complete: {successful}/{len(obj_files)} successful")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"{'='*70}")
    
    if successful > 0:
        print(f"\nModel directories created in: {output_dir}/")
        for obj_file in obj_files:
            base_name = os.path.splitext(os.path.basename(obj_file))[0]
            model_dir = os.path.join(output_dir, base_name)
            if os.path.exists(model_dir):
                num_files = sum([len(files) for _, _, files in os.walk(model_dir)])
                print(f"  {base_name}/ ({num_files} files)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OBJ files directly to model directories with GLTF/GLB"
    )
    parser.add_argument(
        "-i", "--input",
        default="./output/obj",
        help="Directory with input .obj files (default: ./output/obj)"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output/models",
        help="Directory for output model folders (default: ./output/models)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"✗ Error: Input directory '{args.input}' does not exist")
        sys.exit(1)
    
    print("="*70)
    print("OBJ to Model Directory Converter")
    print("="*70)
    print()
    
    process_directory(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()

