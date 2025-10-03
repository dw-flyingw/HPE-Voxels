#!/usr/bin/env python3
"""
glb2model.py - Convert GLB files to model folders

This script takes GLB files from output/glb and creates individual model folders
in output/models, following the structure of the nyctalus-noctula example.

Each model folder will contain:
- scene.glb (the original GLB file)
- scene.gltf (extracted glTF JSON)
- scene.bin (extracted binary data)
- diffuse.png (extracted texture, if present)
"""

import os
import io
import shutil
from pathlib import Path
from pygltflib import GLTF2
from PIL import Image
import json


def create_placeholder_texture(color=(128, 128, 128), size=512):
    """Create a placeholder texture with specified color"""
    # Create a simple colored texture
    img = Image.new('RGB', (size, size), color)
    
    # Add some basic pattern to make it more interesting
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a simple grid pattern
    grid_size = size // 16
    for i in range(0, size, grid_size * 2):
        for j in range(0, size, grid_size * 2):
            draw.rectangle([i, j, i + grid_size, j + grid_size], 
                         fill=(color[0] + 20, color[1] + 20, color[2] + 20))
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


def get_color_for_model(model_name):
    """Get a color for the model based on its name"""
    # Color mapping for common medical/anatomical structures
    color_map = {
        'heart': (220, 20, 60),      # Crimson
        'aorta': (255, 69, 0),       # Red-orange
        'colon': (139, 69, 19),      # Saddle brown
        'left_hip': (70, 130, 180),  # Steel blue
        'right_hip': (70, 130, 180), # Steel blue
        'left_iliac_artery': (255, 140, 0), # Dark orange
        'right_iliac_artery': (255, 140, 0), # Dark orange
    }
    
    # Try to find matching color
    for key, color in color_map.items():
        if key in model_name.lower():
            return color
    
    # Default gray color
    return (128, 128, 128)


def update_gltf_with_texture_reference(gltf_path, model_dir):
    """Update GLTF file to reference the created texture"""
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        # Add image reference
        if 'images' not in gltf_data:
            gltf_data['images'] = []
        
        gltf_data['images'].append({
            "uri": "diffuse.png"
        })
        
        # Add texture reference
        if 'textures' not in gltf_data:
            gltf_data['textures'] = []
        
        gltf_data['textures'].append({
            "source": 0,
            "sampler": 0
        })
        
        # Add sampler if it doesn't exist
        if 'samplers' not in gltf_data:
            gltf_data['samplers'] = []
        
        if not gltf_data['samplers']:
            gltf_data['samplers'].append({
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497
            })
        
        # Update material to use texture
        if 'materials' in gltf_data and gltf_data['materials']:
            material = gltf_data['materials'][0]
            
            # Handle KHR_materials_pbrSpecularGlossiness extension (common in medical models)
            if 'extensions' in material and 'KHR_materials_pbrSpecularGlossiness' in material['extensions']:
                spec_gloss = material['extensions']['KHR_materials_pbrSpecularGlossiness']
                spec_gloss['diffuseTexture'] = {
                    "index": 0,
                    "texCoord": 0
                }
            # Handle standard PBR material
            elif 'pbrMetallicRoughness' not in material:
                material['pbrMetallicRoughness'] = {}
                material['pbrMetallicRoughness']['baseColorTexture'] = {
                    "index": 0,
                    "texCoord": 0
                }
            else:
                material['pbrMetallicRoughness']['baseColorTexture'] = {
                    "index": 0,
                    "texCoord": 0
                }
        
        # Save updated GLTF
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        print("  Updated GLTF to reference texture")
        
    except Exception as e:
        print(f"  Warning: Could not update GLTF with texture reference: {e}")


def extract_textures_from_gltf(gltf, output_dir):
    """Extract textures from glTF and save as diffuse.png, or create placeholder"""
    textures_extracted = False
    
    if not gltf.images:
        print("  No textures found in GLB file - creating placeholder texture")
        # Create a placeholder texture
        model_name = os.path.basename(output_dir)
        color = get_color_for_model(model_name)
        texture_data = create_placeholder_texture(color)
        
        # Save placeholder texture
        with open(os.path.join(output_dir, "diffuse.png"), 'wb') as f:
            f.write(texture_data)
        print(f"  Created placeholder texture: diffuse.png (color: RGB{color})")
        textures_extracted = True
        return textures_extracted
    
    for i, image in enumerate(gltf.images):
        if hasattr(image, 'bufferView') and image.bufferView is not None:
            # Extract image data from buffer view
            buffer_view = gltf.bufferViews[image.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]
            
            # Get image data
            byte_offset = buffer_view.byteOffset or 0
            byte_length = buffer_view.byteLength
            image_data = buffer.data[byte_offset:byte_offset + byte_length]
            
            # Try to load as PIL Image
            try:
                img = Image.open(io.BytesIO(image_data))
                # Save as diffuse.png
                img.save(os.path.join(output_dir, "diffuse.png"))
                print(f"  Extracted texture: diffuse.png")
                textures_extracted = True
                break  # Use first texture found
            except Exception as e:
                print(f"  Warning: Could not extract texture {i}: {e}")
                
        elif hasattr(image, 'uri') and image.uri:
            # External texture reference
            print(f"  Found external texture reference: {image.uri}")
    
    return textures_extracted


def update_gltf_with_texture_reference(gltf_path, model_dir):
    """Update GLTF file to include texture and material references"""
    try:
        # Load the GLTF JSON
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        # Add image reference
        if 'images' not in gltf_data:
            gltf_data['images'] = []
        
        # Add texture reference
        if 'textures' not in gltf_data:
            gltf_data['textures'] = []
        
        # Add material reference
        if 'materials' not in gltf_data:
            gltf_data['materials'] = []
        
        # Add image entry
        image_entry = {
            "uri": "diffuse.png"
        }
        gltf_data['images'].append(image_entry)
        
        # Add texture entry
        texture_entry = {
            "source": len(gltf_data['images']) - 1,
            "sampler": 0
        }
        gltf_data['textures'].append(texture_entry)
        
        # Add sampler if not exists
        if 'samplers' not in gltf_data:
            gltf_data['samplers'] = []
        if not gltf_data['samplers']:
            gltf_data['samplers'].append({
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497
            })
        
        # Add material entry
        material_entry = {
            "doubleSided": True,
            "extensions": {
                "KHR_materials_pbrSpecularGlossiness": {
                    "diffuseFactor": [0.8, 0.8, 0.8, 1.0],
                    "diffuseTexture": {
                        "index": len(gltf_data['textures']) - 1,
                        "texCoord": 0
                    },
                    "glossinessFactor": 0.5,
                    "specularFactor": [0.2, 0.2, 0.2]
                }
            },
            "name": "Material"
        }
        gltf_data['materials'].append(material_entry)
        
        # Update meshes to use the material
        if 'meshes' in gltf_data and gltf_data['meshes']:
            for mesh in gltf_data['meshes']:
                if 'primitives' in mesh and mesh['primitives']:
                    for primitive in mesh['primitives']:
                        primitive['material'] = len(gltf_data['materials']) - 1
        
        # Save updated GLTF
        with open(gltf_path, 'w') as f:
            json.dump(gltf_data, f, indent=2)
        
        print(f"  Updated GLTF with texture and material references")
        
    except Exception as e:
        print(f"  Warning: Could not update GLTF with texture reference: {e}")


def convert_glb_to_model_folder(glb_path, output_base_dir):
    """Convert a single GLB file to a model folder structure"""
    glb_path = Path(glb_path)
    model_name = glb_path.stem
    
    # Create model directory
    model_dir = output_base_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {glb_path.name} to {model_name}/")
    
    try:
        # Load the GLB file
        gltf = GLTF2().load(glb_path)
        
        # Copy original GLB file as scene.glb
        scene_glb_path = model_dir / "scene.glb"
        shutil.copy2(glb_path, scene_glb_path)
        print(f"  Copied: scene.glb")
        
        # Extract glTF JSON and save as scene.gltf
        scene_gltf_path = model_dir / "scene.gltf"
        gltf.save(scene_gltf_path)
        print(f"  Extracted: scene.gltf")
        
        # Extract binary data and save as scene.bin
        if gltf.buffers and len(gltf.buffers) > 0:
            buffer = gltf.buffers[0]
            if hasattr(buffer, 'data') and buffer.data:
                scene_bin_path = model_dir / "scene.bin"
                with open(scene_bin_path, 'wb') as f:
                    f.write(buffer.data)
                print(f"  Extracted: scene.bin")
            else:
                print("  No binary data found in GLB")
        else:
            print("  No buffers found in GLB")
        
        # Extract textures
        has_texture = extract_textures_from_gltf(gltf, model_dir)
        
        # If we created a placeholder texture, update the GLTF to reference it
        if has_texture and not gltf.images:
            update_gltf_with_texture_reference(scene_gltf_path, model_dir)
        
        print(f"  ✓ Successfully converted {model_name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error converting {model_name}: {e}")
        return False


def main():
    """Main function to convert all GLB files to model folders"""
    # Define paths
    input_dir = Path("output/glb")
    output_dir = Path("output/models")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all GLB files
    glb_files = list(input_dir.glob("*.glb"))
    
    if not glb_files:
        print(f"No GLB files found in {input_dir}")
        return
    
    print(f"Found {len(glb_files)} GLB files to convert:")
    for glb_file in glb_files:
        print(f"  - {glb_file.name}")
    
    print(f"\nConverting GLB files to model folders in {output_dir}/")
    print("-" * 50)
    
    success_count = 0
    for glb_file in glb_files:
        if convert_glb_to_model_folder(glb_file, output_dir):
            success_count += 1
        print()  # Empty line between conversions
    
    print("-" * 50)
    print(f"Conversion complete: {success_count}/{len(glb_files)} files converted successfully")
    
    if success_count > 0:
        print(f"\nModel folders created in {output_dir}/:")
        for model_dir in sorted(output_dir.iterdir()):
            if model_dir.is_dir():
                files = list(model_dir.glob("*"))
                print(f"  {model_dir.name}/ ({len(files)} files)")


if __name__ == "__main__":
    main()
