#!/usr/bin/env python3
"""
UV Unwrapped Texture Generator for Medical Organ Models
Creates anatomically accurate textures that properly map to UV coordinates
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import trimesh
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse


def extract_gltf_uv_coordinates(gltf_path: str) -> np.ndarray:
    """
    Extract UV coordinates directly from GLTF file for accurate texture mapping.
    
    Args:
        gltf_path: Path to GLTF file
    
    Returns:
        np.ndarray: UV coordinates (N x 2) or empty array if not found
    """
    try:
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        uvs = []
        
        # Look for TEXCOORD_0 accessor in meshes
        if 'meshes' in gltf_data:
            for mesh in gltf_data['meshes']:
                for primitive in mesh.get('primitives', []):
                    attributes = primitive.get('attributes', {})
                    
                    # Look for TEXCOORD_0 attribute
                    if 'TEXCOORD_0' in attributes:
                        texcoord_index = attributes['TEXCOORD_0']
                        accessor = gltf_data['accessors'][texcoord_index]
                        
                        # Load UV data from buffer
                        buffer_view_index = accessor['bufferView']
                        buffer_view = gltf_data['bufferViews'][buffer_view_index]
                        buffer_index = buffer_view['buffer']
                        buffer_info = gltf_data['buffers'][buffer_index]
                        
                        # Read buffer data
                        buffer_path = os.path.join(model_folder, buffer_info['uri'])
                        if not os.path.exists(buffer_path):
                            print(f"Warning: Buffer file not found: {buffer_path}")
                            continue
                            
                        with open(buffer_path, 'rb') as f:
                            buffer_data = f.read()
                        
                        # Extract UV coordinates
                        byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                        uv_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 8]  # 2 floats * 4 bytes each
                        
                        # Convert to numpy array (assuming FLOAT32)
                        uv_array = np.frombuffer(uv_data, dtype=np.float32)
                        uv_array = uv_array.reshape(-1, 2)  # Each UV pair has 2 components
                        
                        uvs.extend(uv_array)
        
        return np.array(uvs) if uvs else np.array([])
    except Exception as e:
        print(f"Error extracting UVs from GLTF: {e}")
        return np.array([])


def create_uv_unwrap_visualization(uvs: np.ndarray, size: int = 1024) -> Image.Image:
    """
    Create a visualization of the UV unwrap layout.
    
    Args:
        uvs: UV coordinates array (N x 2)
        size: Output image size
    
    Returns:
        PIL Image showing the UV layout
    """
    # Create base image
    img = Image.new('RGB', (size, size), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    if len(uvs) == 0:
        return img
    
    # Convert UV coordinates to image coordinates
    # UV coordinates are in [0,1] range
    uv_coords = uvs * size
    uv_coords = np.clip(uv_coords, 0, size - 1).astype(int)
    
    # Draw UV points
    for u, v in uv_coords:
        # Draw small circle for each UV coordinate
        radius = 1
        draw.ellipse([u-radius, v-radius, u+radius, v+radius], fill=(200, 200, 200))
    
    # Create wireframe by connecting nearby points
    print(f"Creating UV wireframe with {len(uv_coords)} points...")
    for i, (u1, v1) in enumerate(uv_coords):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing point {i}/{len(uv_coords)}")
        
        # Find nearby points and connect them
        for j, (u2, v2) in enumerate(uv_coords[i+1:min(i+10, len(uv_coords))]):
            distance = np.sqrt((u1-u2)**2 + (v1-v2)**2)
            if distance < 20:  # Connect nearby points
                draw.line([u1, v1, u2, v2], fill=(150, 150, 150), width=1)
    
    return img


def get_anatomical_tissue_properties(organ_name: str) -> dict:
    """Get realistic anatomical tissue properties for the organ."""
    tissue_properties = {
        "heart": {
            "color_base": [180, 60, 60],
            "color_variation": 25,
            "vascular_density": 0.8,
            "texture_scale": 1.2,
            "surface_roughness": 0.6,
        },
        "liver": {
            "color_base": [150, 100, 80],
            "color_variation": 30,
            "vascular_density": 0.9,
            "texture_scale": 1.0,
            "surface_roughness": 0.7,
        },
        "colon": {
            "color_base": [170, 120, 100],
            "color_variation": 35,
            "vascular_density": 0.6,
            "texture_scale": 1.1,
            "surface_roughness": 0.8,
        },
        "aorta": {
            "color_base": [200, 80, 80],
            "color_variation": 20,
            "vascular_density": 0.3,
            "texture_scale": 0.8,
            "surface_roughness": 0.4,
        },
        "left_iliac_artery": {
            "color_base": [220, 70, 70],
            "color_variation": 25,
            "vascular_density": 0.3,
            "texture_scale": 0.8,
            "surface_roughness": 0.4,
        },
        "right_iliac_artery": {
            "color_base": [220, 70, 70],
            "color_variation": 25,
            "vascular_density": 0.3,
            "texture_scale": 0.8,
            "surface_roughness": 0.4,
        },
        "left_hip": {
            "color_base": [220, 200, 180],
            "color_variation": 15,
            "vascular_density": 0.2,
            "texture_scale": 1.5,
            "surface_roughness": 0.9,
        },
        "right_hip": {
            "color_base": [220, 200, 180],
            "color_variation": 15,
            "vascular_density": 0.2,
            "texture_scale": 1.5,
            "surface_roughness": 0.9,
        },
    }
    
    # Find the best match
    organ_lower = organ_name.lower()
    
    # Direct match
    if organ_lower in tissue_properties:
        return tissue_properties[organ_lower]
    
    # Partial matches
    for key, props in tissue_properties.items():
        if key in organ_lower or organ_lower in key:
            return props
    
    # Default to soft tissue
    return {
        "color_base": [180, 140, 120],
        "color_variation": 30,
        "vascular_density": 0.5,
        "texture_scale": 1.0,
        "surface_roughness": 0.6,
    }


def paint_anatomical_texture_on_uv(uvs: np.ndarray, organ_name: str, size: int = 1024) -> Image.Image:
    """
    Paint anatomically accurate texture onto the UV unwrap layout.
    
    Args:
        uvs: UV coordinates array (N x 2)
        organ_name: Name of the organ for tissue properties
        size: Output texture size
    
    Returns:
        PIL Image with anatomical texture painted on UV layout
    """
    # Get tissue properties
    props = get_anatomical_tissue_properties(organ_name)
    base_color = props["color_base"]
    variation = props["color_variation"]
    vascular_density = props["vascular_density"]
    
    print(f"Creating anatomical texture for {organ_name}...")
    print(f"Base color: RGB{base_color}, Variation: {variation}")
    
    # Create base texture
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Apply base color
    for i in range(3):
        texture[:, :, i] *= base_color[i] / 255.0
    
    # Create UV mask - areas where UV coordinates exist
    uv_mask = np.zeros((size, size), dtype=np.float32)
    
    if len(uvs) > 0:
        # Convert UV coordinates to image coordinates
        uv_coords = uvs * size
        uv_coords = np.clip(uv_coords, 0, size - 1).astype(int)
        
        # Create filled regions around UV points
        for u, v in uv_coords:
            # Create small filled area around each UV point
            radius = 3
            y_start = max(0, v - radius)
            y_end = min(size, v + radius + 1)
            x_start = max(0, u - radius)
            x_end = min(size, u + radius + 1)
            
            uv_mask[y_start:y_end, x_start:x_end] = 1.0
        
        # Dilate the mask to fill gaps
        from scipy import ndimage
        uv_mask = ndimage.binary_dilation(uv_mask, iterations=5).astype(np.float32)
    else:
        # If no UVs, fill entire texture
        uv_mask.fill(1.0)
    
    # Add color variations within UV regions
    color_noise = np.random.normal(0, variation / 255.0, (size, size, 3))
    texture += color_noise * uv_mask[:, :, np.newaxis]
    
    # Add anatomical surface details
    # Create tissue-specific patterns
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    if "heart" in organ_name.lower():
        # Cardiac muscle patterns
        muscle_pattern = (
            np.sin(x * np.pi * 30) * 0.1 +
            np.sin(y * np.pi * 25) * 0.1
        )
        texture[:, :, 0] += muscle_pattern * uv_mask
    
    elif "liver" in organ_name.lower():
        # Hepatic lobular pattern
        liver_pattern = (
            np.sin(x * np.pi * 20) * 0.05 +
            np.sin(y * np.pi * 20) * 0.05 +
            np.sin((x + y) * np.pi * 15) * 0.03
        )
        texture[:, :, 1] += liver_pattern * uv_mask
    
    elif "colon" in organ_name.lower():
        # Haustra and muscle patterns
        haustra_pattern = (
            np.sin(y * np.pi * 12) * 0.15 +
            np.sin(x * np.pi * 8) * 0.1
        )
        texture[:, :, :] += haustra_pattern[:, :, np.newaxis] * uv_mask[:, :, np.newaxis]
    
    elif "aorta" in organ_name.lower() or "artery" in organ_name.lower():
        # Smooth muscle and endothelial patterns
        vessel_pattern = np.sin(x * np.pi * 15) * 0.08
        texture[:, :, 0] += vessel_pattern * uv_mask
        texture[:, :, 1] -= vessel_pattern * 0.5 * uv_mask
    
    elif "hip" in organ_name.lower():
        # Bone trabecular patterns
        bone_pattern = (
            np.sin(x * np.pi * 40) * 0.1 +
            np.sin(y * np.pi * 35) * 0.1 +
            np.sin((x + y) * np.pi * 25) * 0.08
        )
        texture[:, :, :] -= bone_pattern[:, :, np.newaxis] * 0.1 * uv_mask[:, :, np.newaxis]
    
    # Add vascular network if appropriate
    if vascular_density > 0.3:
        # Create simple vascular patterns
        num_vessels = int(vascular_density * 10)
        for _ in range(num_vessels):
            # Random vessel path
            start_x = np.random.randint(0, size)
            start_y = np.random.randint(0, size)
            
            vessel_length = np.random.randint(size//8, size//4)
            direction = np.random.random() * 2 * np.pi
            
            for step in range(vessel_length):
                x_pos = int(start_x + step * np.cos(direction) + np.random.normal(0, 2))
                y_pos = int(start_y + step * np.sin(direction) + np.random.normal(0, 2))
                
                if 0 <= x_pos < size and 0 <= y_pos < size and uv_mask[y_pos, x_pos] > 0:
                    # Add reddish vessel color
                    vessel_radius = 2
                    for dy in range(-vessel_radius, vessel_radius + 1):
                        for dx in range(-vessel_radius, vessel_radius + 1):
                            nx, ny = x_pos + dx, y_pos + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                if dx*dx + dy*dy <= vessel_radius*vessel_radius:
                                    texture[ny, nx, 0] = min(1.0, texture[ny, nx, 0] + 0.2)
                                    texture[ny, nx, 1] = max(0.0, texture[ny, nx, 1] - 0.1)
                                    texture[ny, nx, 2] = max(0.0, texture[ny, nx, 2] - 0.1)
    
    # Apply mask to only show texture in UV regions
    for i in range(3):
        texture[:, :, i] = texture[:, :, i] * uv_mask + (50/255.0) * (1 - uv_mask)
    
    # Ensure values are in valid range
    texture = np.clip(texture, 0, 1)
    
    # Convert to PIL Image
    texture_uint8 = (texture * 255).astype(np.uint8)
    img = Image.fromarray(texture_uint8)
    
    # Apply final enhancements
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.05)
    
    return img


def process_model_for_uv_texture(model_dir: str, size: int = 1024, overwrite: bool = False) -> bool:
    """
    Process a single model to create UV unwrapped texture.
    
    Args:
        model_dir: Path to model directory
        size: Texture size
        overwrite: Whether to overwrite existing textures
    
    Returns:
        True if successful
    """
    model_name = os.path.basename(model_dir)
    
    print(f"\n[*] Processing '{model_name}' for UV unwrapped texture...")
    
    # Check if texture already exists
    texture_path = os.path.join(model_dir, "textures", "diffuse.png")
    if os.path.exists(texture_path) and not overwrite:
        print(f"    ✓ Texture already exists: {texture_path}")
        return True
    
    try:
        # Find GLTF file
        gltf_path = os.path.join(model_dir, "scene.gltf")
        if not os.path.exists(gltf_path):
            print(f"    ✗ No scene.gltf found in {model_dir}")
            return False
        
        # Extract UV coordinates
        print("    - Extracting UV coordinates from GLTF...")
        uvs = extract_gltf_uv_coordinates(gltf_path)
        
        if len(uvs) == 0:
            print("    ⚠️ No UV coordinates found, creating basic texture")
        else:
            print(f"    ✓ Found {len(uvs)} UV coordinates")
        
        # Create UV unwrap visualization (for debugging)
        uv_viz = create_uv_unwrap_visualization(uvs, size)
        uv_viz_path = os.path.join(model_dir, "textures", "uv_unwrap_debug.png")
        os.makedirs(os.path.dirname(uv_viz_path), exist_ok=True)
        uv_viz.save(uv_viz_path)
        print(f"    ✓ Saved UV unwrap visualization: {uv_viz_path}")
        
        # Generate anatomical texture mapped to UV layout
        print("    - Generating anatomical texture...")
        texture_img = paint_anatomical_texture_on_uv(uvs, model_name, size)
        
        # Save texture
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        texture_img.save(texture_path)
        print(f"    ✓ Saved UV unwrapped texture: {texture_path}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all models or specific organ."""
    parser = argparse.ArgumentParser(description="Generate UV unwrapped anatomical textures")
    parser.add_argument("--organ", help="Specific organ to process")
    parser.add_argument("--size", type=int, default=1024, help="Texture size (default: 1024)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing textures")
    
    args = parser.parse_args()
    
    models_dir = "output/models"
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found")
        return 1
    
    if args.organ:
        # Process specific organ
        organ_dir = os.path.join(models_dir, args.organ)
        if not os.path.exists(organ_dir):
            print(f"Error: Organ directory '{organ_dir}' not found")
            return 1
        
        success = process_model_for_uv_texture(organ_dir, args.size, args.overwrite)
        return 0 if success else 1
    else:
        # Process all models except nyctalus-noctula (it already has good texture)
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) and d != "nyctalus-noctula"]
        
        if not model_dirs:
            print(f"No model directories found in '{models_dir}'")
            return 1
        
        print(f"Found {len(model_dirs)} model directories to process")
        print("Skipping nyctalus-noctula (already has proper UV texture)")
        
        successful = 0
        failed = 0
        
        for model_dir in sorted(model_dirs):
            model_path = os.path.join(models_dir, model_dir)
            if process_model_for_uv_texture(model_path, args.size, args.overwrite):
                successful += 1
            else:
                failed += 1
        
        print(f"\n[*] Summary: {successful} successful, {failed} failed")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
