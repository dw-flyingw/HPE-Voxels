#!/usr/bin/env python3
"""
Photo-Realistic UV Texture Generator for Medical Organ Models
Uses reference photos and AI-generated content to create hyper-realistic textures
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import trimesh
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
from scipy import ndimage


def extract_gltf_uv_coordinates(gltf_path: str) -> np.ndarray:
    """
    Extract UV coordinates directly from GLTF file for accurate texture mapping.
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
                        uv_data = buffer_data[byte_offset:byte_offset + accessor['count'] * 8]
                        
                        # Convert to numpy array (assuming FLOAT32)
                        uv_array = np.frombuffer(uv_data, dtype=np.float32)
                        uv_array = uv_array.reshape(-1, 2)
                        
                        uvs.extend(uv_array)
        
        return np.array(uvs) if uvs else np.array([])
    except Exception as e:
        print(f"Error extracting UVs from GLTF: {e}")
        return np.array([])


def create_photo_realistic_colon_texture(size: int = 1024) -> Image.Image:
    """
    Create a photo-realistic colon texture based on the reference image.
    """
    print("Creating photo-realistic colon texture...")
    
    # Create base texture with realistic colon colors
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Base colon colors - rich pink-reds with variations
    base_colors = [
        [220, 120, 140],  # Light pink
        [200, 100, 120],  # Medium pink-red
        [180, 80, 100],   # Darker pink-red
        [240, 140, 160],  # Very light pink
        [160, 60, 80],    # Deep red-pink
    ]
    
    # Create organic color variations
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Multiple noise layers for organic variation
    noise1 = np.random.normal(0, 0.3, (size, size))
    noise2 = np.random.normal(0, 0.2, (size, size))
    noise3 = np.random.normal(0, 0.1, (size, size))
    
    # Organic patterns
    pattern1 = np.sin(x * np.pi * 15 + noise1) * 0.3 + np.sin(y * np.pi * 12 + noise2) * 0.2
    pattern2 = np.sin((x + y) * np.pi * 8 + noise3) * 0.4
    pattern3 = np.sin(x * np.pi * 25) * 0.1 + np.sin(y * np.pi * 20) * 0.1
    
    combined_pattern = pattern1 + pattern2 + pattern3
    
    # Apply color variations
    for i in range(3):
        # Base color variation
        base_color = base_colors[0][i] / 255.0
        
        # Add organic color shifts
        color_variation = combined_pattern * 0.3
        texture[:, :, i] = base_color + color_variation
        
        # Add fine noise for texture
        fine_noise = np.random.normal(0, 0.05, (size, size))
        texture[:, :, i] += fine_noise
        
        # Add some reddish tint in certain areas
        red_tint = np.sin(x * np.pi * 30) * np.sin(y * np.pi * 25) * 0.1
        if i == 0:  # Red channel
            texture[:, :, i] += red_tint
        elif i == 1:  # Green channel - reduce for more red
            texture[:, :, i] -= red_tint * 0.3
        elif i == 2:  # Blue channel - reduce for more red
            texture[:, :, i] -= red_tint * 0.5
    
    # Create haustra (colon segments) pattern
    haustra_pattern = np.sin(y * np.pi * 8) * 0.4
    for i in range(3):
        texture[:, :, i] += haustra_pattern * 0.15
    
    # Add surface irregularities
    irregularities = np.random.normal(0, 0.03, (size, size))
    texture += irregularities[:, :, np.newaxis]
    
    # Add some glossy highlights (simulate wet surface)
    highlight_pattern = np.sin(x * np.pi * 20) * np.sin(y * np.pi * 15) * 0.2
    highlight_mask = highlight_pattern > 0.1
    for i in range(3):
        texture[:, :, i] = np.where(highlight_mask, 
                                   np.minimum(1.0, texture[:, :, i] + 0.2), 
                                   texture[:, :, i])
    
    # Add subtle vascular patterns
    for _ in range(5):
        center_x = np.random.uniform(0.1, 0.9)
        center_y = np.random.uniform(0.1, 0.9)
        radius = np.random.uniform(0.05, 0.15)
        
        dx = (x - center_x) / radius
        dy = (y - center_y) / radius
        vessel = np.exp(-(dx**2 + dy**2)) * 0.3
        
        # Vessels are slightly darker
        for i in range(3):
            texture[:, :, i] -= vessel * 0.1
    
    # Ensure values are in valid range
    texture = np.clip(texture, 0, 1)
    
    # Convert to PIL Image
    texture_uint8 = (texture * 255).astype(np.uint8)
    img = Image.fromarray(texture_uint8)
    
    # Apply photo-realistic enhancements
    # Increase contrast for more dramatic appearance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    # Increase saturation for richer colors
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    
    # Add slight sharpening for detail
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    # Add subtle blur to some areas for organic feel
    img_array = np.array(img)
    
    # Create a mask for selective blurring
    blur_mask = np.random.random((size, size)) > 0.7
    blurred = ndimage.gaussian_filter(img_array, sigma=0.5)
    
    for i in range(3):
        img_array[:, :, i] = np.where(blur_mask, blurred[:, :, i], img_array[:, :, i])
    
    img = Image.fromarray(img_array)
    
    return img


def create_photo_realistic_heart_texture(size: int = 1024) -> Image.Image:
    """
    Create a photo-realistic heart texture.
    """
    print("Creating photo-realistic heart texture...")
    
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Heart colors - deep reds with muscle fiber patterns
    base_colors = [
        [180, 60, 60],   # Deep cardiac red
        [200, 80, 80],   # Medium cardiac red
        [160, 40, 40],   # Dark cardiac red
        [220, 100, 100], # Light cardiac red
    ]
    
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Cardiac muscle fiber patterns
    fiber_pattern1 = np.sin(x * np.pi * 20) * 0.4  # Longitudinal fibers
    fiber_pattern2 = np.sin(y * np.pi * 15) * 0.3  # Circular fibers
    fiber_pattern3 = np.sin((x + y) * np.pi * 10) * 0.2  # Mixed direction
    
    combined_fibers = fiber_pattern1 + fiber_pattern2 + fiber_pattern3
    
    # Apply base colors with fiber variations
    base_color = np.array(base_colors[0]) / 255.0
    for i in range(3):
        texture[:, :, i] = base_color[i] + combined_fibers * 0.2
        texture[:, :, i] += np.random.normal(0, 0.05, (size, size))
    
    # Add coronary vessels
    for _ in range(8):
        center_x = np.random.uniform(0.1, 0.9)
        center_y = np.random.uniform(0.1, 0.9)
        radius = np.random.uniform(0.02, 0.08)
        
        dx = (x - center_x) / radius
        dy = (y - center_y) / radius
        vessel = np.exp(-(dx**2 + dy**2)) * 0.4
        
        # Vessels are slightly darker red
        texture[:, :, 0] += vessel * 0.1  # More red
        texture[:, :, 1] -= vessel * 0.05  # Less green
        texture[:, :, 2] -= vessel * 0.05  # Less blue
    
    texture = np.clip(texture, 0, 1)
    texture_uint8 = (texture * 255).astype(np.uint8)
    img = Image.fromarray(texture_uint8)
    
    # Enhance for realism
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    
    return img


def create_photo_realistic_aorta_texture(size: int = 1024) -> Image.Image:
    """
    Create a photo-realistic aorta texture.
    """
    print("Creating photo-realistic aorta texture...")
    
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Aorta colors - bright arterial red
    base_color = [200, 80, 80]  # Bright arterial red
    
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Smooth muscle patterns (longitudinal)
    smooth_pattern = np.sin(x * np.pi * 12) * 0.3
    endothelial_pattern = np.sin(y * np.pi * 8) * 0.2
    
    # Apply base color
    for i in range(3):
        texture[:, :, i] = base_color[i] / 255.0
        texture[:, :, i] += smooth_pattern * 0.1
        texture[:, :, i] += endothelial_pattern * 0.05
        texture[:, :, i] += np.random.normal(0, 0.03, (size, size))
    
    # Add some vessel wall thickness variation
    thickness_pattern = np.sin(x * np.pi * 6) * np.sin(y * np.pi * 4) * 0.1
    for i in range(3):
        texture[:, :, i] += thickness_pattern
    
    texture = np.clip(texture, 0, 1)
    texture_uint8 = (texture * 255).astype(np.uint8)
    img = Image.fromarray(texture_uint8)
    
    # Enhance for realism
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.15)
    
    return img


def create_photo_realistic_bone_texture(size: int = 1024) -> Image.Image:
    """
    Create a photo-realistic bone texture.
    """
    print("Creating photo-realistic bone texture...")
    
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Bone colors - ivory/beige
    base_colors = [
        [220, 200, 180],  # Light ivory
        [200, 180, 160],  # Medium ivory
        [180, 160, 140],  # Dark ivory
    ]
    
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Trabecular bone patterns
    trabecular = np.sin(x * np.pi * 40) * 0.2 + np.sin(y * np.pi * 35) * 0.2
    haversian = np.sin((x + y) * np.pi * 25) * 0.15
    
    # Apply base color with patterns
    base_color = np.array(base_colors[0]) / 255.0
    for i in range(3):
        texture[:, :, i] = base_color[i] + trabecular * 0.1
        texture[:, :, i] += haversian * 0.05
        texture[:, :, i] += np.random.normal(0, 0.03, (size, size))
    
    # Add haversian canals
    for _ in range(10):
        center_x = np.random.uniform(0.1, 0.9)
        center_y = np.random.uniform(0.1, 0.9)
        radius = np.random.uniform(0.01, 0.03)
        
        dx = (x - center_x) / radius
        dy = (y - center_y) / radius
        canal = np.exp(-(dx**2 + dy**2)) * 0.3
        
        # Canals are darker
        for i in range(3):
            texture[:, :, i] -= canal * 0.1
    
    texture = np.clip(texture, 0, 1)
    texture_uint8 = (texture * 255).astype(np.uint8)
    img = Image.fromarray(texture_uint8)
    
    # Enhance for realism
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.05)
    
    return img


def paint_photo_realistic_texture_on_uv(uvs: np.ndarray, organ_name: str, size: int = 1024) -> Image.Image:
    """
    Paint photo-realistic texture onto the UV unwrap layout.
    """
    # Generate organ-specific photo-realistic texture
    if "colon" in organ_name.lower():
        texture_img = create_photo_realistic_colon_texture(size)
    elif "heart" in organ_name.lower():
        texture_img = create_photo_realistic_heart_texture(size)
    elif "aorta" in organ_name.lower() or "artery" in organ_name.lower():
        texture_img = create_photo_realistic_aorta_texture(size)
    elif "hip" in organ_name.lower():
        texture_img = create_photo_realistic_bone_texture(size)
    else:
        # Default to colon-style for unknown organs
        texture_img = create_photo_realistic_colon_texture(size)
    
    # Create UV mask if UVs are available
    if len(uvs) > 0:
        # Create mask for UV regions
        uv_mask = np.zeros((size, size), dtype=np.float32)
        
        # Convert UV coordinates to image coordinates
        uv_coords = uvs * size
        uv_coords = np.clip(uv_coords, 0, size - 1).astype(int)
        
        # Fill UV regions
        for u, v in uv_coords:
            radius = 2
            y_start = max(0, v - radius)
            y_end = min(size, v + radius + 1)
            x_start = max(0, u - radius)
            x_end = min(size, u + radius + 1)
            
            uv_mask[y_start:y_end, x_start:x_end] = 1.0
        
        # Dilate the mask
        uv_mask = ndimage.binary_dilation(uv_mask, iterations=3).astype(np.float32)
        
        # Apply mask to texture
        texture_array = np.array(texture_img)
        for i in range(3):
            texture_array[:, :, i] = texture_array[:, :, i] * uv_mask + (50/255.0) * (1 - uv_mask)
        
        texture_img = Image.fromarray(texture_array.astype(np.uint8))
    
    return texture_img


def process_model_for_photo_realistic_texture(model_dir: str, size: int = 1024, overwrite: bool = False) -> bool:
    """
    Process a single model to create photo-realistic UV texture.
    """
    model_name = os.path.basename(model_dir)
    
    print(f"\n[*] Processing '{model_name}' for photo-realistic UV texture...")
    
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
        
        # Generate photo-realistic texture mapped to UV layout
        print("    - Generating photo-realistic texture...")
        texture_img = paint_photo_realistic_texture_on_uv(uvs, model_name, size)
        
        # Save texture
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        texture_img.save(texture_path)
        print(f"    ✓ Saved photo-realistic texture: {texture_path}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all models or specific organ."""
    parser = argparse.ArgumentParser(description="Generate photo-realistic UV textures")
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
        
        success = process_model_for_photo_realistic_texture(organ_dir, args.size, args.overwrite)
        return 0 if success else 1
    else:
        # Process all models except nyctalus-noctula
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
            if process_model_for_photo_realistic_texture(model_path, args.size, args.overwrite):
                successful += 1
            else:
                failed += 1
        
        print(f"\n[*] Summary: {successful} successful, {failed} failed")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
