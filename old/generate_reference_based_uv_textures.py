#!/usr/bin/env python3
"""
Reference-Based UV Texture Generator for Medical Organ Models
Uses actual reference photos to create hyper-realistic textures
"""

import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
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


def load_reference_image(reference_path: str, target_size: int = 1024) -> Optional[Image.Image]:
    """
    Load and prepare reference image for texture generation.
    """
    try:
        if not os.path.exists(reference_path):
            print(f"Reference image not found: {reference_path}")
            return None
        
        # Load reference image
        ref_img = Image.open(reference_path)
        print(f"Loaded reference image: {ref_img.size}")
        
        # Convert to RGB if needed
        if ref_img.mode != 'RGB':
            ref_img = ref_img.convert('RGB')
        
        # Resize to target size
        ref_img = ref_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return ref_img
    except Exception as e:
        print(f"Error loading reference image: {e}")
        return None


def create_uv_mask_from_coordinates(uvs: np.ndarray, size: int = 1024) -> np.ndarray:
    """
    Create a mask from UV coordinates showing which areas of the texture should be filled.
    """
    uv_mask = np.zeros((size, size), dtype=np.float32)
    
    if len(uvs) == 0:
        # If no UVs, fill entire texture
        uv_mask.fill(1.0)
        return uv_mask
    
    # Convert UV coordinates to image coordinates
    uv_coords = uvs * size
    uv_coords = np.clip(uv_coords, 0, size - 1).astype(int)
    
    # Fill UV regions with larger radius for better coverage
    for u, v in uv_coords:
        radius = 4  # Increased radius for better coverage
        y_start = max(0, v - radius)
        y_end = min(size, v + radius + 1)
        x_start = max(0, u - radius)
        x_end = min(size, u + radius + 1)
        
        uv_mask[y_start:y_end, x_start:x_end] = 1.0
    
    # Dilate the mask to fill gaps and create smoother boundaries
    uv_mask = ndimage.binary_dilation(uv_mask, iterations=5).astype(np.float32)
    
    # Apply Gaussian blur for smoother edges
    uv_mask = ndimage.gaussian_filter(uv_mask, sigma=2.0)
    
    return uv_mask


def create_reference_based_texture(reference_img: Image.Image, uv_mask: np.ndarray, 
                                 organ_name: str, size: int = 1024) -> Image.Image:
    """
    Create texture based on reference image, adapted for the UV layout.
    """
    print(f"Creating reference-based texture for {organ_name}...")
    
    # Convert reference to numpy array
    ref_array = np.array(reference_img).astype(np.float32) / 255.0
    
    # Create base texture
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Sample from reference image with tiling to fill UV areas
    # Create a larger tiled version of the reference
    tile_size = size * 2
    tiled_ref = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
    
    # Tile the reference image
    for i in range(0, tile_size, size):
        for j in range(0, tile_size, size):
            tiled_ref[i:i+size, j:j+size] = ref_array
    
    # Add some random offset for variation
    offset_x = np.random.randint(0, size)
    offset_y = np.random.randint(0, size)
    
    # Extract the tiled region
    tiled_region = tiled_ref[offset_y:offset_y+size, offset_x:offset_x+size]
    
    # Apply UV mask to the tiled reference
    for i in range(3):
        texture[:, :, i] = tiled_region[:, :, i] * uv_mask + (0.2) * (1 - uv_mask)
    
    # Add organic variations based on organ type
    if "colon" in organ_name.lower():
        # Add haustra patterns (colon segments)
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        haustra_pattern = np.sin(y * np.pi * 6) * 0.2
        
        # Add subtle color variations for haustra
        for i in range(3):
            texture[:, :, i] += haustra_pattern * 0.1 * uv_mask
    
    # Add fine organic noise
    organic_noise = np.random.normal(0, 0.02, (size, size, 3))
    texture += organic_noise * uv_mask[:, :, np.newaxis]
    
    # Add some surface irregularities
    irregularities = np.random.normal(0, 0.03, (size, size))
    for i in range(3):
        texture[:, :, i] += irregularities * uv_mask
    
    # Add subtle lighting variations
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    lighting = 0.8 + 0.4 * np.sin(x * np.pi * 3) * np.sin(y * np.pi * 2)
    
    for i in range(3):
        texture[:, :, i] *= lighting * uv_mask + (1 - uv_mask)
    
    # Ensure values are in valid range
    texture = np.clip(texture, 0, 1)
    
    # Convert to PIL Image
    texture_uint8 = (texture * 255).astype(np.uint8)
    img = Image.fromarray(texture_uint8)
    
    # Apply photo-realistic enhancements
    # Increase contrast for more dramatic appearance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Increase saturation for richer colors
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.15)
    
    # Add slight sharpening for detail
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    # Add subtle noise for organic feel
    noise = np.random.normal(0, 5, (size, size, 3))
    img_array = np.array(img).astype(np.float32)
    img_array += noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img


def create_fallback_texture(organ_name: str, uv_mask: np.ndarray, size: int = 1024) -> Image.Image:
    """
    Create a fallback texture when no reference image is available.
    """
    print(f"Creating fallback texture for {organ_name}...")
    
    # Organ-specific colors
    organ_colors = {
        "colon": [220, 120, 140],
        "heart": [180, 60, 60],
        "aorta": [200, 80, 80],
        "artery": [220, 70, 70],
        "hip": [220, 200, 180],
        "bone": [225, 210, 190],
    }
    
    # Find best color match
    base_color = [180, 140, 120]  # Default
    for key, color in organ_colors.items():
        if key in organ_name.lower():
            base_color = color
            break
    
    # Create texture
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Apply base color
    for i in range(3):
        texture[:, :, i] *= base_color[i] / 255.0
    
    # Add organic variations
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Add patterns based on organ type
    if "colon" in organ_name.lower():
        pattern = np.sin(y * np.pi * 8) * 0.2
    elif "heart" in organ_name.lower():
        pattern = np.sin(x * np.pi * 15) * 0.15 + np.sin(y * np.pi * 12) * 0.1
    elif "artery" in organ_name.lower() or "aorta" in organ_name.lower():
        pattern = np.sin(x * np.pi * 10) * 0.1
    elif "hip" in organ_name.lower() or "bone" in organ_name.lower():
        pattern = np.sin(x * np.pi * 30) * 0.1 + np.sin(y * np.pi * 25) * 0.1
    else:
        pattern = np.sin(x * np.pi * 20) * 0.1
    
    for i in range(3):
        texture[:, :, i] += pattern * 0.1
    
    # Add noise
    noise = np.random.normal(0, 0.03, (size, size, 3))
    texture += noise
    
    # Apply UV mask
    for i in range(3):
        texture[:, :, i] = texture[:, :, i] * uv_mask + (0.2) * (1 - uv_mask)
    
    texture = np.clip(texture, 0, 1)
    texture_uint8 = (texture * 255).astype(np.uint8)
    img = Image.fromarray(texture_uint8)
    
    # Enhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    
    return img


def process_model_with_reference(model_dir: str, reference_path: str = None, 
                               size: int = 1024, overwrite: bool = False) -> bool:
    """
    Process a single model using reference image for texture generation.
    """
    model_name = os.path.basename(model_dir)
    
    print(f"\n[*] Processing '{model_name}' with reference-based texture...")
    
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
        
        # Create UV mask
        uv_mask = create_uv_mask_from_coordinates(uvs, size)
        
        # Load reference image if available
        reference_img = None
        if reference_path and os.path.exists(reference_path):
            reference_img = load_reference_image(reference_path, size)
        
        # Generate texture
        if reference_img:
            print("    - Generating reference-based texture...")
            texture_img = create_reference_based_texture(reference_img, uv_mask, model_name, size)
        else:
            print("    - Generating fallback texture...")
            texture_img = create_fallback_texture(model_name, uv_mask, size)
        
        # Save texture
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        texture_img.save(texture_path)
        print(f"    ✓ Saved reference-based texture: {texture_path}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process models with reference images."""
    parser = argparse.ArgumentParser(description="Generate reference-based UV textures")
    parser.add_argument("--organ", help="Specific organ to process")
    parser.add_argument("--reference", help="Path to reference image")
    parser.add_argument("--size", type=int, default=1024, help="Texture size (default: 1024)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing textures")
    
    args = parser.parse_args()
    
    models_dir = "output/models"
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found")
        return 1
    
    # Default reference paths
    default_references = {
        "colon": "output/COLONPHOTO.jpg",
        "heart": None,  # No default reference for heart
        "aorta": None,  # No default reference for aorta
    }
    
    if args.organ:
        # Process specific organ
        organ_dir = os.path.join(models_dir, args.organ)
        if not os.path.exists(organ_dir):
            print(f"Error: Organ directory '{organ_dir}' not found")
            return 1
        
        # Use provided reference or default
        reference_path = args.reference
        if not reference_path and args.organ in default_references:
            reference_path = default_references[args.organ]
        
        success = process_model_with_reference(organ_dir, reference_path, args.size, args.overwrite)
        return 0 if success else 1
    else:
        # Process all models with their default references
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
            reference_path = default_references.get(model_dir)
            
            if process_model_with_reference(model_path, reference_path, args.size, args.overwrite):
                successful += 1
            else:
                failed += 1
        
        print(f"\n[*] Summary: {successful} successful, {failed} failed")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
