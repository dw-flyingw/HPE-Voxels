#!/usr/bin/env python3
"""
Specialized Colon Flux Texture Generator
Uses COLONPHOTO.jpg as reference to create hyper-realistic colon textures
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
import requests
import base64
import io
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


def create_colon_reference_prompt() -> str:
    """
    Create a specialized prompt for colon texture based on COLONPHOTO.jpg analysis.
    """
    return """hyper photo-realistic human colon tissue surface, moist pink-red organic texture with deep reddish tones, natural haustra folds creating rounded segmented appearance, glossy wet mucosal surface, detailed biological texture with fine surface irregularities, medical photography quality, anatomically accurate colon anatomy, professional medical imaging, 8k resolution, macro photography style, soft directional lighting, shallow depth of field, wet tissue appearance, natural color variations from deep red in crevices to lighter pink on raised surfaces"""


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
        radius = 3
        y_start = max(0, v - radius)
        y_end = min(size, v + radius + 1)
        x_start = max(0, u - radius)
        x_end = min(size, u + radius + 1)
        
        uv_mask[y_start:y_end, x_start:x_end] = 1.0
    
    # Dilate the mask to fill gaps and create smoother boundaries
    uv_mask = ndimage.binary_dilation(uv_mask, iterations=4).astype(np.float32)
    
    # Apply Gaussian blur for smoother edges
    uv_mask = ndimage.gaussian_filter(uv_mask, sigma=1.5)
    
    return uv_mask


def create_colon_uv_layout_visualization(uvs: np.ndarray, size: int = 1024) -> Image.Image:
    """
    Create a specialized UV layout visualization for colon texture generation.
    """
    # Create base image with colon-appropriate background
    img = Image.new('RGB', (size, size), (30, 20, 25))  # Dark reddish background
    draw = ImageDraw.Draw(img)
    
    if len(uvs) == 0:
        return img
    
    # Convert UV coordinates to image coordinates
    uv_coords = uvs * size
    uv_coords = np.clip(uv_coords, 0, size - 1).astype(int)
    
    # Draw UV points with colon-appropriate colors
    for u, v in uv_coords:
        radius = 2
        # Use colon-like colors (pink-red tones)
        color = (180 + np.random.randint(-20, 20), 80 + np.random.randint(-20, 20), 60 + np.random.randint(-20, 20))
        draw.ellipse([u-radius, v-radius, u+radius, v+radius], fill=color)
    
    # Create haustra-like pattern connections
    print(f"Creating colon UV wireframe with {len(uv_coords)} points...")
    for i, (u1, v1) in enumerate(uv_coords):
        if i % 1000 == 0:  # Progress indicator
            print(f"Processing point {i}/{len(uv_coords)}")
        
        # Find nearby points and connect them with haustra-like patterns
        for j, (u2, v2) in enumerate(uv_coords[i+1:min(i+30, len(uv_coords))]):
            distance = np.sqrt((u1-u2)**2 + (v1-v2)**2)
            if distance < 25:  # Connect nearby points
                # Create segmented appearance like haustra
                segments = 3
                for seg in range(segments):
                    t1 = seg / segments
                    t2 = (seg + 1) / segments
                    
                    x1 = u1 + (u2 - u1) * t1
                    y1 = v1 + (v2 - v1) * t1
                    x2 = u1 + (u2 - u1) * t2
                    y2 = v1 + (v2 - v1) * t2
                    
                    # Add slight curve to simulate haustra
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    offset = np.random.uniform(-2, 2)
                    
                    draw.line([x1, y1, mid_x + offset, mid_y + offset, x2, y2], 
                             fill=(120, 60, 50), width=1)
    
    return img


def get_flux_server_url() -> str:
    """Get Flux server URL from environment or use default."""
    # Check for remote server configuration
    flux_url = os.getenv("FLUX_SERVER_URL")
    if flux_url:
        return flux_url.rstrip('/')
    
    # Fallback to host:port configuration
    port = os.getenv("FLUX_SERVER_PORT", "8000")
    host = os.getenv("FLUX_HOST", "localhost")
    return f"http://{host}:{port}"


def generate_colon_texture_with_flux(prompt: str, uv_guide_image: Image.Image, 
                                   size: int = 1024, guidance_scale: float = 3.5, 
                                   num_steps: int = 50, seed: int = None) -> Optional[Image.Image]:
    """
    Generate colon-specific texture using Flux AI with UV layout guidance.
    """
    try:
        # Convert UV guide to base64
        buffer = io.BytesIO()
        uv_guide_image.save(buffer, format='PNG')
        uv_guide_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Prepare request payload
        payload = {
            "prompt": prompt,
            "width": size,
            "height": size,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "control_image": uv_guide_base64,
            "control_type": "uv_layout"
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        # Make request to Flux server
        flux_url = get_flux_server_url()
        print(f"Sending colon texture request to {flux_url}/generate_with_control")
        
        response = requests.post(
            f"{flux_url}/generate_with_control",
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            # Load generated image
            img_data = response.content
            img = Image.open(io.BytesIO(img_data))
            return img
        else:
            print(f"Error: Server returned {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error generating colon texture with Flux: {e}")
        return None


def apply_colon_texture_to_uv_layout(flux_texture: Image.Image, uv_mask: np.ndarray, 
                                    size: int = 1024) -> Image.Image:
    """
    Apply the Flux-generated colon texture to the UV layout with colon-specific enhancements.
    """
    print("Applying colon texture to UV layout...")
    
    # Convert Flux texture to numpy array
    flux_array = np.array(flux_texture.resize((size, size))).astype(np.float32) / 255.0
    
    # Create final texture
    final_texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Apply Flux texture only in UV regions
    for i in range(3):
        final_texture[:, :, i] = (
            flux_array[:, :, i] * uv_mask + 
            (0.15) * (1 - uv_mask)  # Dark background outside UV areas
        )
    
    # Add colon-specific enhancements
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Add haustra patterns (rounded segmented appearance)
    haustra_pattern = np.sin(y * np.pi * 4) * 0.15 + np.sin(x * np.pi * 2) * 0.08
    for i in range(3):
        final_texture[:, :, i] += haustra_pattern * uv_mask * 0.12
    
    # Add mucosal surface variations
    mucosal_noise = np.random.normal(0, 0.02, (size, size, 3))
    final_texture += mucosal_noise * uv_mask[:, :, np.newaxis]
    
    # Enhance color variations (deeper reds in crevices, lighter pinks on raised areas)
    height_map = np.sin(x * np.pi * 6) * np.cos(y * np.pi * 4) * 0.1
    for i in range(3):
        if i == 0:  # Red channel
            final_texture[:, :, i] += height_map * uv_mask * 0.15
        elif i == 1:  # Green channel
            final_texture[:, :, i] -= height_map * uv_mask * 0.05
        elif i == 2:  # Blue channel
            final_texture[:, :, i] -= height_map * uv_mask * 0.08
    
    # Ensure values are in valid range
    final_texture = np.clip(final_texture, 0, 1)
    
    # Convert to PIL Image
    final_texture_uint8 = (final_texture * 255).astype(np.uint8)
    img = Image.fromarray(final_texture_uint8)
    
    # Apply final enhancements for colon appearance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)  # Slightly higher contrast
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)  # Slightly more saturated
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.05)  # Slightly brighter
    
    return img


def process_colon_with_flux(model_dir: str, size: int = 1024, overwrite: bool = False) -> bool:
    """
    Process the colon model using specialized Flux AI colon texture generation.
    """
    model_name = os.path.basename(model_dir)
    
    if "colon" not in model_name.lower():
        print(f"Warning: This script is specialized for colon models. Processing {model_name} anyway...")
    
    print(f"\n[*] Processing '{model_name}' with specialized colon Flux AI texture generation...")
    
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
        
        # Create specialized colon UV layout visualization
        print("    - Creating colon UV layout visualization...")
        uv_guide = create_colon_uv_layout_visualization(uvs, size)
        
        # Generate colon-specific prompt
        prompt = create_colon_reference_prompt()
        print(f"    - Using specialized colon prompt: {prompt[:100]}...")
        
        # Generate texture with Flux
        print("    - Generating colon texture with Flux AI...")
        flux_texture = generate_colon_texture_with_flux(
            prompt=prompt,
            uv_guide_image=uv_guide,
            size=size,
            guidance_scale=3.5,
            num_steps=50
        )
        
        if flux_texture is None:
            print("    ✗ Failed to generate colon texture with Flux")
            return False
        
        # Apply texture to UV layout
        print("    - Applying colon texture to UV layout...")
        final_texture = apply_colon_texture_to_uv_layout(flux_texture, uv_mask, size)
        
        # Save texture
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        final_texture.save(texture_path)
        print(f"    ✓ Saved colon Flux-generated texture: {texture_path}")
        
        # Save UV guide for debugging
        uv_guide_path = os.path.join(model_dir, "textures", "colon_uv_guide_flux.png")
        uv_guide.save(uv_guide_path)
        print(f"    ✓ Saved colon UV guide: {uv_guide_path}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process colon model with specialized Flux AI."""
    parser = argparse.ArgumentParser(description="Generate specialized colon Flux AI UV textures")
    parser.add_argument("--size", type=int, default=1024, help="Texture size (default: 1024)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing textures")
    
    args = parser.parse_args()
    
    models_dir = "output/models"
    colon_dir = os.path.join(models_dir, "colon")
    
    if not os.path.exists(colon_dir):
        print(f"Error: Colon directory '{colon_dir}' not found")
        return 1
    
    # Check Flux server availability
    flux_url = get_flux_server_url()
    try:
        response = requests.get(f"{flux_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"Error: Flux server at {flux_url} is not responding")
            return 1
    except requests.exceptions.RequestException:
        print(f"Error: Cannot connect to Flux server at {flux_url}")
        print("Please start the Flux server first:")
        print("cd backend && python flux_server.py")
        return 1
    
    print(f"✓ Flux server is running at {flux_url}")
    print("✓ Specialized for colon texture generation based on COLONPHOTO.jpg reference")
    
    success = process_colon_with_flux(colon_dir, args.size, args.overwrite)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
