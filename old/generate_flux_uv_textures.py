#!/usr/bin/env python3
"""
Flux UV Texture Generator for Medical Organ Models
Uses Flux AI to generate hyper-realistic textures that map exactly to UV coordinates
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


def create_uv_layout_visualization(uvs: np.ndarray, size: int = 1024) -> Image.Image:
    """
    Create a visualization of the UV layout to send to Flux as a guide.
    """
    # Create base image with UV layout
    img = Image.new('RGB', (size, size), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    if len(uvs) == 0:
        return img
    
    # Convert UV coordinates to image coordinates
    uv_coords = uvs * size
    uv_coords = np.clip(uv_coords, 0, size - 1).astype(int)
    
    # Draw UV points with larger radius for visibility
    for u, v in uv_coords:
        radius = 2
        draw.ellipse([u-radius, v-radius, u+radius, v+radius], fill=(200, 200, 200))
    
    # Create wireframe by connecting nearby points
    print(f"Creating UV wireframe with {len(uv_coords)} points...")
    for i, (u1, v1) in enumerate(uv_coords):
        if i % 1000 == 0:  # Progress indicator
            print(f"Processing point {i}/{len(uv_coords)}")
        
        # Find nearby points and connect them
        for j, (u2, v2) in enumerate(uv_coords[i+1:min(i+50, len(uv_coords))]):
            distance = np.sqrt((u1-u2)**2 + (v1-v2)**2)
            if distance < 30:  # Connect nearby points
                draw.line([u1, v1, u2, v2], fill=(150, 150, 150), width=1)
    
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


def generate_texture_with_flux(prompt: str, uv_guide_image: Image.Image, 
                             size: int = 1024, guidance_scale: float = 3.5, 
                             num_steps: int = 50, seed: int = None) -> Optional[Image.Image]:
    """
    Generate texture using Flux AI with UV layout guidance.
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
        print(f"Sending request to {flux_url}/generate_with_control")
        
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
        print(f"Error generating texture with Flux: {e}")
        return None


def apply_texture_to_uv_layout(flux_texture: Image.Image, uv_mask: np.ndarray, 
                              organ_name: str, size: int = 1024) -> Image.Image:
    """
    Apply the Flux-generated texture to the UV layout, ensuring it fits exactly.
    """
    print(f"Applying Flux texture to UV layout for {organ_name}...")
    
    # Convert Flux texture to numpy array
    flux_array = np.array(flux_texture.resize((size, size))).astype(np.float32) / 255.0
    
    # Create final texture
    final_texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Apply Flux texture only in UV regions
    for i in range(3):
        final_texture[:, :, i] = (
            flux_array[:, :, i] * uv_mask + 
            (0.2) * (1 - uv_mask)  # Dark background outside UV areas
        )
    
    # Add organ-specific enhancements
    if "colon" in organ_name.lower():
        # Enhance colon-specific features
        # Add subtle haustra patterns
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        haustra_pattern = np.sin(y * np.pi * 6) * 0.1
        for i in range(3):
            final_texture[:, :, i] += haustra_pattern * uv_mask * 0.1
    
    # Add subtle organic noise
    organic_noise = np.random.normal(0, 0.01, (size, size, 3))
    final_texture += organic_noise * uv_mask[:, :, np.newaxis]
    
    # Ensure values are in valid range
    final_texture = np.clip(final_texture, 0, 1)
    
    # Convert to PIL Image
    final_texture_uint8 = (final_texture * 255).astype(np.uint8)
    img = Image.fromarray(final_texture_uint8)
    
    # Apply final enhancements
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.05)
    
    return img


def generate_organ_specific_prompt(organ_name: str, reference_description: str = None) -> str:
    """
    Generate organ-specific prompts for Flux based on medical accuracy.
    """
    prompts = {
        "colon": "hyper photo-realistic human colon tissue surface, moist pink-red organic texture, natural haustra folds, medical photography quality, anatomically accurate, glossy wet surface, detailed biological texture, 8k resolution, professional medical imaging",
        
        "heart": "hyper photo-realistic human cardiac muscle tissue, deep red cardiac muscle fibers, coronary blood vessels visible, medical photography quality, anatomically accurate, detailed muscle striations, professional medical imaging, 8k resolution",
        
        "aorta": "hyper photo-realistic human aortic artery tissue, bright red arterial wall, smooth muscle layer, endothelial surface, medical photography quality, anatomically accurate, detailed vascular texture, professional medical imaging, 8k resolution",
        
        "left_iliac_artery": "hyper photo-realistic human iliac artery tissue, bright red arterial wall, smooth muscle layer, detailed vascular texture, medical photography quality, anatomically accurate, professional medical imaging, 8k resolution",
        
        "right_iliac_artery": "hyper photo-realistic human iliac artery tissue, bright red arterial wall, smooth muscle layer, detailed vascular texture, medical photography quality, anatomically accurate, professional medical imaging, 8k resolution",
        
        "left_hip": "hyper photo-realistic human hip bone tissue, ivory bone texture, trabecular bone structure, haversian canals visible, medical photography quality, anatomically accurate, detailed bone surface, professional medical imaging, 8k resolution",
        
        "right_hip": "hyper photo-realistic human hip bone tissue, ivory bone texture, trabecular bone structure, haversian canals visible, medical photography quality, anatomically accurate, detailed bone surface, professional medical imaging, 8k resolution",
    }
    
    # Find best match
    for key, prompt in prompts.items():
        if key in organ_name.lower():
            return prompt
    
    # Default prompt
    return f"hyper photo-realistic human {organ_name} tissue surface, medical photography quality, anatomically accurate, detailed biological texture, professional medical imaging, 8k resolution"


def process_model_with_flux(model_dir: str, size: int = 1024, overwrite: bool = False) -> bool:
    """
    Process a single model using Flux AI for hyper-realistic texture generation.
    """
    model_name = os.path.basename(model_dir)
    
    print(f"\n[*] Processing '{model_name}' with Flux AI texture generation...")
    
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
        
        # Create UV layout visualization for Flux
        print("    - Creating UV layout visualization...")
        uv_guide = create_uv_layout_visualization(uvs, size)
        
        # Generate organ-specific prompt
        prompt = generate_organ_specific_prompt(model_name)
        print(f"    - Using prompt: {prompt[:100]}...")
        
        # Generate texture with Flux
        print("    - Generating texture with Flux AI...")
        flux_texture = generate_texture_with_flux(
            prompt=prompt,
            uv_guide_image=uv_guide,
            size=size,
            guidance_scale=3.5,
            num_steps=50
        )
        
        if flux_texture is None:
            print("    ✗ Failed to generate texture with Flux")
            return False
        
        # Apply texture to UV layout
        print("    - Applying texture to UV layout...")
        final_texture = apply_texture_to_uv_layout(flux_texture, uv_mask, model_name, size)
        
        # Save texture
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        final_texture.save(texture_path)
        print(f"    ✓ Saved Flux-generated texture: {texture_path}")
        
        # Save UV guide for debugging
        uv_guide_path = os.path.join(model_dir, "textures", "uv_guide_flux.png")
        uv_guide.save(uv_guide_path)
        print(f"    ✓ Saved UV guide: {uv_guide_path}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process models with Flux AI."""
    parser = argparse.ArgumentParser(description="Generate Flux AI UV textures")
    parser.add_argument("--organ", help="Specific organ to process")
    parser.add_argument("--size", type=int, default=1024, help="Texture size (default: 1024)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing textures")
    
    args = parser.parse_args()
    
    models_dir = "output/models"
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found")
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
    
    if args.organ:
        # Process specific organ
        organ_dir = os.path.join(models_dir, args.organ)
        if not os.path.exists(organ_dir):
            print(f"Error: Organ directory '{organ_dir}' not found")
            return 1
        
        success = process_model_with_flux(organ_dir, args.size, args.overwrite)
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
            if process_model_with_flux(model_path, args.size, args.overwrite):
                successful += 1
            else:
                failed += 1
        
        print(f"\n[*] Summary: {successful} successful, {failed} failed")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
