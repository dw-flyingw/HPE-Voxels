#!/usr/bin/env python3
"""
generate_organ_textures.py

Generates realistic diffuse textures for all organ models based on their anatomical characteristics.
This script creates organ-specific textures with appropriate colors, patterns, and surface details.

Usage:
    python generate_organ_textures.py                    # Generate for all models
    python generate_organ_textures.py --organ colon     # Generate for specific organ
    python generate_organ_textures.py --size 1024       # Higher resolution
    python generate_organ_textures.py --overwrite       # Overwrite existing textures
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFilter
import numpy as np
import random


def _normalize_label_name(name: str) -> str:
    """Normalize label names for robust matching between folder names and JSON names."""
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = s.replace("-", "_").replace(" ", "_")
    # Collapse multiple underscores and trim
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def load_label_info(json_path: str) -> dict:
    """Loads a map from label ID to its name and color."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {item['id']: {'name': item['name'], 'color': item['color']} for item in data}
    except FileNotFoundError:
        print(f"[!] Error: Color map not found at '{json_path}'", file=sys.stderr)
        return None


def create_vascular_network(size: int, organ_type: str) -> np.ndarray:
    """Create subtle vascular network pattern for organic bone-like textures."""
    vascular_mask = np.zeros((size, size), dtype=np.uint8)
    
    # More subtle vascular patterns for organic look
    if organ_type in ['heart']:
        num_vessels = random.randint(6, 10)
        vessel_width_range = (2, 5)
        vessel_length_range = (12, 20)
    elif organ_type in ['liver', 'kidney']:
        num_vessels = random.randint(8, 14)
        vessel_width_range = (1.5, 4)
        vessel_length_range = (8, 16)
    elif organ_type in ['colon', 'stomach', 'small bowel']:
        num_vessels = random.randint(10, 18)
        vessel_width_range = (1, 3)
        vessel_length_range = (6, 12)
    elif organ_type in ['left_hip', 'right_hip', 'bone'] or 'hip' in organ_type or 'vertebrae' in organ_type:
        # Bone: very subtle vascular patterns
        num_vessels = random.randint(4, 8)
        vessel_width_range = (1, 2.5)
        vessel_length_range = (5, 10)
    else:  # Default for other organs
        num_vessels = random.randint(8, 12)
        vessel_width_range = (1.5, 3.5)
        vessel_length_range = (8, 15)
    
    for _ in range(num_vessels):
        start_x = random.randint(0, size)
        start_y = random.randint(0, size)
        
        points = []
        current_x, current_y = start_x, start_y
        
        for _ in range(random.randint(*vessel_length_range)):
            points.append((int(current_x), int(current_y)))
            
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(3, 10)
            
            current_x += distance * np.cos(angle)
            current_y += distance * np.sin(angle)
            
            current_x = max(0, min(size-1, current_x))
            current_y = max(0, min(size-1, current_y))
        
        if len(points) > 1:
            width = random.uniform(*vessel_width_range)
            
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                vessel_img = PILImage.new('L', (size, size), 0)
                draw = ImageDraw.Draw(vessel_img)
                
                segment_width = width * random.uniform(0.7, 1.1)
                draw.line([(x1, y1), (x2, y2)], fill=255, width=int(segment_width))
                
                vessel_array = np.array(vessel_img)
                vascular_mask = np.maximum(vascular_mask, vessel_array)
    
    # Apply stronger blur for subtle organic look
    vascular_img = PILImage.fromarray(vascular_mask)
    vascular_img = vascular_img.filter(ImageFilter.GaussianBlur(radius=2.5))
    
    return np.array(vascular_img) / 255.0


def create_organ_specific_pattern(size: int, organ_type: str) -> np.ndarray:
    """Create organ-specific surface patterns inspired by bone/organic textures."""
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Base organic texture with irregular patches and fibrous striations
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    if organ_type in ['left_hip', 'right_hip', 'bone'] or 'hip' in organ_type or 'vertebrae' in organ_type:
        # Bone texture with complex organic structure
        # Create fibrous striations
        fiber_pattern = (
            np.sin(x * np.pi * 25) * 0.4 +
            np.sin(y * np.pi * 18) * 0.3 +
            np.sin((x + y) * np.pi * 15) * 0.2 +
            np.sin((x - y) * np.pi * 12) * 0.15
        )
        
        # Add irregular patches
        num_patches = random.randint(15, 25)
        for _ in range(num_patches):
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            radius_x = random.uniform(0.05, 0.15)
            radius_y = random.uniform(0.05, 0.15)
            
            dx = (x - center_x) / radius_x
            dy = (y - center_y) / radius_y
            
            patch = np.exp(-(dx**2 + dy**2) * 2)
            patch *= random.uniform(0.3, 0.8)
            pattern += patch
        
        pattern = np.clip(pattern, 0, 1)
        pattern = pattern * 0.7 + fiber_pattern * 0.3
        
    elif organ_type == 'heart':
        # Heart muscle with organic bone-like texture
        muscle_pattern = (
            np.sin(x * np.pi * 20) * 0.3 +
            np.sin(y * np.pi * 15) * 0.25 +
            np.sin((x + y) * np.pi * 12) * 0.2
        )
        
        # Add organic patches
        num_patches = random.randint(8, 15)
        for _ in range(num_patches):
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            radius = random.uniform(0.08, 0.18)
            
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            patch = np.exp(-(dx**2 + dy**2) * 1.5)
            patch *= random.uniform(0.2, 0.6)
            pattern += patch
        
        pattern = np.clip(pattern, 0, 1)
        pattern = pattern * 0.6 + muscle_pattern * 0.4
        
    elif organ_type == 'liver':
        # Liver with organic bone-like texture
        lobule_pattern = (
            np.sin(x * np.pi * 18) * 0.25 +
            np.sin(y * np.pi * 12) * 0.2 +
            np.sin((x + y) * np.pi * 10) * 0.15
        )
        
        # Add organic patches
        num_patches = random.randint(12, 20)
        for _ in range(num_patches):
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            radius = random.uniform(0.06, 0.14)
            
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            patch = np.exp(-(dx**2 + dy**2) * 2)
            patch *= random.uniform(0.25, 0.65)
            pattern += patch
        
        pattern = np.clip(pattern, 0, 1)
        pattern = pattern * 0.65 + lobule_pattern * 0.35
        
    elif organ_type == 'kidney':
        # Kidney with organic bone-like texture
        nephron_pattern = (
            np.sin(x * np.pi * 22) * 0.3 +
            np.sin(y * np.pi * 16) * 0.25 +
            np.sin((x - y) * np.pi * 14) * 0.15
        )
        
        # Add organic patches
        num_patches = random.randint(10, 18)
        for _ in range(num_patches):
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            radius = random.uniform(0.07, 0.16)
            
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            patch = np.exp(-(dx**2 + dy**2) * 1.8)
            patch *= random.uniform(0.3, 0.7)
            pattern += patch
        
        pattern = np.clip(pattern, 0, 1)
        pattern = pattern * 0.6 + nephron_pattern * 0.4
        
    elif organ_type == 'colon':
        # Colon with organic bone-like texture
        # Create haustra pattern with organic patches
        num_segments = random.randint(10, 16)
        segment_width = size // num_segments
        
        for i in range(num_segments):
            x_start = i * segment_width
            x_end = min((i + 1) * segment_width, size)
            
            center_x = (x_start + x_end) / 2
            center_y = size / 2
            
            segment_depth = random.uniform(0.4, 0.8) * size * 0.25
            segment_height = random.uniform(0.5, 0.9) * size
            
            y_grid, x_grid = np.ogrid[:size, :size]
            dx = (x_grid - center_x) / (segment_width / 2)
            dy = (y_grid - center_y) / (segment_height / 2)
            
            ellipse_mask = 1.0 - (dx**2 + dy**2)
            ellipse_mask = np.clip(ellipse_mask, 0, 1)
            
            depth_mask = np.exp(-(dx**2 + dy**2) * 2) * segment_depth
            pattern += ellipse_mask * depth_mask
        
        pattern = np.clip(pattern, 0, 1)
        
    elif organ_type == 'lung':
        # Lung with organic bone-like texture
        alveolar_pattern = (
            np.sin(x * np.pi * 20) * 0.25 +
            np.sin(y * np.pi * 15) * 0.2 +
            np.sin((x + y) * np.pi * 12) * 0.15
        )
        
        # Add organic patches
        num_patches = random.randint(15, 25)
        for _ in range(num_patches):
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            radius = random.uniform(0.05, 0.12)
            
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            patch = np.exp(-(dx**2 + dy**2) * 2.5)
            patch *= random.uniform(0.2, 0.5)
            pattern += patch
        
        pattern = np.clip(pattern, 0, 1)
        pattern = pattern * 0.7 + alveolar_pattern * 0.3
        
    elif organ_type in ['aorta', 'left_iliac_artery', 'right_iliac_artery'] or 'artery' in organ_type:
        # Blood vessel with organic bone-like texture
        vessel_pattern = (
            np.sin(x * np.pi * 16) * 0.3 +
            np.sin(y * np.pi * 12) * 0.25 +
            np.sin((x + y) * np.pi * 8) * 0.15
        )
        
        # Add organic patches
        num_patches = random.randint(8, 15)
        for _ in range(num_patches):
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            radius = random.uniform(0.08, 0.18)
            
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            patch = np.exp(-(dx**2 + dy**2) * 1.5)
            patch *= random.uniform(0.25, 0.65)
            pattern += patch
        
        pattern = np.clip(pattern, 0, 1)
        pattern = pattern * 0.6 + vessel_pattern * 0.4
        
    else:
        # Generic organic bone-like pattern
        organic_pattern = (
            np.sin(x * np.pi * 20) * 0.3 +
            np.sin(y * np.pi * 15) * 0.25 +
            np.sin((x + y) * np.pi * 10) * 0.2
        )
        
        # Add organic patches
        num_patches = random.randint(12, 20)
        for _ in range(num_patches):
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            radius = random.uniform(0.06, 0.14)
            
            dx = (x - center_x) / radius
            dy = (y - center_y) / radius
            
            patch = np.exp(-(dx**2 + dy**2) * 2)
            patch *= random.uniform(0.25, 0.65)
            pattern += patch
        
        pattern = np.clip(pattern, 0, 1)
        pattern = pattern * 0.65 + organic_pattern * 0.35
    
    # Add fine details like pores and cracks
    # Create pore/crack pattern
    pore_pattern = np.random.random((size, size)) * 0.1
    pore_pattern = (pore_pattern > 0.95).astype(np.float32) * 0.3
    
    # Create fine crack patterns
    crack_pattern = np.zeros((size, size))
    num_cracks = random.randint(3, 8)
    for _ in range(num_cracks):
        start_x = random.randint(0, size-1)
        start_y = random.randint(0, size-1)
        length = random.randint(size//4, size//2)
        angle = random.uniform(0, 2 * np.pi)
        
        for i in range(length):
            x_pos = int(start_x + i * np.cos(angle))
            y_pos = int(start_y + i * np.sin(angle))
            
            if 0 <= x_pos < size and 0 <= y_pos < size:
                crack_pattern[y_pos, x_pos] = 0.4
    
    # Combine all patterns
    pattern = pattern + pore_pattern + crack_pattern
    pattern = np.clip(pattern, 0, 1)
    
    # Smooth the pattern
    pattern_img = PILImage.fromarray((pattern * 255).astype(np.uint8))
    pattern_img = pattern_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    return np.array(pattern_img) / 255.0


def create_fatty_tissue_pattern(size: int) -> np.ndarray:
    """Create subtle areas of fatty/connective tissue with organic characteristics."""
    fat_mask = np.zeros((size, size))
    
    num_fat_areas = random.randint(8, 15)  # More areas for organic look
    
    for _ in range(num_fat_areas):
        center_x = random.randint(size//6, 5*size//6)
        center_y = random.randint(size//6, 5*size//6)
        radius = random.randint(size//12, size//6)  # Smaller, more varied sizes
        
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create more irregular, organic shapes
        noise_factor = np.random.normal(1, 0.2, (size, size))
        distance *= noise_factor
        
        fat_area = np.exp(-(distance / radius)**2)
        fat_area *= random.uniform(0.2, 0.6)  # More subtle intensity
        fat_mask += fat_area
    
    # Add some irregular, elongated patches
    num_elongated = random.randint(3, 6)
    for _ in range(num_elongated):
        center_x = random.randint(size//4, 3*size//4)
        center_y = random.randint(size//4, 3*size//4)
        
        # Create elongated shape
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(size//8, size//4)
        width = random.randint(size//20, size//10)
        
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        
        # Rotate coordinates
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = (x - center_x) * cos_a + (y - center_y) * sin_a
        y_rot = -(x - center_x) * sin_a + (y - center_y) * cos_a
        
        # Create elliptical mask
        ellipse_mask = (x_rot**2 / (length/2)**2 + y_rot**2 / (width/2)**2) <= 1
        fat_area = ellipse_mask.astype(np.float32) * random.uniform(0.15, 0.4)
        fat_mask += fat_area
    
    fat_mask = np.clip(fat_mask, 0, 1)
    fat_img = PILImage.fromarray((fat_mask * 255).astype(np.uint8))
    fat_img = fat_img.filter(ImageFilter.GaussianBlur(radius=6))  # Less blur for more detail
    
    return np.array(fat_img) / 255.0


def create_realistic_organ_diffuse(organ_name: str, base_color: tuple, size: int = 512) -> PILImage.Image:
    """Create a realistic diffuse texture for any organ with bone/organic characteristics."""
    print(f"    Creating {organ_name} diffuse texture {size}x{size}px with base color RGB{base_color}")
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Convert base color to numpy array and create earthy bone-like base
    base_color = np.array(base_color, dtype=np.float32)
    
    # Transform bright colors to more earthy, organic tones
    if organ_name in ['left_hip', 'right_hip', 'bone'] or 'hip' in organ_name or 'vertebrae' in organ_name:
        # Bone colors: muted beige, light brown, off-white
        earthy_base = np.array([200, 185, 165])  # Warm beige
    elif organ_name == 'heart':
        # Heart: muted red-brown
        earthy_base = np.array([180, 120, 100])
    elif organ_name == 'liver':
        # Liver: muted brown-green
        earthy_base = np.array([160, 140, 120])
    elif organ_name == 'kidney':
        # Kidney: muted blue-brown
        earthy_base = np.array([170, 150, 140])
    elif organ_name == 'colon':
        # Colon: muted orange-brown
        earthy_base = np.array([190, 160, 130])
    elif organ_name == 'lung':
        # Lung: muted green-brown
        earthy_base = np.array([165, 155, 135])
    elif organ_name in ['aorta', 'left_iliac_artery', 'right_iliac_artery'] or 'artery' in organ_name:
        # Arteries: muted red-brown
        earthy_base = np.array([175, 125, 105])
    else:
        # Generic organic: warm beige
        earthy_base = np.array([185, 170, 155])
    
    # Blend original color with earthy base
    final_base = (base_color * 0.3 + earthy_base * 0.7).astype(np.float32)
    
    # Create base image
    img_array = np.full((size, size, 3), final_base, dtype=np.float32)
    
    # Generate texture components
    print(f"    - Generating {organ_name} specific pattern...")
    organ_pattern = create_organ_specific_pattern(size, organ_name)
    
    print("    - Generating vascular network...")
    vascular_mask = create_vascular_network(size, organ_name)
    
    print("    - Generating fatty tissue areas...")
    fat_mask = create_fatty_tissue_pattern(size)
    
    # Create color variations with earthy tones
    print("    - Applying organic color variations...")
    
    # Base color variations (more subtle for organic look)
    color_variation = np.random.normal(0, 8, img_array.shape)
    img_array += color_variation
    
    # Apply organ-specific pattern with organic color shifts
    pattern_variation = (organ_pattern - 0.5) * 25
    
    # Create organic color variations based on pattern
    for c in range(3):
        # Add warm brown variations in recessed areas
        recessed_mask = (organ_pattern < 0.4).astype(np.float32)
        brown_variation = recessed_mask * np.random.normal(-12, 5, (size, size))
        
        # Add lighter variations in raised areas
        raised_mask = (organ_pattern > 0.6).astype(np.float32)
        light_variation = raised_mask * np.random.normal(8, 3, (size, size))
        
        img_array[:, :, c] += pattern_variation + brown_variation + light_variation
    
    # Apply vascular network with muted colors
    if organ_name == 'heart':
        vein_color = np.array([140, 80, 60])  # Muted dark red
        artery_color = np.array([160, 90, 70])  # Muted red
    elif organ_name in ['liver', 'kidney']:
        vein_color = np.array([120, 90, 70])  # Muted brown
        artery_color = np.array([140, 100, 80])  # Muted brown-red
    elif organ_name in ['colon', 'stomach', 'small bowel']:
        capillary_color = np.array([130, 85, 65])  # Muted brown
        vessel_color = np.array([150, 95, 75])  # Muted brown-red
    else:
        vein_color = np.array([125, 85, 65])  # Muted brown
        artery_color = np.array([145, 95, 75])  # Muted brown-red
    
    # Apply vascular colors with subtle blending
    for c in range(3):
        if organ_name in ['colon', 'stomach', 'small bowel']:
            capillary_component = vascular_mask * (capillary_color[c] - final_base[c])
            vessel_component = vascular_mask * (vessel_color[c] - final_base[c])
            vascular_blend = capillary_component * 0.8 + vessel_component * 0.2
            img_array[:, :, c] += vascular_blend * 0.3
        else:
            vein_component = vascular_mask * (vein_color[c] - final_base[c])
            artery_component = vascular_mask * (artery_color[c] - final_base[c])
            vascular_blend = vein_component * 0.7 + artery_component * 0.3
            img_array[:, :, c] += vascular_blend * 0.4
    
    # Apply fatty tissue (lighter organic areas)
    fat_color_adjustment = np.array([15, 12, 8])  # Subtle lightening
    for c in range(3):
        img_array[:, :, c] += fat_mask * fat_color_adjustment[c]
    
    # Add organic noise with earthy characteristics
    organic_noise = np.random.normal(0, 6, img_array.shape)
    img_array += organic_noise
    
    # Add blue-grey patches (like in reference texture)
    blue_grey_mask = np.random.random((size, size)) < 0.08
    blue_grey_color = np.array([160, 155, 150])  # Subtle blue-grey
    for c in range(3):
        img_array[:, :, c] = np.where(blue_grey_mask, 
                                     img_array[:, :, c] * 0.7 + blue_grey_color[c] * 0.3,
                                     img_array[:, :, c])
    
    # Add gradient variations for natural lighting
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Subtle radial gradient from center
    center_gradient = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    center_gradient = 1.0 - np.clip(center_gradient * 1.2, 0, 1)
    
    for c in range(3):
        img_array[:, :, c] += center_gradient * 6
    
    # Add directional lighting variation
    directional_gradient = (x * 0.2 + y * 0.5) * 5
    for c in range(3):
        img_array[:, :, c] += directional_gradient
    
    # Add subtle veining/crack patterns
    vein_pattern = np.zeros((size, size))
    num_veins = random.randint(5, 12)
    for _ in range(num_veins):
        start_x = random.randint(0, size-1)
        start_y = random.randint(0, size-1)
        length = random.randint(size//8, size//4)
        angle = random.uniform(0, 2 * np.pi)
        
        for i in range(length):
            x_pos = int(start_x + i * np.cos(angle) + np.random.normal(0, 2))
            y_pos = int(start_y + i * np.sin(angle) + np.random.normal(0, 2))
            
            if 0 <= x_pos < size and 0 <= y_pos < size:
                vein_pattern[y_pos, x_pos] = 0.3
    
    # Apply vein pattern with darker color
    vein_color_adjustment = np.array([-8, -6, -4])
    for c in range(3):
        img_array[:, :, c] += vein_pattern * vein_color_adjustment[c]
    
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 255)
    
    # Convert to PIL Image
    result_img = PILImage.fromarray(img_array.astype(np.uint8))
    
    # Apply final smoothing for organic feel
    result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    print(f"    ✓ {organ_name} diffuse texture created successfully")
    return result_img


def get_color_for_model(model_name: str, label_info: dict) -> list:
    """Get the appropriate color for a model based on its name."""
    color = [128, 128, 128]  # Default gray
    
    if label_info and model_name:
        normalized_model = _normalize_label_name(model_name)
        color_found = False
        
        # Try exact match first
        for label_id, info in label_info.items():
            label_name = _normalize_label_name(info['name'])
            if normalized_model == label_name:
                color = info['color']
                color_found = True
                print(f"    ✓ Exact match: '{model_name}' -> '{info['name']}' RGB{color}")
                break
        
        # If no exact match, try partial matching
        if not color_found:
            for label_id, info in label_info.items():
                label_name = _normalize_label_name(info['name'])
                if normalized_model in label_name or label_name in normalized_model:
                    color = info['color']
                    color_found = True
                    print(f"    ✓ Partial match: '{model_name}' -> '{info['name']}' RGB{color}")
                    break
        
        if not color_found:
            print(f"    ⚠ No color match for '{model_name}', using default gray RGB{color}")
    
    return color


def process_model_directory(model_dir: Path, label_info: dict, texture_size: int = 512, overwrite: bool = False) -> bool:
    """Process a single model directory and generate its diffuse texture."""
    model_name = model_dir.name
    textures_dir = model_dir / "textures"
    diffuse_path = textures_dir / "diffuse.png"
    
    # Create textures directory if it doesn't exist
    textures_dir.mkdir(exist_ok=True)
    
    # Check if diffuse.png already exists
    if diffuse_path.exists() and not overwrite:
        print(f"[⊘] Skipping '{model_name}' - diffuse.png already exists")
        return True
    
    try:
        print(f"[*] Processing '{model_name}'...")
        
        # Get color for this model
        color = get_color_for_model(model_name, label_info)
        
        # Generate diffuse texture
        diffuse_img = create_realistic_organ_diffuse(
            organ_name=model_name,
            base_color=tuple(color),
            size=texture_size
        )
        
        # Save the texture
        diffuse_img.save(diffuse_path, format='PNG', optimize=True)
        
        print(f"    ✓ Created: {diffuse_path.relative_to(model_dir.parent.parent)}")
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def get_matching_models(models_dir: Path, label_info: dict) -> list:
    """
    Get model directories that have matching organs in vista3d_label_colors.json.
    
    Args:
        models_dir: Path to models directory
        label_info: Dictionary of label information from vista3d_label_colors.json
        
    Returns:
        List of matching model directories
    """
    # Find all model directories
    all_model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and (d / "scene.glb").exists()]
    
    matching_models = []
    
    for model_dir in all_model_dirs:
        model_name = model_dir.name
        normalized_model = _normalize_label_name(model_name)
        
        # Check if this model matches any organ in the JSON
        for label_id, info in label_info.items():
            label_name = _normalize_label_name(info['name'])
            if normalized_model == label_name:
                matching_models.append(model_dir)
                print(f"    ✓ Found matching model: '{model_name}' -> '{info['name']}'")
                break
    
    return matching_models


def main():
    """Main function to generate realistic diffuse textures for all models."""
    
    parser = argparse.ArgumentParser(
        description='Generate realistic diffuse textures for organ models that match vista3d_label_colors.json'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./output/models',
        help='Path to the models directory (default: ./output/models)'
    )
    parser.add_argument(
        '--colors-json',
        type=str,
        default='./vista3d_label_colors.json',
        help='Path to the label colors JSON file (default: ./vista3d_label_colors.json)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Texture size in pixels (default: 512)'
    )
    parser.add_argument(
        '--organ',
        type=str,
        help='Generate texture for specific organ only'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing diffuse.png files'
    )
    parser.add_argument(
        '--show-matches',
        action='store_true',
        help='Show which models match organs in the JSON file'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    models_dir = Path(args.models_dir)
    colors_json = Path(args.colors_json)
    
    # Validate paths
    if not models_dir.exists():
        print(f"[!] Error: Models directory not found: {models_dir}", file=sys.stderr)
        return 1
    
    if not colors_json.exists():
        print(f"[!] Error: Colors JSON not found: {colors_json}", file=sys.stderr)
        return 1
    
    # Load label information
    print(f"[*] Loading color information from {colors_json}...")
    label_info = load_label_info(str(colors_json))
    if label_info is None:
        return 1
    
    print(f"    ✓ Loaded {len(label_info)} label colors")
    
    # Find matching model directories
    print(f"[*] Finding models that match organs in {colors_json.name}...")
    model_dirs = get_matching_models(models_dir, label_info)
    
    if not model_dirs:
        print(f"[!] No matching models found in {models_dir}", file=sys.stderr)
        print(f"    Models must have matching names in {colors_json.name}")
        return 1
    
    # Show matches if requested
    if args.show_matches:
        print(f"\nFound {len(model_dirs)} matching models:")
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            for label_id, info in label_info.items():
                label_name = _normalize_label_name(info['name'])
                normalized_model = _normalize_label_name(model_name)
                if normalized_model == label_name:
                    print(f"  ✓ {model_name} -> {info['name']} RGB{info['color']}")
                    break
        return 0
    
    # Filter by specific organ if requested
    if args.organ:
        model_dirs = [d for d in model_dirs if d.name.lower() == args.organ.lower()]
        if not model_dirs:
            print(f"[!] No matching model found for organ '{args.organ}'", file=sys.stderr)
            return 1
    
    print(f"\n{'='*60}")
    print(f"🎨 Realistic Organ Diffuse Texture Generator")
    print(f"{'='*60}")
    print(f"Models directory: {models_dir}")
    print(f"Found {len(model_dirs)} model(s)")
    print(f"Texture size: {args.size}x{args.size}px")
    print(f"Overwrite existing: {args.overwrite}")
    if args.organ:
        print(f"Organ filter: {args.organ}")
    print(f"{'='*60}\n")
    
    # Process each model directory
    successful = 0
    failed = 0
    skipped = 0
    
    for model_dir in sorted(model_dirs):
        result = process_model_directory(model_dir, label_info, args.size, args.overwrite)
        if result:
            diffuse_path = model_dir / "textures" / "diffuse.png"
            if diffuse_path.exists():
                successful += 1
            else:
                skipped += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"[✓] Generation complete!")
    print(f"    - Successfully generated: {successful}")
    if skipped > 0:
        print(f"    - Skipped (already exist): {skipped}")
    if failed > 0:
        print(f"    - Failed: {failed}")
    print(f"    - Output directory: {models_dir}")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
