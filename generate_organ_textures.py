#!/usr/bin/env python3
"""
Enhanced realistic organ texture generator with voxel shape awareness.
Creates anatomically accurate diffuse textures for 3D medical models.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import trimesh
from scipy.ndimage import gaussian_filter, zoom
import nibabel as nib

# Set random seed for reproducible results
random.seed(42)
np.random.seed(42)

def analyze_voxel_shape(model_path: str) -> dict:
    """Analyze the 3D shape of the voxel model to inform texture generation."""
    try:
        # Load the 3D model
        mesh = trimesh.load(model_path)
        
        # Calculate basic shape properties
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        volume = mesh.volume
        surface_area = mesh.surface_area
        
        # Calculate aspect ratios
        max_dim = max(size)
        aspect_ratios = size / max_dim
        
        # Determine shape type
        if all(ratio > 0.8 for ratio in aspect_ratios):
            shape_type = "spherical"
        elif max(aspect_ratios) > 2.0:
            shape_type = "elongated"
        else:
            shape_type = "irregular"
        
        # Calculate curvature level (simplified)
        curvature_level = min(1.0, surface_area / (6 * (volume ** (2/3))))
        
        return {
            "bounds": bounds.tolist(),
            "size": size.tolist(),
            "aspect_ratios": aspect_ratios.tolist(),
            "shape_type": shape_type,
            "curvature_level": curvature_level,
            "volume": float(volume),
            "surface_area": float(surface_area)
        }
    except Exception as e:
        print(f"    Warning: Could not analyze shape: {e}")
        return {
            "bounds": [[0, 0, 0], [1, 1, 1]],
            "size": [1, 1, 1],
            "aspect_ratios": [1, 1, 1],
            "shape_type": "irregular",
            "curvature_level": 0.5,
            "volume": 1.0,
            "surface_area": 6.0
        }

def get_anatomical_tissue_properties(organ_name: str) -> dict:
    """Get realistic anatomical tissue properties for the organ."""
    tissue_properties = {
        "heart": {
            "tissue_type": "cardiac_muscle",
            "color_base": [180, 60, 60],
            "color_variation": 25,
            "surface_roughness": 0.6,
            "vascular_density": 0.8,
            "fiber_direction": "circular_longitudinal",
            "fat_content": 0.1,
            "texture_scale": 1.2
        },
        "liver": {
            "tissue_type": "glandular",
            "color_base": [150, 100, 80],
            "color_variation": 30,
            "surface_roughness": 0.7,
            "vascular_density": 0.9,
            "fiber_direction": "mixed",
            "fat_content": 0.05,
            "texture_scale": 1.0
        },
        "kidney": {
            "tissue_type": "glandular",
            "color_base": [160, 90, 70],
            "color_variation": 20,
            "surface_roughness": 0.5,
            "vascular_density": 0.8,
            "fiber_direction": "mixed",
            "fat_content": 0.08,
            "texture_scale": 0.9
        },
        "colon": {
            "tissue_type": "hollow_muscle",
            "color_base": [200, 160, 140],  # More pinkish-peach like the reference
            "color_variation": 25,
            "surface_roughness": 0.9,
            "vascular_density": 0.7,
            "fiber_direction": "circular",
            "fat_content": 0.08,
            "texture_scale": 1.3  # Larger scale for more visible haustra
        },
        "stomach": {
            "tissue_type": "hollow_muscle",
            "color_base": [175, 130, 110],
            "color_variation": 30,
            "surface_roughness": 0.7,
            "vascular_density": 0.7,
            "fiber_direction": "circular",
            "fat_content": 0.12,
            "texture_scale": 1.0
        },
        "aorta": {
            "tissue_type": "vascular_smooth",
            "color_base": [200, 80, 80],
            "color_variation": 20,
            "surface_roughness": 0.4,
            "vascular_density": 0.3,
            "fiber_direction": "longitudinal",
            "fat_content": 0.05,
            "texture_scale": 0.8
        },
        "iliac_artery": {
            "tissue_type": "vascular_smooth",
            "color_base": [220, 70, 70],
            "color_variation": 25,
            "surface_roughness": 0.4,
            "vascular_density": 0.3,
            "fiber_direction": "longitudinal",
            "fat_content": 0.05,
            "texture_scale": 0.8
        },
        "hip": {
            "tissue_type": "bone",
            "color_base": [220, 200, 180],
            "color_variation": 15,
            "surface_roughness": 0.9,
            "vascular_density": 0.2,
            "fiber_direction": "none",
            "fat_content": 0.02,
            "texture_scale": 1.5
        },
        "bone": {
            "tissue_type": "bone",
            "color_base": [225, 210, 190],
            "color_variation": 10,
            "surface_roughness": 0.9,
            "vascular_density": 0.1,
            "fiber_direction": "none",
            "fat_content": 0.01,
            "texture_scale": 1.5
        }
    }
    
    # Find the best match for the organ
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
        "tissue_type": "soft_tissue",
        "color_base": [180, 140, 120],
        "color_variation": 30,
        "surface_roughness": 0.6,
        "vascular_density": 0.5,
        "fiber_direction": "mixed",
        "fat_content": 0.1,
        "texture_scale": 1.0
    }

def create_anatomical_vessel(x, y, size, width_range, length_range, curvature, branching_prob, fiber_direction):
    """Create an individual anatomical vessel with proper branching."""
    vessel_mask = np.zeros((size, size), dtype=np.float32)
    
    # Ensure integer values for random range
    min_length = max(1, int(length_range[0]))
    max_length = max(1, int(length_range[1]))
    vessel_length = random.randint(min_length, max_length)
    
    if vessel_length <= 0:
        return vessel_mask
    
    # Start position
    start_x = random.randint(0, size-1)
    start_y = random.randint(0, size-1)
    
    # Direction based on fiber direction
    if fiber_direction == "longitudinal":
        direction = [1, 0] if random.random() > 0.5 else [-1, 0]
    elif fiber_direction == "circular":
        direction = [0, 1] if random.random() > 0.5 else [0, -1]
    else:
        direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    # Normalize direction
    norm = np.sqrt(direction[0]**2 + direction[1]**2)
    if norm > 0:
        direction = [direction[0]/norm, direction[1]/norm]
    
    # Generate vessel path
    points = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    
    for _ in range(vessel_length):
        # Add curvature
        if curvature > 0:
            curve_x = random.uniform(-curvature, curvature)
            curve_y = random.uniform(-curvature, curvature)
            direction[0] += curve_x
            direction[1] += curve_y
            
            # Renormalize
            norm = np.sqrt(direction[0]**2 + direction[1]**2)
            if norm > 0:
                direction = [direction[0]/norm, direction[1]/norm]
        
        # Move to next point
        step_size = random.uniform(2, 5)
        current_x += direction[0] * step_size
        current_y += direction[1] * step_size
        
        # Wrap around edges
        current_x = current_x % size
        current_y = current_y % size
        
        points.append((int(current_x), int(current_y)))
        
        # Check for branching
        if random.random() < branching_prob and len(points) > 3:
            branch = create_anatomical_vessel(
                x, y, size, 
                (width_range[0] * 0.7, width_range[1] * 0.7),
                (length_range[0] * 0.5, length_range[1] * 0.5),
                curvature * 0.8, branching_prob * 0.5, fiber_direction
            )
            vessel_mask = np.maximum(vessel_mask, branch * 0.7)
    
    # Draw the vessel
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # Interpolate between points
        steps = max(1, int(np.sqrt((x2-x1)**2 + (y2-y1)**2)))
        for j in range(steps + 1):
            t = j / steps
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            
            if 0 <= px < size and 0 <= py < size:
                width = random.uniform(*width_range)
                for dx in range(-int(width), int(width) + 1):
                    for dy in range(-int(width), int(width) + 1):
                        if dx*dx + dy*dy <= width*width:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                vessel_mask[ny, nx] = 1.0
    
    return vessel_mask

def create_anatomical_vascular_network(size: int, organ_name: str, tissue_props: dict, shape_info: dict) -> np.ndarray:
    """Create anatomically accurate vascular network."""
    vascular_mask = np.zeros((size, size), dtype=np.float32)
    
    # Get tissue properties
    vascular_density = tissue_props["vascular_density"]
    fiber_direction = tissue_props["fiber_direction"]
    
    # Adjust for shape
    shape_type = shape_info["shape_type"]
    if shape_type == "elongated":
        vascular_density *= 1.2
    elif shape_type == "spherical":
        vascular_density *= 0.8
    
    # Create main vessels
    num_main_vessels = max(2, int(vascular_density * 8))
    
    for _ in range(num_main_vessels):
        vessel = create_anatomical_vessel(
            x=None, y=None, size=size,
            width_range=(3, 8),
            length_range=(size//4, size//2),
            curvature=0.3,
            branching_prob=0.3,
            fiber_direction=fiber_direction
        )
        vascular_mask = np.maximum(vascular_mask, vessel)
    
    # Create capillary networks
    num_capillaries = max(10, int(vascular_density * 25))
    
    for _ in range(num_capillaries):
        vessel = create_anatomical_vessel(
            x=None, y=None, size=size,
            width_range=(1, 3),
            length_range=(size//8, size//4),
            curvature=0.5,
            branching_prob=0.4,
            fiber_direction=fiber_direction
        )
        vascular_mask = np.maximum(vascular_mask, vessel * 0.6)
    
    # Apply anatomical blurring
    vascular_mask = gaussian_filter(vascular_mask, sigma=1.5)
    
    return np.clip(vascular_mask, 0, 1)

def create_bone_trabecular_pattern(x, y, size, texture_scale, roughness):
    """Create bone trabecular pattern."""
    scale_factor = 40 * texture_scale
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Trabecular structure
    trabecular = (
        np.sin(x * np.pi * scale_factor) * 0.4 +
        np.sin(y * np.pi * scale_factor) * 0.4 +
        np.sin((x + y) * np.pi * scale_factor * 0.7) * 0.3
    )
    
    # Haversian canals
    num_canals = random.randint(8, 15)
    for _ in range(num_canals):
        center_x = random.uniform(0.1, 0.9)
        center_y = random.uniform(0.1, 0.9)
        radius = random.uniform(0.02, 0.08)
        
        dx = (x - center_x) / radius
        dy = (y - center_y) / radius
        canal = np.exp(-(dx**2 + dy**2)) * 0.6
        pattern += canal
    
    pattern = pattern * 0.6 + trabecular * 0.4
    return np.clip(pattern, 0, 1)

def create_muscle_fiber_pattern(x, y, size, texture_scale, roughness, fiber_direction):
    """Create muscle fiber bundle pattern."""
    pattern = np.zeros((size, size), dtype=np.float32)
    
    scale_factor = 25 * texture_scale
    
    if fiber_direction == 'circular_longitudinal':
        # Heart muscle - mixed directions
        fiber_pattern = (
            np.sin(x * np.pi * scale_factor) * 0.3 +
            np.sin(y * np.pi * scale_factor * 0.8) * 0.25 +
            np.sin((x + y) * np.pi * scale_factor * 0.6) * 0.2
        )
    elif fiber_direction == 'longitudinal':
        # Longitudinal muscle fibers
        fiber_pattern = np.sin(x * np.pi * scale_factor) * 0.4
    elif fiber_direction == 'circular':
        # Circular muscle fibers
        fiber_pattern = np.sin(y * np.pi * scale_factor) * 0.4
    else:
        # Mixed fiber directions
        fiber_pattern = (
            np.sin(x * np.pi * scale_factor) * 0.2 +
            np.sin(y * np.pi * scale_factor) * 0.2
        )
    
    # Muscle fiber bundles
    num_bundles = random.randint(6, 12)
    for _ in range(num_bundles):
        center_x = random.uniform(0.1, 0.9)
        center_y = random.uniform(0.1, 0.9)
        radius_x = random.uniform(0.08, 0.2)
        radius_y = random.uniform(0.03, 0.08)
        
        dx = (x - center_x) / radius_x
        dy = (y - center_y) / radius_y
        
        bundle = np.exp(-(dx**2 + dy**2) * 2) * 0.4
        pattern += bundle
    
    pattern = pattern * 0.7 + fiber_pattern * 0.3
    return np.clip(pattern, 0, 1)

def create_glandular_lobular_pattern(x, y, size, texture_scale, roughness, organ_name):
    """Create glandular lobular pattern."""
    scale_factor = 30 * texture_scale
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Lobular structure
    num_lobules = random.randint(15, 25)
    for _ in range(num_lobules):
        center_x = random.uniform(0.1, 0.9)
        center_y = random.uniform(0.1, 0.9)
        radius = random.uniform(0.05, 0.15)
        
        dx = (x - center_x) / radius
        dy = (y - center_y) / radius
        
        lobule = np.exp(-(dx**2 + dy**2)) * 0.5
        pattern += lobule
    
    # Glandular texture
    glandular = (
        np.sin(x * np.pi * scale_factor) * 0.3 +
        np.sin(y * np.pi * scale_factor) * 0.3 +
        np.sin((x + y) * np.pi * scale_factor * 0.8) * 0.2
    )
    
    pattern = pattern * 0.6 + glandular * 0.4
    return np.clip(pattern, 0, 1)

def create_hollow_muscle_pattern(x, y, size, texture_scale, roughness):
    """Create hollow muscle (digestive tract) pattern."""
    scale_factor = 35 * texture_scale
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Muscle layers
    circular_muscle = np.sin(y * np.pi * scale_factor) * 0.4
    longitudinal_muscle = np.sin(x * np.pi * scale_factor * 0.6) * 0.3
    
    # Serosal surface
    serosal = (
        np.sin(x * np.pi * scale_factor * 0.3) * 0.2 +
        np.sin(y * np.pi * scale_factor * 0.3) * 0.2
    )
    
    pattern = circular_muscle + longitudinal_muscle + serosal
    return np.clip(pattern, 0, 1)

def create_colon_haustra_pattern(x, y, size, texture_scale, roughness):
    """Create realistic colon haustra (sacculations) pattern."""
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Create haustra (sacculations) - the characteristic segmented appearance
    haustra_spacing = 0.15 * texture_scale  # Distance between haustra
    haustra_width = 0.08 * texture_scale    # Width of each haustrum
    
    # Generate multiple rows of haustra
    for row in range(4):
        row_offset = row * 0.25
        for i in range(int(1.0 / haustra_spacing) + 1):
            haustrum_center = i * haustra_spacing + random.uniform(-0.02, 0.02)
            
            # Create elliptical haustrum
            dx = (x - haustrum_center) / haustra_width
            dy = (y - row_offset) / (haustra_width * 0.6)
            
            # Add some randomness to shape
            shape_noise = random.uniform(0.8, 1.2)
            haustrum = np.exp(-(dx**2 + dy**2 * shape_noise) * 3) * 0.6
            
            # Add slight indentation between haustra
            if i > 0:
                indent_center = (i - 0.5) * haustra_spacing
                indent_dx = (x - indent_center) / (haustra_width * 0.3)
                indent_dy = (y - row_offset) / (haustra_width * 0.4)
                indent = np.exp(-(indent_dx**2 + indent_dy**2) * 2) * 0.3
                haustrum -= indent
            
            pattern += haustrum
    
    return np.clip(pattern, 0, 1)

def create_teniae_coli_pattern(x, y, size, texture_scale, roughness):
    """Create teniae coli (longitudinal bands) pattern."""
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Three longitudinal bands (teniae coli)
    band_width = 0.08 * texture_scale
    band_positions = [0.25, 0.5, 0.75]  # Three bands across the width
    
    for band_pos in band_positions:
        # Add slight curvature to bands
        curvature = np.sin(x * np.pi * 2) * 0.02
        band_y = band_pos + curvature
        
        # Create the band
        dy = (y - band_y) / band_width
        band = np.exp(-dy**2 * 8) * 0.4
        
        # Add some texture variation along the band
        band_texture = np.sin(x * np.pi * 20 * texture_scale) * 0.1
        band += band_texture
        
        pattern += band
    
    return np.clip(pattern, 0, 1)

def create_vascular_smooth_pattern(x, y, size, texture_scale, roughness):
    """Create vascular smooth muscle pattern."""
    scale_factor = 20 * texture_scale
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Smooth muscle fibers
    fiber_pattern = np.sin(x * np.pi * scale_factor) * 0.5
    
    # Endothelial surface
    endothelial = (
        np.sin(x * np.pi * scale_factor * 0.5) * 0.2 +
        np.sin(y * np.pi * scale_factor * 0.5) * 0.2
    )
    
    pattern = fiber_pattern + endothelial
    return np.clip(pattern, 0, 1)

def create_soft_tissue_pattern(x, y, size, texture_scale, roughness):
    """Create general soft tissue pattern."""
    scale_factor = 25 * texture_scale
    pattern = np.zeros((size, size), dtype=np.float32)
    
    # Soft tissue texture
    tissue_pattern = (
        np.sin(x * np.pi * scale_factor) * 0.3 +
        np.sin(y * np.pi * scale_factor) * 0.3 +
        np.sin((x + y) * np.pi * scale_factor * 0.7) * 0.2
    )
    
    # Connective tissue
    num_fibers = random.randint(8, 15)
    for _ in range(num_fibers):
        center_x = random.uniform(0.1, 0.9)
        center_y = random.uniform(0.1, 0.9)
        radius = random.uniform(0.03, 0.08)
        
        dx = (x - center_x) / radius
        dy = (y - center_y) / radius
        
        fiber = np.exp(-(dx**2 + dy**2)) * 0.3
        pattern += fiber
    
    pattern = pattern * 0.7 + tissue_pattern * 0.3
    return np.clip(pattern, 0, 1)

def create_anatomical_surface_pattern(size: int, organ_name: str, tissue_props: dict, shape_info: dict) -> np.ndarray:
    """Create tissue-specific surface pattern based on anatomy."""
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    tissue_type = tissue_props["tissue_type"]
    texture_scale = tissue_props["texture_scale"]
    roughness = tissue_props["surface_roughness"]
    
    # Apply shape-aware modifications
    shape_type = shape_info["shape_type"]
    if shape_type == "elongated":
        # Stretch pattern along major axis
        aspect_ratios = shape_info["aspect_ratios"]
        max_axis = np.argmax(aspect_ratios)
        if max_axis == 0:  # X-axis is longest
            x *= 1.5
        elif max_axis == 1:  # Y-axis is longest
            y *= 1.5
    
    # Generate tissue-specific pattern
    if tissue_type == "bone":
        pattern = create_bone_trabecular_pattern(x, y, size, texture_scale, roughness)
    elif tissue_type == "cardiac_muscle":
        pattern = create_muscle_fiber_pattern(x, y, size, texture_scale, roughness, tissue_props["fiber_direction"])
    elif tissue_type == "glandular":
        pattern = create_glandular_lobular_pattern(x, y, size, texture_scale, roughness, organ_name)
    elif tissue_type == "hollow_muscle":
        # Special case for colon - use anatomically accurate patterns
        if "colon" in organ_name.lower():
            # Combine haustra and teniae coli patterns
            haustra_pattern = create_colon_haustra_pattern(x, y, size, texture_scale, roughness)
            teniae_pattern = create_teniae_coli_pattern(x, y, size, texture_scale, roughness)
            pattern = haustra_pattern * 0.7 + teniae_pattern * 0.3
        else:
            pattern = create_hollow_muscle_pattern(x, y, size, texture_scale, roughness)
    elif tissue_type == "vascular_smooth":
        pattern = create_vascular_smooth_pattern(x, y, size, texture_scale, roughness)
    else:
        pattern = create_soft_tissue_pattern(x, y, size, texture_scale, roughness)
    
    # Apply curvature-based modifications
    curvature_level = shape_info["curvature_level"]
    if curvature_level > 0.7:
        # High curvature - add more detail
        detail_noise = np.random.normal(0, 0.1, (size, size))
        pattern += detail_noise
    elif curvature_level < 0.3:
        # Low curvature - smooth more
        pattern = gaussian_filter(pattern, sigma=2.0)
    
    return np.clip(pattern, 0, 1)

def create_realistic_organ_diffuse(
    organ_name: str,
    base_color: Tuple[int, int, int],
    size: int = 512,
    model_path: Optional[str] = None
) -> Image.Image:
    """Create a realistic organ diffuse texture with anatomical accuracy."""
    
    print(f"    Creating {organ_name} diffuse texture {size}x{size}px with base color RGB{base_color}")
    
    # Get anatomical tissue properties
    tissue_props = get_anatomical_tissue_properties(organ_name)
    
    # Analyze voxel shape if model path provided
    shape_info = None
    if model_path and os.path.exists(model_path):
        print(f"    - Analyzing voxel shape from {model_path}...")
        shape_info = analyze_voxel_shape(model_path)
        print(f"    - Shape type: {shape_info['shape_type']}, Curvature: {shape_info['curvature_level']:.2f}")
    else:
        # Default shape info
        shape_info = {
            "shape_type": "irregular",
            "curvature_level": 0.5,
            "aspect_ratios": [1, 1, 1]
        }
    
    # Blend base color with anatomical base color
    anatomical_base = np.array(tissue_props["color_base"])
    base_rgb = np.array(base_color)
    blended_color = (base_rgb * 0.7 + anatomical_base * 0.3).astype(np.uint8)
    
    # Generate anatomical surface pattern
    print("    - Generating vascular surface pattern...")
    surface_pattern = create_anatomical_surface_pattern(size, organ_name, tissue_props, shape_info)
    
    # Generate vascular network
    print("    - Generating anatomical vascular network...")
    vascular_mask = create_anatomical_vascular_network(size, organ_name, tissue_props, shape_info)
    
    # Create base texture
    texture = np.ones((size, size, 3), dtype=np.float32)
    
    # Apply base color
    for i in range(3):
        texture[:, :, i] *= blended_color[i] / 255.0
    
    # Apply color variations
    color_variation = tissue_props["color_variation"] / 255.0
    color_noise = np.random.normal(0, color_variation, (size, size, 3))
    texture += color_noise
    
    # Apply surface pattern
    surface_intensity = tissue_props["surface_roughness"]
    for i in range(3):
        texture[:, :, i] += surface_pattern * surface_intensity * 0.1
    
    # Apply vascular network colors
    if "colon" in organ_name.lower():
        # Colon-specific vascular coloring - more subtle, following teniae coli
        vascular_color = [0.85, 0.45, 0.45]  # Slightly more pink
        for i in range(3):
            texture[:, :, i] = np.where(
                vascular_mask > 0.15,
                texture[:, :, i] * (1 - vascular_mask * 0.2) + vascular_color[i] * vascular_mask * 0.2,
                texture[:, :, i]
            )
    else:
        # Standard vascular coloring
        vascular_color = [0.8, 0.4, 0.4]  # Reddish for vessels
        for i in range(3):
            texture[:, :, i] = np.where(
                vascular_mask > 0.1,
                texture[:, :, i] * (1 - vascular_mask * 0.3) + vascular_color[i] * vascular_mask * 0.3,
                texture[:, :, i]
            )
    
    # Add fatty tissue variations
    fat_content = tissue_props["fat_content"]
    if fat_content > 0.05:
        fat_mask = np.random.random((size, size)) < fat_content * 0.1
        fat_color = [0.9, 0.85, 0.7]  # Yellowish fat
        for i in range(3):
            texture[:, :, i] = np.where(
                fat_mask,
                texture[:, :, i] * 0.7 + fat_color[i] * 0.3,
                texture[:, :, i]
            )
    
    # Add anatomical noise
    anatomical_noise = np.random.normal(0, 0.02, (size, size, 3))
    texture += anatomical_noise
    
    # Colon-specific enhancements
    if "colon" in organ_name.lower():
        # Add subtle moisture/gloss effect
        moisture_noise = np.random.normal(0, 0.01, (size, size, 3))
        texture += moisture_noise
        
        # Enhance haustra depth with shadows
        haustra_shadows = surface_pattern * 0.05
        for i in range(3):
            texture[:, :, i] -= haustra_shadows
        
        # Add slight color variation in haustra
        haustra_color_variation = surface_pattern * 0.03
        texture[:, :, 0] += haustra_color_variation  # Slightly more red in haustra
        texture[:, :, 1] -= haustra_color_variation * 0.5  # Slightly less green
    
    # Apply shape-aware lighting
    if shape_info["shape_type"] == "spherical":
        # Spherical lighting
        center_x, center_y = size // 2, size // 2
        x_coords, y_coords = np.meshgrid(np.arange(size), np.arange(size))
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = size // 2
        lighting = 1.0 - (distance / max_distance) * 0.2
        lighting = np.clip(lighting, 0.7, 1.0)
        
        for i in range(3):
            texture[:, :, i] *= lighting
    
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

def load_label_colors() -> Dict[str, Tuple[int, int, int]]:
    """Load label colors from the JSON file."""
    try:
        with open('vista3d_label_colors.json', 'r') as f:
            data = json.load(f)
            return {item['label']: tuple(item['rgb']) for item in data}
    except Exception as e:
        print(f"Warning: Could not load label colors: {e}")
        return {}

def find_best_color_match(organ_name: str, label_colors: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Find the best color match for the organ name."""
    organ_lower = organ_name.lower()
    
    # Direct match
    if organ_lower in label_colors:
        return label_colors[organ_lower]
    
    # Partial matches
    for label, color in label_colors.items():
        if label.lower() in organ_lower or organ_lower in label.lower():
            return color
    
    # Default color
    return (180, 140, 120)

def process_model_directory(model_dir: str, size: int = 512, overwrite: bool = False) -> bool:
    """Process a single model directory to generate diffuse texture."""
    model_name = os.path.basename(model_dir)
    
    # Load label colors
    label_colors = load_label_colors()
    base_color = find_best_color_match(model_name, label_colors)
    
    print(f"[*] Processing '{model_name}'...")
    print(f"    ✓ Exact match: '{model_name}' -> '{model_name}' RGB{base_color}")
    
    # Check if texture already exists
    texture_path = os.path.join(model_dir, "textures", "diffuse.png")
    if os.path.exists(texture_path) and not overwrite:
        print(f"    ✓ Texture already exists: {texture_path}")
        return True
    
    try:
        # Find model file for shape analysis
        model_path = None
        possible_paths = [
            os.path.join(model_dir, "scene.glb"),
            os.path.join(model_dir, "scene.gltf"),
            os.path.join(model_dir, "..", "glb", f"{model_name}.glb")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        # Generate texture
        diffuse_img = create_realistic_organ_diffuse(
            model_name,
            base_color,
            size,
            model_path
        )
        
        # Save texture
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        diffuse_img.save(texture_path)
        print(f"    ✓ Saved diffuse texture: {texture_path}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        return False

def main():
    """Main function to process all models or specific organ."""
    parser = argparse.ArgumentParser(description="Generate realistic organ diffuse textures")
    parser.add_argument("--organ", help="Specific organ to process")
    parser.add_argument("--size", type=int, default=512, help="Texture size (default: 512)")
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
        
        success = process_model_directory(organ_dir, args.size, args.overwrite)
        return 0 if success else 1
    else:
        # Process all models
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d))]
        
        if not model_dirs:
            print(f"No model directories found in '{models_dir}'")
            return 1
        
        print(f"Found {len(model_dirs)} model directories")
        
        successful = 0
        failed = 0
        
        for model_dir in sorted(model_dirs):
            model_path = os.path.join(models_dir, model_dir)
            if process_model_directory(model_path, args.size, args.overwrite):
                successful += 1
            else:
                failed += 1
        
        print(f"\n[*] Summary: {successful} successful, {failed} failed")
        return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
