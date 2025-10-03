#!/usr/bin/env python3
"""
generate_heart_diffuse.py

Generates a highly realistic diffuse texture for heart models that matches the appearance
of actual heart tissue with vascular networks, muscle texture, and natural color variations.

Based on the heart reference image which shows:
- Rich reddish-brown base color with natural variations
- Prominent vascular network (coronary arteries and veins)
- Subtle muscle texture and organic surface details
- Fatty/connective tissue areas with lighter coloration
- Natural gradients and color transitions

Usage:
    python generate_heart_diffuse.py
    python generate_heart_diffuse.py --size 1024  # for higher resolution
    python generate_heart_diffuse.py --output heart/diffuse.png
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFilter
import numpy as np
import random


def create_vascular_network(size: int, base_color: tuple) -> np.ndarray:
    """
    Create a realistic vascular network pattern for the heart.
    
    Args:
        size: Texture size in pixels
        base_color: RGB base color of the heart tissue
        
    Returns:
        Vascular network mask as numpy array
    """
    # Create a black canvas
    vascular_mask = np.zeros((size, size), dtype=np.uint8)
    
    # Generate main coronary vessels
    num_main_vessels = random.randint(8, 12)
    
    for _ in range(num_main_vessels):
        # Random vessel path
        start_x = random.randint(0, size)
        start_y = random.randint(0, size)
        
        # Create curved vessel path
        points = []
        current_x, current_y = start_x, start_y
        
        for _ in range(random.randint(15, 25)):
            points.append((int(current_x), int(current_y)))
            
            # Add some curvature and randomness
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(8, 15)
            
            current_x += distance * np.cos(angle)
            current_y += distance * np.sin(angle)
            
            # Keep within bounds
            current_x = max(0, min(size-1, current_x))
            current_y = max(0, min(size-1, current_y))
    
        # Draw the vessel
        if len(points) > 1:
            # Create vessel width variation
            width = random.uniform(3, 8)
            
            # Draw vessel as thick line
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                # Create vessel segment
                vessel_img = PILImage.new('L', (size, size), 0)
                draw = ImageDraw.Draw(vessel_img)
                
                # Vary width along the vessel
                segment_width = width * random.uniform(0.8, 1.2)
                draw.line([(x1, y1), (x2, y2)], fill=255, width=int(segment_width))
                
                # Add to vascular mask
                vessel_array = np.array(vessel_img)
                vascular_mask = np.maximum(vascular_mask, vessel_array)
    
    # Add smaller branching vessels
    num_branch_vessels = random.randint(20, 30)
    
    for _ in range(num_branch_vessels):
        start_x = random.randint(0, size)
        start_y = random.randint(0, size)
        
        # Shorter, thinner vessels
        points = []
        current_x, current_y = start_x, start_y
        
        for _ in range(random.randint(5, 12)):
            points.append((int(current_x), int(current_y)))
            
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(3, 8)
            
            current_x += distance * np.cos(angle)
            current_y += distance * np.sin(angle)
            
            current_x = max(0, min(size-1, current_x))
            current_y = max(0, min(size-1, current_y))
        
        if len(points) > 1:
            width = random.uniform(1, 3)
            
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                
                vessel_img = PILImage.new('L', (size, size), 0)
                draw = ImageDraw.Draw(vessel_img)
                draw.line([(x1, y1), (x2, y2)], fill=255, width=int(width))
                
                vessel_array = np.array(vessel_img)
                vascular_mask = np.maximum(vascular_mask, vessel_array)
    
    # Apply Gaussian blur for smoothness
    vascular_img = PILImage.fromarray(vascular_mask)
    vascular_img = vascular_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    return np.array(vascular_img) / 255.0


def create_muscle_texture(size: int) -> np.ndarray:
    """
    Create muscle fiber texture pattern.
    
    Args:
        size: Texture size in pixels
        
    Returns:
        Muscle texture as numpy array (0-1 range)
    """
    # Create base noise
    noise = np.random.random((size, size))
    
    # Create directional patterns for muscle fibers
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Multiple frequency patterns for muscle texture
    fiber_pattern = (
        np.sin(x * np.pi * 8) * 0.3 +
        np.sin(y * np.pi * 6) * 0.2 +
        np.sin((x + y) * np.pi * 4) * 0.1
    )
    
    # Combine with noise
    muscle_texture = noise * 0.3 + fiber_pattern * 0.7 + 0.5
    muscle_texture = np.clip(muscle_texture, 0, 1)
    
    # Apply some smoothing
    muscle_img = PILImage.fromarray((muscle_texture * 255).astype(np.uint8))
    muscle_img = muscle_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    return np.array(muscle_img) / 255.0


def create_fatty_tissue_pattern(size: int) -> np.ndarray:
    """
    Create areas of fatty/connective tissue (lighter areas).
    
    Args:
        size: Texture size in pixels
        
    Returns:
        Fatty tissue mask as numpy array (0-1 range)
    """
    # Create blob-like patterns for fatty tissue
    fat_mask = np.zeros((size, size))
    
    # Generate several fatty tissue areas
    num_fat_areas = random.randint(6, 10)
    
    for _ in range(num_fat_areas):
        center_x = random.randint(size//4, 3*size//4)
        center_y = random.randint(size//4, 3*size//4)
        radius = random.randint(size//8, size//4)
        
        # Create circular area with soft edges
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Soft falloff
        fat_area = np.exp(-(distance / radius)**2)
        fat_mask += fat_area * random.uniform(0.3, 0.8)
    
    # Normalize and smooth
    fat_mask = np.clip(fat_mask, 0, 1)
    fat_img = PILImage.fromarray((fat_mask * 255).astype(np.uint8))
    fat_img = fat_img.filter(ImageFilter.GaussianBlur(radius=8))
    
    return np.array(fat_img) / 255.0


def create_realistic_heart_diffuse(base_color: tuple, size: int = 512) -> PILImage.Image:
    """
    Create a highly realistic heart diffuse texture.
    
    Args:
        base_color: RGB color as (r, g, b) where each value is 0-255
        size: Texture size in pixels (will be square)
        
    Returns:
        PIL Image object with realistic heart tissue appearance
    """
    print(f"    Creating heart diffuse texture {size}x{size}px with base color RGB{base_color}")
    
    # Set random seed for reproducible results (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Convert base color to numpy array for easier manipulation
    base_color = np.array(base_color, dtype=np.float32)
    
    # Create base image
    img_array = np.full((size, size, 3), base_color, dtype=np.float32)
    
    # Generate texture components
    print("    - Generating vascular network...")
    vascular_mask = create_vascular_network(size, base_color)
    
    print("    - Generating muscle texture...")
    muscle_texture = create_muscle_texture(size)
    
    print("    - Generating fatty tissue areas...")
    fat_mask = create_fatty_tissue_pattern(size)
    
    # Create color variations
    print("    - Applying color variations...")
    
    # Darker areas (deeper tissue, shadows)
    dark_variation = np.random.normal(0, 15, img_array.shape)
    img_array += dark_variation
    
    # Vascular network colors
    # Veins (darker, more blue-purple)
    vein_color = np.array([180, 15, 45])  # Darker red with blue tint
    # Arteries (slightly lighter, more red)
    artery_color = np.array([240, 25, 70])  # Brighter red
    
    # Apply vascular network
    for c in range(3):
        # Mix vein and artery colors based on vascular mask intensity
        vein_component = vascular_mask * (vein_color[c] - base_color[c])
        artery_component = vascular_mask * (artery_color[c] - base_color[c])
        
        # Blend based on local intensity (darker areas = veins, lighter = arteries)
        vascular_blend = vein_component * 0.7 + artery_component * 0.3
        img_array[:, :, c] += vascular_blend * 0.6
    
    # Apply muscle texture
    muscle_variation = (muscle_texture - 0.5) * 20  # Scale muscle texture effect
    for c in range(3):
        img_array[:, :, c] += muscle_variation
    
    # Apply fatty tissue (lighter areas)
    fat_color_adjustment = np.array([25, 20, 15])  # Lighter, more yellowish
    for c in range(3):
        img_array[:, :, c] += fat_mask * fat_color_adjustment[c]
    
    # Add subtle organic noise
    organic_noise = np.random.normal(0, 8, img_array.shape)
    img_array += organic_noise
    
    # Add gradient variations for natural lighting
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Subtle radial gradient from center
    center_gradient = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    center_gradient = 1.0 - np.clip(center_gradient * 1.5, 0, 1)
    
    # Apply subtle center lighting
    for c in range(3):
        img_array[:, :, c] += center_gradient * 10
    
    # Add some directional lighting variation
    directional_gradient = (x * 0.3 + y * 0.7) * 8
    for c in range(3):
        img_array[:, :, c] += directional_gradient
    
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 255)
    
    # Convert to PIL Image
    result_img = PILImage.fromarray(img_array.astype(np.uint8))
    
    # Apply final smoothing to blend everything together
    result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    print("    ✓ Heart diffuse texture created successfully")
    return result_img


def main():
    """Main function to generate realistic heart diffuse texture."""
    
    parser = argparse.ArgumentParser(
        description='Generate realistic heart diffuse texture with vascular networks and tissue variations'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output/models/heart/diffuse.png',
        help='Output path for the diffuse texture (default: ./output/models/heart/diffuse.png)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Texture size in pixels (default: 512)'
    )
    parser.add_argument(
        '--base-color',
        type=int,
        nargs=3,
        default=[220, 20, 60],
        help='Base RGB color for heart tissue (default: 220 20 60)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"❤️  Realistic Heart Diffuse Texture Generator")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Size: {args.size}x{args.size}px")
    print(f"Base color: RGB{args.base_color}")
    print(f"{'='*60}\n")
    
    try:
        # Generate the realistic heart diffuse texture
        heart_diffuse = create_realistic_heart_diffuse(
            base_color=tuple(args.base_color),
            size=args.size
        )
        
        # Save the texture
        heart_diffuse.save(output_path, format='PNG', optimize=True)
        
        print(f"\n{'='*60}")
        print(f"[✓] Heart diffuse texture generated successfully!")
        print(f"    Output: {output_path}")
        print(f"    Size: {args.size}x{args.size}px")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        print(f"\n[!] Error generating heart diffuse texture: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
