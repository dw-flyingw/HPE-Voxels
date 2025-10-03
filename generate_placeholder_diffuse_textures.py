#!/usr/bin/env python3
"""
generate_placeholder_diffuse_textures.py

Generates simple placeholder diffuse.png textures for all models in the models folder based on vista3d_label_colors.json.
This script creates solid color textures using the anatomical colors from the JSON file.

Usage:
    python generate_placeholder_diffuse_textures.py
    python generate_placeholder_diffuse_textures.py --size 1024  # for higher resolution
    python generate_placeholder_diffuse_textures.py --models-dir ./output/models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image as PILImage
import numpy as np


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


def create_diffuse_texture(color: list, size: int = 512) -> PILImage.Image:
    """
    Create a simple placeholder diffuse texture with the specified color from vista3d_label_colors.json.
    
    Args:
        color: RGB color as [r, g, b] where each value is 0-255
        size: Texture size in pixels (will be square)
        
    Returns:
        PIL Image object with the placeholder diffuse map
    """
    # Create a simple solid color image with the anatomical color
    return PILImage.new('RGB', (size, size), tuple(color))


def get_color_for_model(model_name: str, label_info: dict) -> list:
    """
    Get the appropriate color for a model based on its name.
    
    Args:
        model_name: Name of the model (folder name)
        label_info: Dictionary of label information from vista3d_label_colors.json
        
    Returns:
        RGB color as [r, g, b]
    """
    color = [128, 128, 128]  # Default gray
    
    if label_info and model_name:
        # Find matching label by name
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
        
        # If no exact match, try partial matching for common cases
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
    """
    Process a single model directory and generate its diffuse texture.
    
    Args:
        model_dir: Path to the model directory
        label_info: Dictionary of label information
        texture_size: Size of the texture to generate
        overwrite: Whether to overwrite existing diffuse.png files
        
    Returns:
        True if successful, False otherwise
    """
    model_name = model_dir.name
    diffuse_path = model_dir / "diffuse.png"
    
    # Check if diffuse.png already exists
    if diffuse_path.exists() and not overwrite:
        print(f"[⊘] Skipping '{model_name}' - diffuse.png already exists")
        return True
    
    try:
        print(f"[*] Processing '{model_name}'...")
        
        # Get color for this model
        color = get_color_for_model(model_name, label_info)
        
        # Generate diffuse texture
        diffuse_img = create_diffuse_texture(color, size=texture_size)
        
        # Save the texture
        diffuse_img.save(diffuse_path, format='PNG', optimize=True)
        
        print(f"    ✓ Created: {diffuse_path.relative_to(model_dir.parent.parent)}")
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to process '{model_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to generate diffuse textures for all models."""
    
    parser = argparse.ArgumentParser(
        description='Generate simple placeholder diffuse.png textures for all models based on vista3d_label_colors.json'
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
        '--overwrite',
        action='store_true',
        help='Overwrite existing diffuse.png files'
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
    
    # Find all model directories (directories containing scene.glb)
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and (d / "scene.glb").exists()]
    
    if not model_dirs:
        print(f"[!] No model directories found in {models_dir}", file=sys.stderr)
        print(f"    (Looking for directories containing scene.glb)")
        return 1
    
    print(f"\n{'='*60}")
    print(f"🎨 Placeholder Diffuse Texture Generator")
    print(f"{'='*60}")
    print(f"Models directory: {models_dir}")
    print(f"Found {len(model_dirs)} model(s)")
    print(f"Texture size: {args.size}x{args.size}px")
    print(f"Overwrite existing: {args.overwrite}")
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

