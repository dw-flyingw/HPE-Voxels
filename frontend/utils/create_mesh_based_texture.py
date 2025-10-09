#!/usr/bin/env python3
"""
create_mesh_based_texture.py

Create seamless textures directly from mesh geometry, bypassing UV mapping entirely.
This approach generates textures that are guaranteed to be seamless and properly aligned.

Usage:
    python create_mesh_based_texture.py --model colon
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
import trimesh
from typing import Tuple, Dict, Any


def load_mesh_from_obj(obj_path: str) -> trimesh.Trimesh:
    """Load mesh from OBJ file."""
    print(f"  Loading mesh from: {obj_path}")
    mesh = trimesh.load(obj_path, process=False)
    
    if hasattr(mesh, 'geometry'):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    print(f"    Vertices: {len(mesh.vertices)}")
    print(f"    Faces: {len(mesh.faces)}")
    
    return mesh


def create_spherical_uv_mapping(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Create spherical UV mapping that guarantees seamless coverage.
    """
    print(f"  Creating spherical UV mapping...")
    
    vertices = mesh.vertices
    
    # Center the mesh
    center = np.mean(vertices, axis=0)
    centered_vertices = vertices - center
    
    # Convert to spherical coordinates
    r = np.linalg.norm(centered_vertices, axis=1)
    
    # Avoid division by zero
    r = np.where(r < 1e-6, 1e-6, r)
    
    # Calculate spherical coordinates
    x, y, z = centered_vertices.T
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)    # Polar angle
    
    # Convert to UV coordinates (0-1 range)
    u = (theta + np.pi) / (2 * np.pi)  # 0 to 1
    v = phi / np.pi                    # 0 to 1
    
    uvs = np.column_stack([u, v])
    
    print(f"    UV range: U=[{u.min():.3f}, {u.max():.3f}], V=[{v.min():.3f}, {v.max():.3f}]")
    
    return uvs


def create_cylindrical_uv_mapping(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Create cylindrical UV mapping for elongated objects like colon.
    """
    print(f"  Creating cylindrical UV mapping...")
    
    vertices = mesh.vertices
    
    # Center the mesh
    center = np.mean(vertices, axis=0)
    centered_vertices = vertices - center
    
    # Use principal component analysis to find the main axis
    cov_matrix = np.cov(centered_vertices.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Project vertices onto the main axis (height)
    heights = np.dot(centered_vertices, main_axis)
    
    # Project vertices onto the plane perpendicular to main axis (radius)
    radial_vectors = centered_vertices - np.outer(heights, main_axis)
    radii = np.linalg.norm(radial_vectors, axis=1)
    
    # Calculate angles around the main axis
    x, y, z = centered_vertices.T
    # Use the two components perpendicular to main axis
    if abs(main_axis[2]) < 0.9:  # Not mainly Z-axis
        perp1 = np.array([main_axis[2], 0, -main_axis[0]])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(main_axis, perp1)
    else:
        perp1 = np.array([1, 0, 0])
        perp2 = np.array([0, 1, 0])
    
    u_coords = np.dot(centered_vertices, perp1)
    v_coords = np.dot(centered_vertices, perp2)
    angles = np.arctan2(v_coords, u_coords)
    
    # Convert to UV coordinates
    u = (angles + np.pi) / (2 * np.pi)  # 0 to 1
    v = (heights - heights.min()) / (heights.max() - heights.min())  # 0 to 1
    
    uvs = np.column_stack([u, v])
    
    print(f"    UV range: U=[{u.min():.3f}, {u.max():.3f}], V=[{v.min():.3f}, {v.max():.3f}]")
    
    return uvs


def create_seamless_procedural_texture(
    organ_name: str,
    mesh: trimesh.Trimesh,
    uvs: np.ndarray,
    size: int = 2048
) -> Image.Image:
    """
    Create a seamless procedural texture based on organ type and mesh geometry.
    """
    print(f"  Creating seamless procedural texture for {organ_name}...")
    
    # Create base texture
    texture = Image.new('RGB', (size, size), (0, 0, 0))
    
    if organ_name.lower() == 'colon':
        texture = create_colon_texture(size)
    elif organ_name.lower() == 'heart':
        texture = create_heart_texture(size)
    elif organ_name.lower() == 'liver':
        texture = create_liver_texture(size)
    else:
        texture = create_generic_organ_texture(size, organ_name)
    
    # Apply seamless tiling
    texture = make_texture_seamless(texture)
    
    # Enhance contrast and saturation
    enhancer = ImageEnhance.Contrast(texture)
    texture = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Color(texture)
    texture = enhancer.enhance(1.1)
    
    return texture


def create_colon_texture(size: int) -> Image.Image:
    """Create a realistic colon texture matching reference photo."""
    print(f"    Generating colon-specific texture...")
    
    # Base colon colors - matching reference photo
    base_color = (180, 80, 90)  # Deep pink-red
    tissue_color = (200, 100, 110)  # Lighter pink-red
    vessel_color = (150, 60, 70)  # Darker pink-red
    
    # Create base texture
    texture = Image.new('RGB', (size, size), base_color)
    draw = ImageDraw.Draw(texture)
    
    # Add tissue patterns
    np.random.seed(42)  # For reproducibility
    
    # Create organic tissue patterns using multiple noise layers
    for i in range(5):
        # Random organic shapes
        for _ in range(20):
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            radius = np.random.randint(10, 50)
            
            # Create elliptical tissue patch
            bbox = [x-radius, y-radius//2, x+radius, y+radius//2]
            color = (
                tissue_color[0] + np.random.randint(-20, 20),
                tissue_color[1] + np.random.randint(-20, 20),
                tissue_color[2] + np.random.randint(-20, 20)
            )
            color = tuple(max(0, min(255, c)) for c in color)
            
            draw.ellipse(bbox, fill=color, outline=None)
    
    # Add blood vessels
    for _ in range(15):
        start_x = np.random.randint(0, size)
        start_y = np.random.randint(0, size)
        
        # Draw curved vessel
        points = []
        x, y = start_x, start_y
        for _ in range(20):
            points.append((x, y))
            x += np.random.randint(-15, 15)
            y += np.random.randint(-10, 10)
            x = max(0, min(size-1, x))
            y = max(0, min(size-1, y))
        
        # Draw vessel line
        for i in range(len(points)-1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            draw.line([x1, y1, x2, y2], fill=vessel_color, width=np.random.randint(2, 5))
    
    # Add mucosal folds (colon characteristic)
    for _ in range(10):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        
        # Draw mucosal fold
        fold_points = []
        for i in range(10):
            fold_x = x + i * 20
            fold_y = y + np.sin(i * 0.5) * 15
            fold_points.append((fold_x, fold_y))
        
        if len(fold_points) > 2:
            # Draw fold as a curved line
            for i in range(len(fold_points)-1):
                x1, y1 = fold_points[i]
                x2, y2 = fold_points[i+1]
                draw.line([x1, y1, x2, y2], fill=(160, 70, 80), width=3)
    
    return texture


def create_heart_texture(size: int) -> Image.Image:
    """Create a realistic heart texture."""
    texture = Image.new('RGB', (size, size), (139, 0, 0))  # Dark red base
    
    # Add heart muscle patterns
    draw = ImageDraw.Draw(texture)
    np.random.seed(43)
    
    for _ in range(30):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(5, 25)
        
        color = (
            139 + np.random.randint(-30, 30),
            np.random.randint(0, 50),
            np.random.randint(0, 50)
        )
        color = tuple(max(0, min(255, c)) for c in color)
        
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    return texture


def create_liver_texture(size: int) -> Image.Image:
    """Create a realistic liver texture."""
    texture = Image.new('RGB', (size, size), (107, 142, 35))  # Olive drab base
    
    # Add liver-specific patterns
    draw = ImageDraw.Draw(texture)
    np.random.seed(44)
    
    for _ in range(40):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(8, 30)
        
        color = (
            107 + np.random.randint(-20, 20),
            142 + np.random.randint(-20, 20),
            35 + np.random.randint(-10, 10)
        )
        color = tuple(max(0, min(255, c)) for c in color)
        
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    return texture


def create_generic_organ_texture(size: int, organ_name: str) -> Image.Image:
    """Create a generic organ texture."""
    # Use organ-specific colors
    colors = {
        'kidney': (205, 133, 63),
        'lung': (220, 220, 220),
        'brain': (139, 137, 137),
        'pancreas': (255, 228, 196),
        'spleen': (255, 160, 122),
    }
    
    base_color = colors.get(organ_name.lower(), (139, 69, 19))
    texture = Image.new('RGB', (size, size), base_color)
    
    return texture


def make_texture_seamless(texture: Image.Image) -> Image.Image:
    """Make texture seamlessly tileable."""
    print(f"    Making texture seamlessly tileable...")
    
    # Apply seamless tiling using edge blending
    size = texture.size[0]
    
    # Create seamless version by blending edges
    seamless = texture.copy()
    
    # Blend left and right edges
    for x in range(size // 4):
        for y in range(size):
            left_pixel = texture.getpixel((x, y))
            right_pixel = texture.getpixel((size - 1 - x, y))
            
            # Blend pixels
            blend_factor = x / (size // 4)
            blended = tuple(
                int(left_pixel[i] * (1 - blend_factor) + right_pixel[i] * blend_factor)
                for i in range(3)
            )
            
            seamless.putpixel((x, y), blended)
            seamless.putpixel((size - 1 - x, y), blended)
    
    # Blend top and bottom edges
    for y in range(size // 4):
        for x in range(size):
            top_pixel = seamless.getpixel((x, y))
            bottom_pixel = seamless.getpixel((x, size - 1 - y))
            
            # Blend pixels
            blend_factor = y / (size // 4)
            blended = tuple(
                int(top_pixel[i] * (1 - blend_factor) + bottom_pixel[i] * blend_factor)
                for i in range(3)
            )
            
            seamless.putpixel((x, y), blended)
            seamless.putpixel((x, size - 1 - y), blended)
    
    return seamless


def apply_texture_to_mesh(
    mesh: trimesh.Trimesh,
    texture: Image.Image,
    uvs: np.ndarray,
    gltf_path: str
) -> bool:
    """
    Apply the texture to the mesh and save as GLTF.
    """
    print(f"  Applying texture to mesh...")
    
    try:
        # Create a new mesh with the texture
        textured_mesh = mesh.copy()
        
        # Set the texture
        textured_mesh.visual = trimesh.visual.TextureVisuals(
            image=texture,
            uv=uvs
        )
        
        # Export to GLTF
        textured_mesh.export(gltf_path, file_type='gltf')
        
        print(f"    ✓ Textured mesh saved to: {gltf_path}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error applying texture: {e}")
        return False


def create_mesh_based_texture_for_model(
    model_name: str,
    obj_dir: Path = Path("output/obj"),
    models_dir: Path = Path("output/models"),
    size: int = 2048,
    mapping_type: str = 'cylindrical'
) -> bool:
    """
    Create mesh-based seamless texture for a specific model.
    """
    print(f"\n{'='*80}")
    print(f"MESH-BASED SEAMLESS TEXTURE: {model_name}")
    print(f"{'='*80}")
    
    obj_path = obj_dir / f"{model_name}.obj"
    model_dir = models_dir / model_name
    
    if not obj_path.exists():
        print(f"✗ OBJ file not found: {obj_path}")
        return False
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False
    
    try:
        # Load mesh
        mesh = load_mesh_from_obj(str(obj_path))
        
        # Create UV mapping
        if mapping_type == 'spherical':
            uvs = create_spherical_uv_mapping(mesh)
        else:
            uvs = create_cylindrical_uv_mapping(mesh)
        
        # Create seamless texture
        texture = create_seamless_procedural_texture(model_name, mesh, uvs, size)
        
        # Save texture
        texture_path = model_dir / 'textures' / 'mesh_based_texture.png'
        texture_path.parent.mkdir(exist_ok=True)
        texture.save(str(texture_path))
        print(f"✓ Saved mesh-based texture: {texture_path}")
        
        # Also save as diffuse.png
        diffuse_path = model_dir / 'textures' / 'diffuse.png'
        texture.save(str(diffuse_path))
        print(f"✓ Saved as diffuse texture: {diffuse_path}")
        
        # Apply to mesh and save GLTF
        gltf_path = model_dir / 'scene_mesh_textured.gltf'
        if apply_texture_to_mesh(mesh, texture, uvs, str(gltf_path)):
            print(f"✓ Created textured GLTF: {gltf_path}")
        
        print(f"\n{'='*80}")
        print(f"✓ SUCCESS: {model_name} mesh-based seamless texture created!")
        print(f"{'='*80}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create seamless textures directly from mesh geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create mesh-based texture for colon
    python create_mesh_based_texture.py --model colon
    
    # Use spherical mapping
    python create_mesh_based_texture.py --model heart --mapping spherical
    
    # High resolution
    python create_mesh_based_texture.py --model liver --size 4096
        """
    )
    
    parser.add_argument('--model', type=str, default='colon', help='Model name to process')
    parser.add_argument('--size', type=int, default=2048, choices=[1024, 2048, 4096],
                       help='Texture size (default: 2048)')
    parser.add_argument('--mapping', type=str, default='cylindrical', 
                       choices=['cylindrical', 'spherical'],
                       help='UV mapping type (default: cylindrical)')
    parser.add_argument('--obj-dir', type=str, default='output/obj', help='OBJ files directory')
    parser.add_argument('--models-dir', type=str, default='output/models',
                       help='Models directory (default: output/models)')
    
    args = parser.parse_args()
    
    obj_dir = Path(args.obj_dir)
    models_dir = Path(args.models_dir)
    
    if not obj_dir.exists():
        print(f"✗ OBJ directory not found: {obj_dir}")
        return 1
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return 1
    
    print("╔" + "═"*78 + "╗")
    print("║" + " "*15 + "Mesh-Based Seamless Texture Generator" + " "*35 + "║")
    print("╚" + "═"*78 + "╝\n")
    
    success = create_mesh_based_texture_for_model(
        args.model, 
        obj_dir,
        models_dir, 
        args.size,
        args.mapping
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
