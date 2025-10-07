"""
OBJ Texture Generator - Business Logic
Handles file operations, Flux API calls, and texture application
"""

import os
import tempfile
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple
import trimesh
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_flux_server_url() -> str:
    """Get Flux server URL from environment or use default."""
    port = os.getenv("FLUX_SERVER_PORT", "8000")
    host = os.getenv("FLUX_HOST", "localhost")
    return f"http://{host}:{port}"


def check_flux_server_health(flux_url: str) -> Dict:
    """Check if Flux server is healthy."""
    try:
        response = requests.get(f"{flux_url}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"Server returned {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def upload_obj_file(uploaded_file) -> Tuple[str, Dict]:
    """
    Save uploaded OBJ file and return path and metadata.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (file_path, metadata_dict)
    """
    # Create temp directory for uploaded files
    temp_dir = Path(tempfile.gettempdir()) / "obj_texture_generator"
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = temp_dir / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract metadata
    metadata = analyze_obj_file(str(file_path))
    
    return str(file_path), metadata


def analyze_obj_file(obj_path: str) -> Dict:
    """
    Analyze OBJ file and extract metadata.
    
    Args:
        obj_path: Path to OBJ file
        
    Returns:
        Dictionary with model information
    """
    try:
        # Try to load with trimesh
        mesh = trimesh.load(obj_path, process=False)
        
        # Get basic info
        vertices = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0
        
        # Check for UVs and normals
        has_uvs = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
        has_normals = hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None
        
        # File size
        size_bytes = os.path.getsize(obj_path)
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024*1024 else f"{size_bytes / (1024*1024):.1f} MB"
        
        return {
            'vertices': vertices,
            'faces': faces,
            'has_uvs': has_uvs,
            'has_normals': has_normals,
            'size': size_str,
            'bounds': mesh.bounds.tolist() if hasattr(mesh, 'bounds') else None
        }
    except Exception as e:
        print(f"Error analyzing OBJ: {e}")
        
        # Fallback: count lines
        vertices = 0
        faces = 0
        has_uvs = False
        
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices += 1
                elif line.startswith('f '):
                    faces += 1
                elif line.startswith('vt '):
                    has_uvs = True
        
        size_bytes = os.path.getsize(obj_path)
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024*1024 else f"{size_bytes / (1024*1024):.1f} MB"
        
        return {
            'vertices': vertices,
            'faces': faces,
            'has_uvs': has_uvs,
            'has_normals': False,
            'size': size_str
        }


def generate_texture_from_prompt(
    prompt: str,
    flux_url: str,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_steps: int = 50,
    seed: Optional[int] = None
) -> Optional[str]:
    """
    Generate texture image from prompt using Flux API.
    
    Args:
        prompt: Text prompt for texture generation
        flux_url: Flux server URL
        width: Texture width
        height: Texture height
        guidance_scale: Guidance scale for generation
        num_steps: Number of inference steps
        seed: Random seed (optional)
        
    Returns:
        Path to generated texture image or None if failed
    """
    try:
        # Prepare request
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        # Make request to Flux server
        print(f"Sending request to {flux_url}/generate")
        print(f"Payload: {payload}")
        
        response = requests.post(
            f"{flux_url}/generate",
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            # Save image to temp directory
            temp_dir = Path(tempfile.gettempdir()) / "obj_texture_generator"
            temp_dir.mkdir(exist_ok=True)
            
            texture_path = temp_dir / "generated_texture.png"
            
            # Save image
            with open(texture_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Texture saved to {texture_path}")
            return str(texture_path)
        else:
            print(f"Error: Server returned {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error generating texture: {e}")
        return None


def apply_texture_to_obj(obj_path: str, texture_path: str) -> Optional[str]:
    """
    Apply texture to OBJ file by creating/updating MTL file.
    
    Args:
        obj_path: Path to OBJ file
        texture_path: Path to texture image
        
    Returns:
        Path to output OBJ file or None if failed
    """
    try:
        # Create output directory
        temp_dir = Path(tempfile.gettempdir()) / "obj_texture_generator" / "output"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Output paths
        output_obj_path = temp_dir / "textured_model.obj"
        output_mtl_path = temp_dir / "textured_model.mtl"
        output_texture_path = temp_dir / "texture.png"
        
        # Load the mesh
        mesh = trimesh.load(obj_path, process=False)
        
        # Check if mesh has UV coordinates
        has_uvs = False
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            has_uvs = True
            print("Model has UV coordinates")
        else:
            print("Model doesn't have UV coordinates - generating basic UVs")
            # Generate basic spherical UV mapping
            mesh = generate_uv_mapping(mesh)
            has_uvs = True
        
        # Copy texture to output directory
        import shutil
        shutil.copy(texture_path, output_texture_path)
        
        # Create MTL file
        mtl_content = f"""# Material file for textured model
newmtl TexturedMaterial
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.3 0.3 0.3
Ns 32.0
d 1.0
illum 2
map_Kd texture.png
"""
        
        with open(output_mtl_path, 'w') as f:
            f.write(mtl_content)
        
        # Read original OBJ
        with open(obj_path, 'r') as f:
            obj_lines = f.readlines()
        
        # Modify OBJ to reference MTL
        output_lines = []
        mtl_referenced = False
        usemtl_added = False
        
        for line in obj_lines:
            # Skip existing mtllib and usemtl lines
            if line.startswith('mtllib') or line.startswith('usemtl'):
                continue
            
            # Add mtllib reference at the start (after comments)
            if not mtl_referenced and not line.startswith('#'):
                output_lines.append('mtllib textured_model.mtl\n')
                mtl_referenced = True
            
            # Add usemtl before first face
            if not usemtl_added and line.startswith('f '):
                output_lines.append('usemtl TexturedMaterial\n')
                usemtl_added = True
            
            output_lines.append(line)
        
        # If no faces found, still add the references
        if not mtl_referenced:
            output_lines.insert(0, 'mtllib textured_model.mtl\n')
        
        # Write output OBJ
        with open(output_obj_path, 'w') as f:
            f.writelines(output_lines)
        
        print(f"Textured model saved to {output_obj_path}")
        print(f"Material file saved to {output_mtl_path}")
        print(f"Texture saved to {output_texture_path}")
        
        return str(output_obj_path)
        
    except Exception as e:
        print(f"Error applying texture: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_uv_mapping(mesh) -> trimesh.Trimesh:
    """
    Generate basic UV mapping for a mesh without UVs.
    Uses spherical projection.
    
    Args:
        mesh: Trimesh object
        
    Returns:
        Mesh with UV coordinates
    """
    import numpy as np
    
    # Get vertices
    vertices = mesh.vertices
    
    # Calculate spherical coordinates
    # Center the mesh
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    # Calculate spherical coordinates
    r = np.sqrt(np.sum(centered**2, axis=1))
    r = np.where(r == 0, 1e-10, r)  # Avoid division by zero
    
    # Theta (azimuth): angle in XY plane
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    # Normalize to [0, 1]
    u = (theta + np.pi) / (2 * np.pi)
    
    # Phi (elevation): angle from Z axis
    phi = np.arccos(np.clip(centered[:, 2] / r, -1, 1))
    # Normalize to [0, 1]
    v = phi / np.pi
    
    # Stack to create UV coordinates
    uvs = np.column_stack([u, v])
    
    # Create a new mesh with UV coordinates
    # Note: For proper UV mapping, we need to handle the mesh visual
    if not hasattr(mesh, 'visual'):
        mesh.visual = trimesh.visual.TextureVisuals()
    
    mesh.visual.uv = uvs
    
    return mesh


def load_obj_preview(obj_path: str) -> Optional[Dict]:
    """
    Load OBJ file for 3D preview.
    
    Args:
        obj_path: Path to OBJ file
        
    Returns:
        Dictionary with preview data or None
    """
    try:
        mesh = trimesh.load(obj_path)
        
        # Get preview image (screenshot)
        scene = mesh.scene()
        png = scene.save_image(resolution=[800, 600])
        
        return {
            'image': png,
            'bounds': mesh.bounds.tolist(),
            'centroid': mesh.centroid.tolist()
        }
    except Exception as e:
        print(f"Error loading preview: {e}")
        return None

