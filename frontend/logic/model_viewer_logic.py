"""
Logic for model_viewer.py - handles texture generation via Flux API
"""

import os
import requests
import trimesh
import numpy as np
from PIL import Image
import io
import base64
import json
import re
from pathlib import Path


def load_vista3d_prompts():
    """Load Vista3D texture generation prompts from JSON file."""
    try:
        # Get the project root directory (two levels up from this file)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        prompts_json = project_root / "vista3d_prompts.json"
        
        with open(prompts_json, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Warning: Could not load Vista3D prompts: {e}")
        return {
            "default_template": "hyper photo-realistic human {structure} anatomical structure, medical photography, anatomically accurate surface texture, natural clinical appearance, high detail, 8K resolution, professional medical illustration",
            "prompts": []
        }


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name by removing directional prefixes and cleaning up.
    
    Examples:
        'left_kidney' -> 'kidney'
        'right_hip' -> 'hip'
        'vertebrae_L5' -> 'vertebrae l5'
    """
    # Remove common prefixes
    name = re.sub(r'^(left|right)_', '', model_name, flags=re.IGNORECASE)
    name = re.sub(r'_(left|right)$', '', name, flags=re.IGNORECASE)
    
    # Replace underscores with spaces
    name = name.replace('_', ' ').lower().strip()
    
    return name


def generate_texture_prompt(model_name: str) -> str:
    """
    Generate a hyper photo-realistic prompt for texture generation based on Vista3D anatomical structures.
    Reads prompts from vista3d_prompts.json for maintainability.
    
    Args:
        model_name: Name of the organ/model (e.g., 'colon', 'left_kidney', 'vertebrae_L5')
    
    Returns:
        str: Hyper photo-realistic prompt for texture generation
    """
    # Normalize the model name
    normalized_name = normalize_model_name(model_name)
    
    # Load prompts from JSON file
    prompts_data = load_vista3d_prompts()
    prompts_list = prompts_data.get('prompts', [])
    default_template = prompts_data.get('default_template', 
        'hyper photo-realistic human {structure} anatomical structure, medical photography, '
        'anatomically accurate surface texture, natural clinical appearance, high detail, '
        '8K resolution, professional medical illustration')
    
    # Try exact match first (by name)
    for prompt_entry in prompts_list:
        entry_name = prompt_entry.get('name', '').lower()
        if entry_name == normalized_name:
            return prompt_entry.get('prompt', default_template.format(structure=normalized_name))
    
    # Try partial match (for compound names like "left hip" matching "hip")
    for prompt_entry in prompts_list:
        entry_name = prompt_entry.get('name', '').lower()
        # Remove directional prefixes from entry name too
        entry_normalized = normalize_model_name(entry_name)
        if entry_normalized in normalized_name or normalized_name in entry_normalized:
            return prompt_entry.get('prompt', default_template.format(structure=normalized_name))
    
    # Use default template with the normalized name
    return default_template.format(structure=normalized_name)


def call_flux_api(prompt: str, width: int = 1024, height: int = 1024, 
                  guidance_scale: float = 3.5, num_steps: int = 50) -> Image.Image:
    """
    Call the Flux API to generate a texture image.
    
    Args:
        prompt: Text prompt for texture generation
        width: Image width
        height: Image height
        guidance_scale: Guidance scale for generation
        num_steps: Number of inference steps
    
    Returns:
        PIL.Image: Generated texture image
    
    Raises:
        Exception: If API call fails
    """
    # Get port from environment variable with fallback
    flux_port = int(os.environ.get('FLUX_SERVER_PORT', 8000))
    flux_host = os.environ.get('FLUX_HOST', 'localhost')
    api_url = f"http://{flux_host}:{flux_port}/generate"
    
    # Prepare request
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
        "return_base64": True
    }
    
    # Make API call
    try:
        response = requests.post(api_url, json=payload, timeout=300)  # 5 min timeout
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        if data.get('success') and data.get('image_base64'):
            # Decode base64 image
            img_data = base64.b64decode(data['image_base64'])
            image = Image.open(io.BytesIO(img_data))
            return image
        else:
            raise Exception("API returned no image data")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Flux API call failed: {str(e)}")


def generate_uv_mapping(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Generate UV coordinates for a mesh using spherical mapping.
    
    Args:
        mesh: Trimesh object
    
    Returns:
        np.ndarray: UV coordinates (N x 2)
    """
    vertices = mesh.vertices
    
    # Center vertices
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    # Normalize
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = centered / norms
    
    # Spherical mapping
    u = 0.5 + np.arctan2(normalized[:, 2], normalized[:, 0]) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(normalized[:, 1], -1, 1)) / np.pi
    
    return np.column_stack([u, v])


def apply_texture_to_model(model_path: str, texture_image: Image.Image, 
                           output_folder: str) -> dict:
    """
    Apply a texture image to a 3D model and save the result.
    
    Args:
        model_path: Path to the input model file
        texture_image: PIL Image to use as texture
        output_folder: Folder to save the textured model
    
    Returns:
        dict: Information about the saved files
    """
    # Load the model
    mesh = trimesh.load(model_path)
    
    # Handle Scene vs Mesh
    if hasattr(mesh, 'geometry'):
        # Combine all meshes in scene
        meshes = list(mesh.geometry.values())
        if len(meshes) == 0:
            raise ValueError("No meshes found in scene")
        mesh = trimesh.util.concatenate(meshes)
    
    # Generate or update UV mapping
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        uv_coords = generate_uv_mapping(mesh)
    else:
        uv_coords = mesh.visual.uv
    
    # Save texture image
    os.makedirs(output_folder, exist_ok=True)
    texture_subfolder = os.path.join(output_folder, 'textures')
    os.makedirs(texture_subfolder, exist_ok=True)
    
    texture_path = os.path.join(texture_subfolder, 'diffuse.png')
    texture_image.save(texture_path)
    
    # Create textured material
    material = trimesh.visual.material.SimpleMaterial(
        image=texture_image,
        diffuse=[255, 255, 255, 255]
    )
    
    # Create texture visual
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv_coords,
        image=texture_image,
        material=material
    )
    
    # Export as OBJ with MTL (best format for textures)
    obj_path = os.path.join(output_folder, 'textured_model.obj')
    mesh.export(obj_path, file_type='obj', include_texture=True)
    
    # Also export as GLTF/GLB (modern format)
    gltf_path = os.path.join(output_folder, 'scene.gltf')
    glb_path = os.path.join(output_folder, 'scene.glb')
    
    try:
        mesh.export(gltf_path, file_type='gltf', include_normals=True)
    except Exception as e:
        print(f"Warning: GLTF export failed: {e}")
    
    try:
        mesh.export(glb_path, file_type='glb', include_normals=True)
    except Exception as e:
        print(f"Warning: GLB export failed: {e}")
    
    return {
        'obj_path': obj_path,
        'gltf_path': gltf_path if os.path.exists(gltf_path) else None,
        'glb_path': glb_path if os.path.exists(glb_path) else None,
        'texture_path': texture_path,
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'uv_coords': len(uv_coords)
    }


def generate_photorealistic_texture(model_folder: str, model_name: str,
                                    custom_prompt: str = None,
                                    texture_size: int = 1024) -> dict:
    """
    Main function to generate photo-realistic texture for a model.
    
    Args:
        model_folder: Path to the model folder
        model_name: Name of the model (e.g., 'colon')
        custom_prompt: Optional custom prompt (if None, auto-generate)
        texture_size: Size of the texture (width/height)
    
    Returns:
        dict: Result information including paths and status
    """
    try:
        # Find the model file
        model_file = None
        for ext in ['glb', 'gltf', 'obj']:
            scene_file = os.path.join(model_folder, f'scene.{ext}')
            if os.path.exists(scene_file):
                model_file = scene_file
                break
        
        if not model_file:
            # Look for any model file
            for ext in ['glb', 'gltf', 'obj']:
                files = list(Path(model_folder).glob(f'*.{ext}'))
                if files:
                    model_file = str(files[0])
                    break
        
        if not model_file:
            raise FileNotFoundError(f"No model file found in {model_folder}")
        
        # Generate or use custom prompt
        prompt = custom_prompt if custom_prompt else generate_texture_prompt(model_name)
        
        # Step 1: Generate texture using Flux
        print(f"Generating texture with prompt: {prompt}")
        texture_image = call_flux_api(
            prompt=prompt,
            width=texture_size,
            height=texture_size,
            guidance_scale=3.5,
            num_steps=50
        )
        
        # Step 2: Apply texture to model
        print(f"Applying texture to model: {model_file}")
        result = apply_texture_to_model(
            model_path=model_file,
            texture_image=texture_image,
            output_folder=model_folder
        )
        
        result['success'] = True
        result['message'] = 'Texture generated and applied successfully'
        result['prompt'] = prompt
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {str(e)}',
            'error': str(e)
        }


def check_flux_server_health() -> dict:
    """
    Check if Flux server is running and healthy.
    
    Returns:
        dict: Server status information
    """
    flux_port = int(os.environ.get('FLUX_SERVER_PORT', 8000))
    flux_host = os.environ.get('FLUX_HOST', 'localhost')
    health_url = f"http://{flux_host}:{flux_port}/health"
    
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            'available': True,
            'status': data.get('status', 'unknown'),
            'details': data
        }
    except Exception as e:
        return {
            'available': False,
            'status': 'unavailable',
            'error': str(e)
        }

