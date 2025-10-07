"""
Logic for model_viewer.py - handles texture generation via Flux API
"""

import os
import requests
import trimesh
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def extract_physical_dimensions(mesh: trimesh.Trimesh, model_path: str = None) -> dict:
    """
    Extract physical dimensions and surface properties from a 3D model.
    
    Args:
        mesh: Trimesh object
        model_path: Optional path to model file for additional metadata
    
    Returns:
        dict: Physical dimension analysis
    """
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]  # [width, height, depth]
    
    # Calculate surface area
    surface_area = mesh.area if hasattr(mesh, 'area') else 0
    
    # Calculate volume
    volume = mesh.volume if hasattr(mesh, 'volume') and mesh.is_watertight else 0
    
    # Determine primary axis (longest dimension)
    primary_axis = np.argmax(dimensions)
    axis_names = ['X (width)', 'Y (height)', 'Z (depth)']
    
    # Calculate texture resolution recommendations based on surface area
    # Rule: ~0.5-1.0 pixels per mm² for medical models
    if surface_area > 0:
        recommended_texture_size = min(4096, max(512, int(np.sqrt(surface_area * 0.75))))
        # Round to nearest power of 2
        recommended_texture_size = 2 ** int(np.log2(recommended_texture_size))
    else:
        recommended_texture_size = 1024
    
    return {
        'bounds': {
            'min': bounds[0].tolist(),
            'max': bounds[1].tolist()
        },
        'dimensions': {
            'width': float(dimensions[0]),
            'height': float(dimensions[1]), 
            'depth': float(dimensions[2]),
            'units': 'mm'  # Assuming medical data in mm
        },
        'surface_area': float(surface_area),
        'volume': float(volume),
        'primary_axis': {
            'index': int(primary_axis),
            'name': axis_names[primary_axis],
            'length': float(dimensions[primary_axis])
        },
        'recommended_texture_size': recommended_texture_size,
        'texture_density': float(recommended_texture_size ** 2 / max(surface_area, 1)),  # pixels per mm²
        'aspect_ratios': {
            'xy': float(dimensions[0] / dimensions[1]) if dimensions[1] > 0 else 1.0,
            'xz': float(dimensions[0] / dimensions[2]) if dimensions[2] > 0 else 1.0,
            'yz': float(dimensions[1] / dimensions[2]) if dimensions[2] > 0 else 1.0
        }
    }


def analyze_uv_mapping(mesh: trimesh.Trimesh, uvs: np.ndarray = None) -> dict:
    """
    Analyze UV mapping of a mesh to understand texture layout requirements.
    
    Args:
        mesh: Trimesh object
        uvs: UV coordinates (if None, will extract or generate)
    
    Returns:
        dict: UV analysis information
    """
    if uvs is None:
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv
        else:
            uvs = generate_uv_mapping(mesh)
    
    # UV space statistics
    u_min, u_max = uvs[:, 0].min(), uvs[:, 0].max()
    v_min, v_max = uvs[:, 1].min(), uvs[:, 1].max()
    
    # UV coverage (how much of the 0-1 space is used)
    u_coverage = u_max - u_min
    v_coverage = v_max - v_min
    
    # UV density (vertices per UV area)
    uv_area = u_coverage * v_coverage
    uv_density = len(uvs) / max(uv_area, 0.001)
    
    # Detect UV islands (disconnected regions)
    faces_uvs = uvs[mesh.faces]  # UV coords for each face
    
    analysis = {
        'total_vertices': len(uvs),
        'uv_bounds': {'u': (u_min, u_max), 'v': (v_min, v_max)},
        'uv_coverage': {'u': u_coverage, 'v': v_coverage},
        'uv_area_used': uv_area,
        'uv_density': uv_density,
        'has_seams': u_coverage > 0.9 or v_coverage > 0.9,  # Likely unwrapped with seams
        'mapping_type': 'existing' if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None else 'generated'
    }
    
    return analysis


def create_uv_unwrap_visualization(mesh: trimesh.Trimesh, uvs: np.ndarray = None, 
                                 output_path: str = None, size: int = 1024) -> Image.Image:
    """
    Create a UV unwrap visualization showing how the 3D model maps to 2D texture space.
    This helps understand where different parts of the model appear in the texture.
    
    Args:
        mesh: Trimesh object
        uvs: UV coordinates (if None, will extract or generate)
        output_path: Path to save the visualization image
        size: Size of the output image
    
    Returns:
        PIL Image: UV unwrap visualization
    """
    if uvs is None:
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv
        else:
            uvs = generate_uv_mapping(mesh)
    
    # Create image
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Convert UV coordinates to image space
    uv_img = uvs.copy()
    uv_img[:, 0] = uv_img[:, 0] * (size - 1)
    uv_img[:, 1] = (1 - uv_img[:, 1]) * (size - 1)  # Flip V coordinate
    
    # Draw faces as wireframe
    for face in mesh.faces:
        face_uvs = uv_img[face]
        
        # Draw triangle edges
        for i in range(3):
            start = tuple(face_uvs[i].astype(int))
            end = tuple(face_uvs[(i + 1) % 3].astype(int))
            draw.line([start, end], fill='blue', width=1)
    
    # Draw vertices as dots
    for uv in uv_img:
        x, y = int(uv[0]), int(uv[1])
        draw.ellipse([x-1, y-1, x+1, y+1], fill='red')
    
    # Add grid lines for reference
    grid_spacing = size // 8
    for i in range(0, size, grid_spacing):
        draw.line([(i, 0), (i, size)], fill='lightgray', width=1)
        draw.line([(0, i), (size, i)], fill='lightgray', width=1)
    
    # Add UV space labels
    try:
        font = ImageFont.load_default()
        draw.text((10, 10), "UV Space (0,1 = top-left, 1,0 = bottom-right)", fill='black', font=font)
        draw.text((10, size-30), "Blue lines = mesh edges, Red dots = vertices", fill='black', font=font)
    except:
        pass  # If font fails, continue without text
    
    if output_path:
        img.save(output_path)
    
    return img


def extract_gltf_uv_coordinates(gltf_path: str) -> np.ndarray:
    """
    Extract UV coordinates directly from GLTF file for accurate texture mapping.
    
    Args:
        gltf_path: Path to GLTF file
    
    Returns:
        np.ndarray: UV coordinates (N x 2) or empty array if not found
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
                        if os.path.exists(buffer_path):
                            with open(buffer_path, 'rb') as f:
                                buffer_data = f.read()
                            
                            # Extract UV coordinates
                            byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                            uv_data = buffer_data[byte_offset:byte_offset + buffer_view['byteLength']]
                            
                            # Convert to numpy array (assuming FLOAT32)
                            uv_array = np.frombuffer(uv_data, dtype=np.float32)
                            uv_array = uv_array.reshape(-1, 2)  # Each UV pair has 2 components
                            
                            uvs.extend(uv_array)
        
        return np.array(uvs) if uvs else np.array([])
    except Exception as e:
        print(f"Error extracting UVs from GLTF: {e}")
        return np.array([])


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
    # Load prompts from JSON file
    prompts_data = load_vista3d_prompts()
    prompts_list = prompts_data.get('prompts', [])
    default_template = prompts_data.get('default_template', 
        'hyper photo-realistic human {structure} anatomical structure, medical photography, '
        'anatomically accurate surface texture, natural clinical appearance, high detail, '
        '8K resolution, professional medical illustration')
    
    # Try exact match first (preserve left/right information)
    model_name_lower = model_name.lower()
    model_name_normalized = model_name_lower.replace('_', ' ')
    for prompt_entry in prompts_list:
        entry_name = prompt_entry.get('name', '').lower()
        if entry_name == model_name_lower or entry_name == model_name_normalized:
            return prompt_entry.get('prompt', default_template.format(structure=model_name_lower))
    
    # Try normalized match (remove left/right for generic matches)
    normalized_name = normalize_model_name(model_name)
    for prompt_entry in prompts_list:
        entry_name = prompt_entry.get('name', '').lower()
        entry_normalized = normalize_model_name(entry_name)
        if entry_normalized == normalized_name:
            return prompt_entry.get('prompt', default_template.format(structure=normalized_name))
    
    # Try partial match (for compound names)
    for prompt_entry in prompts_list:
        entry_name = prompt_entry.get('name', '').lower()
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


def generate_uv_mapping(mesh: trimesh.Trimesh, method: str = 'smart') -> np.ndarray:
    """
    Generate UV coordinates for a mesh using various mapping strategies.
    
    Args:
        mesh: Trimesh object
        method: UV mapping method ('spherical', 'cylindrical', 'planar', 'smart')
    
    Returns:
        np.ndarray: UV coordinates (N x 2)
    """
    vertices = mesh.vertices
    
    # Center vertices
    center = vertices.mean(axis=0)
    centered = vertices - center
    
    # Normalize for consistent mapping
    bbox_size = mesh.bounds[1] - mesh.bounds[0]
    max_dim = np.max(bbox_size)
    normalized = centered / (max_dim * 0.5)
    
    if method == 'spherical' or method == 'smart':
        # Spherical mapping - good for organic shapes
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        sphere_normalized = normalized / norms
        
        u = 0.5 + np.arctan2(sphere_normalized[:, 2], sphere_normalized[:, 0]) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(sphere_normalized[:, 1], -1, 1)) / np.pi
        
    elif method == 'cylindrical':
        # Cylindrical mapping - good for elongated objects
        u = 0.5 + np.arctan2(normalized[:, 2], normalized[:, 0]) / (2 * np.pi)
        v = (normalized[:, 1] - normalized[:, 1].min()) / (normalized[:, 1].max() - normalized[:, 1].min())
        
    elif method == 'planar':
        # Planar mapping - simple XZ projection
        u = (normalized[:, 0] - normalized[:, 0].min()) / (normalized[:, 0].max() - normalized[:, 0].min())
        v = (normalized[:, 2] - normalized[:, 2].min()) / (normalized[:, 2].max() - normalized[:, 2].min())
    
    # Ensure UV coordinates are in [0, 1] range
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
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
    
    # Extract UV coordinates with preference for existing GLTF UVs
    uv_coords = None
    
    # First, try to extract UVs from GLTF if it's a GLTF model
    if model_path.endswith('.gltf'):
        uv_coords = extract_gltf_uv_coordinates(model_path)
        if len(uv_coords) > 0:
            print(f"Using {len(uv_coords)} UV coordinates from GLTF file")
    
    # If no GLTF UVs, try trimesh visual UVs
    if uv_coords is None or len(uv_coords) == 0:
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv_coords = mesh.visual.uv
            print(f"Using {len(uv_coords)} UV coordinates from mesh visual")
        else:
            # Generate UV mapping using smart method
            uv_coords = generate_uv_mapping(mesh, method='smart')
            print(f"Generated {len(uv_coords)} UV coordinates using smart mapping")
    
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
    
    # Export as GLTF/GLB (modern format)
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
        'gltf_path': gltf_path if os.path.exists(gltf_path) else None,
        'glb_path': glb_path if os.path.exists(glb_path) else None,
        'texture_path': texture_path,
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces),
        'uv_coords': len(uv_coords)
    }


def generate_physically_aware_texture_prompt(model_name: str, physical_dims: dict, 
                                           uv_analysis: dict, base_prompt: str = None) -> str:
    """
    Generate a texture prompt that considers both UV mapping and physical dimensions.
    
    Args:
        model_name: Name of the model (e.g., 'colon')
        physical_dims: Physical dimension analysis from extract_physical_dimensions()
        uv_analysis: UV mapping analysis
        base_prompt: Optional base prompt to enhance
    
    Returns:
        str: Enhanced prompt for texture generation
    """
    if base_prompt:
        enhanced_prompt = base_prompt
    else:
        # Anatomical texture prompts based on model name
        anatomical_prompts = {
            'colon': 'realistic medical colon tissue texture, smooth pink mucosa surface, subtle anatomical folds, medical photography quality',
            'heart': 'realistic cardiac muscle texture, deep red myocardium, subtle fiber patterns, medical grade detail',
            'liver': 'realistic hepatic tissue texture, reddish-brown parenchyma, smooth lobular surface, medical photography',
            'aorta': 'realistic arterial wall texture, smooth endothelium, slight translucency, medical vessel appearance',
            'kidney': 'realistic renal tissue texture, smooth cortex surface, reddish-brown nephron patterns',
            'lung': 'realistic pulmonary tissue texture, spongy alveolar surface, pinkish-gray coloration'
        }
        enhanced_prompt = anatomical_prompts.get(model_name.lower(), 
                                               f'realistic medical {model_name} tissue texture, anatomically accurate surface detail')
    
    # Add physical scale context to the prompt
    dims = physical_dims['dimensions']
    surface_area = physical_dims['surface_area']
    texture_size = physical_dims['recommended_texture_size']
    
    # Scale descriptors based on physical size
    if dims['width'] > 200 or dims['height'] > 200 or dims['depth'] > 200:
        scale_desc = "large organ scale"
    elif dims['width'] < 50 and dims['height'] < 50 and dims['depth'] < 50:
        scale_desc = "fine anatomical detail scale"
    else:
        scale_desc = "standard organ scale"
    
    # Surface area considerations
    if surface_area > 10000:  # Large surface area
        detail_level = "highly detailed surface patterns, complex anatomical microstructures"
    elif surface_area > 5000:  # Medium surface area
        detail_level = "detailed surface texture, visible anatomical features"
    else:  # Small surface area
        detail_level = "smooth surface texture, subtle anatomical detail"
    
    # UV mapping considerations
    if uv_analysis.get('has_seams', False):
        mapping_hint = "seamless texture pattern that wraps naturally around complex 3D anatomy"
    else:
        mapping_hint = "uniform texture distribution across curved anatomical surfaces"
    
    # Primary axis orientation for texture flow
    primary_axis = physical_dims['primary_axis']['name']
    if 'width' in primary_axis.lower():
        orientation_hint = "horizontally oriented anatomical patterns"
    elif 'height' in primary_axis.lower():
        orientation_hint = "vertically flowing tissue structures"
    else:
        orientation_hint = "depth-wise anatomical layering"
    
    # Combine all elements
    enhanced_prompt = f"{enhanced_prompt}, {scale_desc}, {detail_level}, {mapping_hint}, {orientation_hint}"
    
    # Add technical specifications for texture quality
    enhanced_prompt += f", {texture_size}x{texture_size} resolution, medical photography lighting, accurate anatomical coloration"
    
    # Add material properties hint
    enhanced_prompt += ", soft tissue material properties, subtle subsurface scattering, realistic medical specimen appearance"
    
    return enhanced_prompt


def generate_uv_aware_texture_prompt(model_name: str, uv_analysis: dict, 
                                    base_prompt: str = None) -> str:
    """
    Generate a UV-aware texture prompt that accounts for the model's UV layout.
    
    Args:
        model_name: Name of the model
        uv_analysis: UV analysis result from analyze_uv_mapping()
        base_prompt: Base texture prompt (if None, auto-generate)
    
    Returns:
        str: Enhanced prompt for UV-aware texture generation
    """
    if base_prompt is None:
        base_prompt = generate_texture_prompt(model_name)
    
    # Add UV-specific guidance based on analysis
    uv_guidance = []
    
    if uv_analysis.get('has_seams', False):
        uv_guidance.append("seamless texture atlas")
        uv_guidance.append("no visible seams or edges")
    
    if uv_analysis.get('uv_density', 0) > 1000:  # High density
        uv_guidance.append("high detail texture")
        uv_guidance.append("fine surface details")
    
    # Coverage-based guidance
    u_coverage = uv_analysis.get('uv_coverage', {}).get('u', 1.0)
    v_coverage = uv_analysis.get('uv_coverage', {}).get('v', 1.0)
    
    if u_coverage < 0.8 or v_coverage < 0.8:
        uv_guidance.append("compact texture layout")
    else:
        uv_guidance.append("full UV space utilization")
    
    # Model-specific enhancements
    model_lower = model_name.lower()
    if any(term in model_lower for term in ['skull', 'bone', 'nyctalus', 'vertebra']):
        uv_guidance.extend([
            "bone texture with natural surface variations",
            "anatomical surface detail with pores and texture",
            "natural weathering and surface imperfections"
        ])
    elif any(term in model_lower for term in ['organ', 'tissue', 'muscle']):
        uv_guidance.extend([
            "organic tissue surface",
            "natural biological texture"
        ])
    
    # Combine prompts
    if uv_guidance:
        enhanced_prompt = f"{base_prompt}, {', '.join(uv_guidance)}"
    else:
        enhanced_prompt = base_prompt
    
    # Add UV layout instructions
    enhanced_prompt += ", texture atlas format, consistent lighting across UV islands"
    
    return enhanced_prompt


def generate_physically_scaled_texture(model_folder: str, model_name: str,
                                      custom_prompt: str = None,
                                      texture_size: int = None,  # Now auto-calculated if None
                                      generate_uv_preview: bool = True,
                                      use_physical_scaling: bool = True) -> dict:
    """
    Generate physically-accurate texture for a model using physical dimensions and UV analysis.
    
    Args:
        model_folder: Path to the model folder
        model_name: Name of the model (e.g., 'colon')
        custom_prompt: Optional custom prompt (if None, auto-generate physically-aware prompt)
        texture_size: Size of the texture (if None, auto-calculate from physical dimensions)
        generate_uv_preview: Whether to generate UV unwrap visualization
        use_physical_scaling: Whether to use physical dimensions for scaling
    
    Returns:
        dict: Result information including paths, status, and physical analysis
    """
    try:
        # Find the model file
        model_file = None
        for ext in ['gltf', 'glb', 'obj']:  # Prioritize GLTF for better UV support
            scene_file = os.path.join(model_folder, f'scene.{ext}')
            if os.path.exists(scene_file):
                model_file = scene_file
                break
        
        if not model_file:
            # Look for any model file
            for ext in ['gltf', 'glb', 'obj']:
                files = list(Path(model_folder).glob(f'*.{ext}'))
                if files:
                    model_file = str(files[0])
                    break
        
        if not model_file:
            raise FileNotFoundError(f"No model file found in {model_folder}")
        
        # Load mesh for analysis
        mesh = trimesh.load(model_file)
        if hasattr(mesh, 'geometry'):
            meshes = list(mesh.geometry.values())
            if len(meshes) == 0:
                raise ValueError("No meshes found in scene")
            mesh = trimesh.util.concatenate(meshes)

        # Step 1: Extract physical dimensions
        if use_physical_scaling:
            physical_dims = extract_physical_dimensions(mesh, model_file)
            print(f"Physical Analysis:")
            print(f"  Dimensions: {physical_dims['dimensions']['width']:.1f} x {physical_dims['dimensions']['height']:.1f} x {physical_dims['dimensions']['depth']:.1f} {physical_dims['dimensions']['units']}")
            print(f"  Surface Area: {physical_dims['surface_area']:.1f} mm²")
            print(f"  Recommended Texture Size: {physical_dims['recommended_texture_size']}x{physical_dims['recommended_texture_size']}")
            
            # Use recommended texture size if not specified
            if texture_size is None:
                texture_size = physical_dims['recommended_texture_size']
        else:
            physical_dims = None
            if texture_size is None:
                texture_size = 1024
        
        # Step 2: Extract UV coordinates for analysis
        uv_coords = None
        if model_file.endswith('.gltf'):
            uv_coords = extract_gltf_uv_coordinates(model_file)
        
        if uv_coords is None or len(uv_coords) == 0:
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                uv_coords = mesh.visual.uv
            else:
                uv_coords = generate_uv_mapping(mesh, method='smart')
        
        # Step 3: Analyze UV mapping
        uv_analysis = analyze_uv_mapping(mesh, uv_coords)
        print(f"UV Analysis: {uv_analysis}")
        
        # Step 4: Generate physically-aware prompt
        if custom_prompt:
            prompt = custom_prompt
        elif use_physical_scaling and physical_dims:
            prompt = generate_physically_aware_texture_prompt(model_name, physical_dims, uv_analysis)
        else:
            prompt = generate_uv_aware_texture_prompt(model_name, uv_analysis)
        
        print(f"Generated prompt: {prompt}")
        
        # Step 5: Generate texture using Flux with optimal size
        print(f"Generating {texture_size}x{texture_size} texture with Flux API...")
        texture_image = call_flux_api(
            prompt=prompt,
            width=texture_size,
            height=texture_size,
            guidance_scale=3.5,
            num_steps=50
        )
        
        # Step 6: Apply texture to model
        print(f"Applying texture to model: {model_file}")
        result = apply_texture_to_model(
            model_path=model_file,
            texture_image=texture_image,
            output_folder=model_folder
        )
        
        # Add physical analysis to results
        result['success'] = True
        result['message'] = 'Physically-scaled texture generated and applied successfully'
        result['prompt'] = prompt
        result['texture_size'] = texture_size
        result['physical_dimensions'] = physical_dims
        result['uv_analysis'] = uv_analysis
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {str(e)}',
            'error': str(e)
        }


def generate_flux_uv_texture(model_folder: str, model_name: str, 
                           custom_prompt: str = None, texture_size: int = 1024) -> dict:
    """
    Generate hyper-realistic texture using Flux AI with UV coordinate guidance.
    This creates textures that map exactly to the model's UV layout.
    """
    try:
        print(f"Starting Flux UV texture generation for {model_name}...")
        
        # Import UV extraction functions
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        
        from generate_flux_uv_textures import (
            extract_gltf_uv_coordinates,
            create_uv_layout_visualization,
            generate_texture_with_flux,
            apply_texture_to_uv_layout,
            generate_organ_specific_prompt
        )
        
        # Find GLTF file
        gltf_path = os.path.join(model_folder, "scene.gltf")
        if not os.path.exists(gltf_path):
            return {
                'success': False,
                'message': f"No scene.gltf found in {model_folder}"
            }
        
        # Extract UV coordinates
        print("Extracting UV coordinates...")
        uvs = extract_gltf_uv_coordinates(gltf_path)
        print(f"Found {len(uvs)} UV coordinates")
        
        # Create UV layout visualization
        print("Creating UV layout visualization...")
        uv_guide = create_uv_layout_visualization(uvs, texture_size)
        
        # Generate organ-specific prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = generate_organ_specific_prompt(model_name)
        
        print(f"Using prompt: {prompt[:100]}...")
        
        # Generate texture with Flux
        print("Generating texture with Flux AI...")
        flux_texture = generate_texture_with_flux(
            prompt=prompt,
            uv_guide_image=uv_guide,
            size=texture_size,
            guidance_scale=3.5,
            num_steps=50
        )
        
        if flux_texture is None:
            return {
                'success': False,
                'message': "Failed to generate texture with Flux AI"
            }
        
        # Create UV mask and apply texture
        print("Applying texture to UV layout...")
        from generate_flux_uv_textures import create_uv_mask_from_coordinates
        uv_mask = create_uv_mask_from_coordinates(uvs, texture_size)
        final_texture = apply_texture_to_uv_layout(flux_texture, uv_mask, model_name, texture_size)
        
        # Save texture
        texture_path = os.path.join(model_folder, "textures", "diffuse.png")
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        final_texture.save(texture_path)
        
        # Save UV guide for debugging
        uv_guide_path = os.path.join(model_folder, "textures", "uv_guide_flux.png")
        uv_guide.save(uv_guide_path)
        
        # Load mesh for stats
        try:
            import trimesh
            scene_or_mesh = trimesh.load(gltf_path)
            if hasattr(scene_or_mesh, 'geometry'):
                vertices = np.vstack([geom.vertices for geom in scene_or_mesh.geometry.values()])
                faces = np.vstack([geom.faces for geom in scene_or_mesh.geometry.values()])
            else:
                vertices = scene_or_mesh.vertices
                faces = scene_or_mesh.faces
        except:
            vertices = np.array([])
            faces = np.array([])
        
        return {
            'success': True,
            'message': f"Hyper-realistic Flux texture generated successfully for {model_name}",
            'texture_path': texture_path,
            'uv_guide_path': uv_guide_path,
            'vertices': len(vertices),
            'faces': len(faces),
            'uv_coords': len(uvs),
            'prompt': prompt
        }
        
    except Exception as e:
        print(f"Error in Flux UV texture generation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f"Flux UV texture generation failed: {str(e)}",
            'error': str(e)
        }


def generate_photorealistic_texture(model_folder: str, model_name: str,
                                    custom_prompt: str = None,
                                    texture_size: int = 1024,
                                    generate_uv_preview: bool = True) -> dict:
    """
    Backward compatibility wrapper for the original texture generation function.
    """
    return generate_physically_scaled_texture(
        model_folder=model_folder,
        model_name=model_name,
        custom_prompt=custom_prompt,
        texture_size=texture_size,
        generate_uv_preview=generate_uv_preview,
        use_physical_scaling=False  # Use old behavior by default
    )


def check_flux_server_health() -> dict:
    """
    Check if Flux server is running and healthy.
    
    Returns:
        dict: Server status information
    """
    # Check for remote server configuration
    flux_url = os.environ.get('FLUX_SERVER_URL')
    if flux_url:
        health_url = f"{flux_url.rstrip('/')}/health"
    else:
        # Fallback to host:port configuration
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

