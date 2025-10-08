#!/usr/bin/env python3
"""
FLUX-based UV Texture Generator for Medical Organ Models
Uses UV masks and Vista3D prompts to generate anatomically accurate textures
"""

import os
import sys
import json
import base64
import argparse
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw
import numpy as np
from dotenv import load_dotenv


def load_vista3d_prompts(prompts_file: str = "vista3d_prompts.json") -> Dict[str, str]:
    """
    Load prompts from vista3d_prompts.json file.
    
    Args:
        prompts_file: Path to the prompts JSON file
        
    Returns:
        Dictionary mapping structure names to prompts
    """
    # Look for prompts file in frontend/conf directory
    script_dir = Path(__file__).parent
    prompts_path = script_dir.parent / "conf" / prompts_file
    
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(prompts_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from name to prompt
    prompts = {}
    for item in data.get('prompts', []):
        name = item.get('name', '').lower().strip()
        prompt = item.get('prompt', '')
        if name and prompt:
            prompts[name] = prompt
    
    # Add default template
    prompts['_default'] = data.get('default_template', 
        "hyper photo-realistic human {structure} anatomical structure, medical photography, "
        "anatomically accurate surface texture, natural clinical appearance, high detail, "
        "8K resolution, professional medical illustration")
    
    return prompts


def get_prompt_for_organ(organ_name: str, prompts: Dict[str, str]) -> str:
    """
    Get the appropriate prompt for an organ, with fallback to default.
    
    Args:
        organ_name: Name of the organ
        prompts: Dictionary of prompts
        
    Returns:
        Prompt string for the organ
    """
    organ_key = organ_name.lower().strip()
    
    # Try exact match
    if organ_key in prompts:
        return prompts[organ_key]
    
    # Try with underscores replaced by spaces
    organ_key_spaces = organ_key.replace('_', ' ')
    if organ_key_spaces in prompts:
        return prompts[organ_key_spaces]
    
    # Use default template
    default_template = prompts.get('_default', '')
    if default_template:
        return default_template.replace('{structure}', organ_name.replace('_', ' '))
    
    # Final fallback
    return f"hyper photo-realistic human {organ_name.replace('_', ' ')} anatomical structure, medical photography, 8K resolution"


def extract_gltf_uv_coordinates(gltf_path: str) -> np.ndarray:
    """
    Extract UV coordinates directly from GLTF file.
    
    Args:
        gltf_path: Path to GLTF file
        
    Returns:
        UV coordinates array (N x 2) or empty array if not found
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
                            print(f"⚠ Warning: Buffer file not found: {buffer_path}")
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
        print(f"✗ Error extracting UVs from GLTF: {e}")
        return np.array([])


def create_enhanced_uv_mask(uv_mask_path: str, size: int = 1024) -> Image.Image:
    """
    Load and enhance the UV mask for better texture generation control.
    
    Args:
        uv_mask_path: Path to the UV mask PNG file
        size: Target size for the mask
        
    Returns:
        Enhanced UV mask image
    """
    if not os.path.exists(uv_mask_path):
        print(f"⚠ Warning: UV mask not found: {uv_mask_path}")
        # Create a simple filled mask as fallback
        return Image.new('RGB', (size, size), (255, 255, 255))
    
    # Load the UV mask
    mask = Image.open(uv_mask_path)
    
    # Convert to RGB if needed
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')
    
    # Resize if needed
    if mask.size != (size, size):
        mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    
    return mask


def generate_texture_with_flux(
    prompt: str,
    uv_mask: Image.Image,
    flux_server: str,
    size: int = 1024,
    guidance_scale: float = 3.5,
    num_steps: int = 50,
    seed: Optional[int] = None,
    use_uv_guidance: bool = True
) -> Optional[Image.Image]:
    """
    Generate texture using FLUX server with UV mask guidance.
    
    Args:
        prompt: Text prompt for generation
        uv_mask: UV mask image
        flux_server: FLUX server address (e.g., 'localhost:8000')
        size: Texture size
        guidance_scale: Guidance scale for generation
        num_steps: Number of inference steps
        seed: Random seed for reproducibility
        use_uv_guidance: Whether to use UV-guided generation
        
    Returns:
        Generated texture image or None if failed
    """
    # Prepare the request
    server_url = f"http://{flux_server}"
    
    # Check if server is available
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"✗ FLUX server not healthy: {health_response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Cannot connect to FLUX server at {server_url}: {e}")
        print(f"  Make sure the server is running on {flux_server}")
        return None
    
    print(f"✓ Connected to FLUX server at {server_url}")
    
    # Choose endpoint based on UV guidance
    if use_uv_guidance:
        endpoint = f"{server_url}/generate_with_control"
        
        # Encode UV mask as base64
        import io
        buffered = io.BytesIO()
        uv_mask.save(buffered, format="PNG")
        mask_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        payload = {
            "prompt": prompt,
            "control_image": mask_b64,
            "control_type": "uv_layout",
            "height": size,
            "width": size,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "seed": seed,
            "return_base64": False  # Get PNG directly
        }
    else:
        endpoint = f"{server_url}/generate"
        payload = {
            "prompt": prompt,
            "height": size,
            "width": size,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "max_sequence_length": 512,
            "seed": seed,
            "return_base64": False
        }
    
    print(f"🎨 Generating texture...")
    print(f"   Prompt: {prompt[:100]}...")
    print(f"   Size: {size}x{size}")
    print(f"   Steps: {num_steps}")
    print(f"   Guidance: {guidance_scale}")
    if seed is not None:
        print(f"   Seed: {seed}")
    
    try:
        response = requests.post(endpoint, json=payload, timeout=300)  # 5 min timeout
        
        if response.status_code == 200:
            # Load image from response
            from io import BytesIO
            image = Image.open(BytesIO(response.content))
            print(f"✓ Texture generated successfully!")
            return image
        else:
            print(f"✗ Generation failed: {response.status_code}")
            print(f"  {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        return None


def apply_uv_mask_to_texture(texture: Image.Image, uv_mask: Image.Image) -> Image.Image:
    """
    Apply UV mask to texture to ensure only UV-mapped areas have content.
    
    Args:
        texture: Generated texture image
        uv_mask: UV mask image
        
    Returns:
        Masked texture image
    """
    # Convert to arrays
    texture_array = np.array(texture).astype(np.float32)
    mask_array = np.array(uv_mask.convert('L')).astype(np.float32) / 255.0
    
    # Apply mask to each channel
    for i in range(3):
        texture_array[:, :, i] = texture_array[:, :, i] * mask_array
    
    # Convert back to image
    return Image.fromarray(texture_array.astype(np.uint8))


def generate_organ_texture(
    organ_name: str,
    models_dir: str = "output/models",
    size: int = 1024,
    guidance_scale: float = 3.5,
    num_steps: int = 50,
    seed: Optional[int] = None,
    output_name: str = "flux_texture.png",
    apply_mask: bool = True,
    use_uv_guidance: bool = True,
    flux_server: Optional[str] = None
) -> bool:
    """
    Generate texture for a specific organ model.
    
    Args:
        organ_name: Name of the organ (matches folder in models_dir)
        models_dir: Directory containing model folders
        size: Texture size
        guidance_scale: Guidance scale for generation
        num_steps: Number of inference steps
        seed: Random seed
        output_name: Output filename for the texture
        apply_mask: Whether to apply UV mask to constrain generation
        use_uv_guidance: Whether to use UV-guided generation
        flux_server: FLUX server address (overrides .env)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Generating Texture for: {organ_name}")
    print(f"{'='*60}\n")
    
    # Load environment variables
    script_dir = Path(__file__).parent
    env_path = script_dir.parent / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded configuration from {env_path}")
    
    # Get FLUX server address
    if flux_server is None:
        flux_server = os.getenv('FLUX_SERVER', 'localhost:8000')
    
    print(f"  FLUX Server: {flux_server}")
    
    # Find model directory
    project_root = script_dir.parent.parent
    model_dir = project_root / models_dir / organ_name
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False
    
    print(f"✓ Model directory: {model_dir}")
    
    # Find GLTF file
    gltf_file = model_dir / "scene.gltf"
    if not gltf_file.exists():
        print(f"✗ GLTF file not found: {gltf_file}")
        return False
    
    print(f"✓ GLTF file: {gltf_file}")
    
    # Extract UV coordinates (for validation)
    print(f"\n📐 Extracting UV coordinates...")
    uvs = extract_gltf_uv_coordinates(str(gltf_file))
    if len(uvs) > 0:
        print(f"✓ Found {len(uvs)} UV coordinates")
        uv_min = uvs.min(axis=0)
        uv_max = uvs.max(axis=0)
        print(f"  UV range: [{uv_min[0]:.3f}, {uv_min[1]:.3f}] to [{uv_max[0]:.3f}, {uv_max[1]:.3f}]")
    else:
        print(f"⚠ No UV coordinates found - texture may not map correctly")
    
    # Load UV mask
    print(f"\n🎭 Loading UV mask...")
    uv_mask_file = model_dir / "uv_mask.png"
    uv_mask = create_enhanced_uv_mask(str(uv_mask_file), size)
    print(f"✓ UV mask loaded: {uv_mask.size}")
    
    # Load prompts
    print(f"\n📝 Loading prompts...")
    try:
        prompts = load_vista3d_prompts()
        print(f"✓ Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"✗ Failed to load prompts: {e}")
        return False
    
    # Get prompt for this organ
    prompt = get_prompt_for_organ(organ_name, prompts)
    print(f"✓ Prompt: {prompt[:80]}...")
    
    # Generate texture
    print(f"\n🚀 Starting texture generation...")
    texture = generate_texture_with_flux(
        prompt=prompt,
        uv_mask=uv_mask,
        flux_server=flux_server,
        size=size,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        seed=seed,
        use_uv_guidance=use_uv_guidance
    )
    
    if texture is None:
        print(f"✗ Texture generation failed")
        return False
    
    # Apply UV mask if requested
    if apply_mask:
        print(f"\n🎨 Applying UV mask to texture...")
        texture = apply_uv_mask_to_texture(texture, uv_mask)
        print(f"✓ Mask applied")
    
    # Save texture
    textures_dir = model_dir / "textures"
    textures_dir.mkdir(exist_ok=True)
    
    output_path = textures_dir / output_name
    texture.save(output_path)
    print(f"\n✓ Texture saved to: {output_path}")
    
    # Also save a copy as diffuse.png (default texture name)
    if output_name != "diffuse.png":
        diffuse_path = textures_dir / "diffuse.png"
        texture.save(diffuse_path)
        print(f"✓ Also saved as: {diffuse_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS: Texture generated for {organ_name}")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate anatomically accurate textures for medical organ models using FLUX.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate texture for colon model
  python generate_flux_texture.py --organ colon
  
  # High quality generation with custom settings
  python generate_flux_texture.py --organ heart --size 2048 --steps 100 --guidance 4.0
  
  # With specific seed for reproducibility
  python generate_flux_texture.py --organ liver --seed 42
  
  # Without UV guidance (basic generation)
  python generate_flux_texture.py --organ brain --no-uv-guidance
  
  # Custom FLUX server
  python generate_flux_texture.py --organ colon --server 192.168.1.100:8000
        """
    )
    
    parser.add_argument(
        '--organ',
        type=str,
        required=True,
        help='Organ name (must match folder in models directory)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='output/models',
        help='Directory containing model folders (default: output/models)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=1024,
        choices=[512, 1024, 2048],
        help='Texture size in pixels (default: 1024)'
    )
    
    parser.add_argument(
        '--guidance',
        type=float,
        default=3.5,
        help='Guidance scale for generation (default: 3.5, higher = more prompt adherence)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of inference steps (default: 50, higher = better quality but slower)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='flux_texture.png',
        help='Output filename for the texture (default: flux_texture.png)'
    )
    
    parser.add_argument(
        '--no-mask',
        action='store_true',
        help='Do not apply UV mask to final texture'
    )
    
    parser.add_argument(
        '--no-uv-guidance',
        action='store_true',
        help='Disable UV-guided generation (use basic text-to-image)'
    )
    
    parser.add_argument(
        '--server',
        type=str,
        default=None,
        help='FLUX server address (default: from .env or localhost:8000)'
    )
    
    args = parser.parse_args()
    
    # Generate texture
    success = generate_organ_texture(
        organ_name=args.organ,
        models_dir=args.models_dir,
        size=args.size,
        guidance_scale=args.guidance,
        num_steps=args.steps,
        seed=args.seed,
        output_name=args.output,
        apply_mask=not args.no_mask,
        use_uv_guidance=not args.no_uv_guidance,
        flux_server=args.server
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

