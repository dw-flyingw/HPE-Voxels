#!/usr/bin/env python3
"""
Stable Diffusion-based UV Texture Generator for Medical Organ Models
Uses UV masks and ControlNet to generate anatomically accurate textures locally.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw
import numpy as np
from dotenv import load_dotenv

# Try to import PyTorch and Diffusers
try:
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from diffusers.utils import load_image
    import cv2
except ImportError:
    print("✗ Error: PyTorch or Diffusers not installed.")
    print("  Please install the required dependencies from frontend/requirements.txt")
    sys.exit(1)


def load_vista3d_prompts(prompts_file: str = "vista3d_prompts.json") -> Dict[str, str]:
    """
    Load prompts from vista3d_prompts.json file.
    """
    script_dir = Path(__file__).parent
    prompts_path = script_dir.parent / "conf" / prompts_file
    
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(prompts_path, 'r') as f:
        data = json.load(f)
    
    prompts = {}
    for item in data.get('prompts', []):
        name = item.get('name', '').lower().strip()
        prompt = item.get('prompt', '')
        if name and prompt:
            prompts[name] = prompt
    
    prompts['_default'] = data.get('default_template', 
        "hyper photo-realistic human {structure} anatomical structure, medical photography, "
        "anatomically accurate surface texture, natural clinical appearance, high detail, "
        "8K resolution, professional medical illustration")
    
    return prompts


def get_prompt_for_organ(organ_name: str, prompts: Dict[str, str]) -> str:
    """
    Get the appropriate prompt for an organ, with fallback to default.
    """
    organ_key = organ_name.lower().strip()
    
    if organ_key in prompts:
        return prompts[organ_key]
    
    organ_key_spaces = organ_key.replace('_', ' ')
    if organ_key_spaces in prompts:
        return prompts[organ_key_spaces]
    
    default_template = prompts.get('_default', '')
    if default_template:
        return default_template.replace('{structure}', organ_name.replace('_', ' '))
    
    return f"hyper photo-realistic human {organ_name.replace('_', ' ')} anatomical structure, medical photography, 8K resolution"


def create_enhanced_uv_mask(uv_mask_path: str, size: int = 1024) -> Image.Image:
    """
    Load and enhance the UV mask for better texture generation control.
    """
    if not os.path.exists(uv_mask_path):
        print(f"⚠ Warning: UV mask not found: {uv_mask_path}")
        return Image.new('RGB', (size, size), (255, 255, 255))
    
    mask = Image.open(uv_mask_path).convert('RGB')
    
    if mask.size != (size, size):
        mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    
    return mask


def generate_texture_with_sd(
    prompt: str,
    uv_mask: Image.Image,
    size: int = 1024,
    guidance_scale: float = 7.5,
    num_steps: int = 20,
    seed: Optional[int] = None,
    model_id: str = "dreamlike-art/dreamlike-photoreal-2.0",
    controlnet_id: str = "lllyasviel/sd-controlnet-canny"
) -> Optional[Image.Image]:
    """
    Generate texture using a local Stable Diffusion ControlNet pipeline.
    """
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("✗ Error: No CUDA-enabled GPU found. This script requires a GPU.")
        return None
    
    print("✓ CUDA GPU detected.")
    
    # --- Prepare ControlNet Input ---
    print("🎨 Preparing control image from UV mask...")
    control_image_np = np.array(uv_mask)
    control_image_np = cv2.Canny(control_image_np, 100, 200)
    control_image_np = control_image_np[:, :, None]
    control_image_np = np.concatenate([control_image_np, control_image_np, control_image_np], axis=2)
    control_image = Image.fromarray(control_image_np)
    print("✓ Control image prepared.")

    # --- Load Models ---
    print(f"🚀 Loading models...")
    print(f"   ControlNet: {controlnet_id}")
    print(f"   Base Model: {model_id}")
    
    try:
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None # Disabling for performance and to avoid false positives
        )
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("  This may be a network issue or the model ID is incorrect.")
        print("  Try running `huggingface-cli login` if it's a private model.")
        return None

    # --- Configure Pipeline ---
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload() # Saves VRAM

    # Optional: Enable xformers for memory-efficient attention
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers memory-efficient attention enabled.")
    except Exception:
        print("⚠ xformers is not available. For faster performance, consider installing it.")

    
    print("✓ Models loaded and configured.")

    # --- Generate Texture ---
    generator = torch.manual_seed(seed) if seed is not None else torch.Generator()
    
    print(f"🎨 Generating texture...")
    print(f"   Prompt: {prompt[:100]}...")
    print(f"   Size: {size}x{size}")
    print(f"   Steps: {num_steps}")
    print(f"   Guidance: {guidance_scale}")
    if seed is not None:
        print(f"   Seed: {seed}")

    try:
        output = pipe(
            prompt,
            num_inference_steps=num_steps,
            generator=generator,
            image=control_image,
            height=size,
            width=size,
            guidance_scale=guidance_scale,
        )
        texture = output.images[0]
        print("✓ Texture generated successfully!")
        return texture
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        return None


def apply_uv_mask_to_texture(texture: Image.Image, uv_mask: Image.Image) -> Image.Image:
    """
    Apply UV mask to texture to ensure only UV-mapped areas have content.
    """
    texture_array = np.array(texture).astype(np.float32)
    mask_array = np.array(uv_mask.convert('L')).astype(np.float32) / 255.0
    
    for i in range(3):
        texture_array[:, :, i] = texture_array[:, :, i] * mask_array
    
    return Image.fromarray(texture_array.astype(np.uint8))


def generate_organ_texture(
    organ_name: str,
    models_dir: str = "output/models",
    size: int = 1024,
    guidance_scale: float = 7.5,
    num_steps: int = 30,
    seed: Optional[int] = None,
    output_name: str = "sd_texture.png",
    apply_mask: bool = True
) -> bool:
    """
    Generate texture for a specific organ model.
    """
    print(f"\n{'='*60}")
    print(f"Generating Texture for: {organ_name}")
    print(f"{ '='*60}\n")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    model_dir = project_root / models_dir / organ_name
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False
    
    print(f"✓ Model directory: {model_dir}")
    
    uv_mask_file = model_dir / "uv_mask.png"
    uv_mask = create_enhanced_uv_mask(str(uv_mask_file), size)
    print(f"✓ UV mask loaded: {uv_mask.size}")
    
    print(f"\n📝 Loading prompts...")
    try:
        prompts = load_vista3d_prompts()
        print(f"✓ Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"✗ Failed to load prompts: {e}")
        return False
    
    prompt = get_prompt_for_organ(organ_name, prompts)
    print(f"✓ Prompt: {prompt[:80]}...")
    
    print(f"\n🚀 Starting texture generation...")
    texture = generate_texture_with_sd(
        prompt=prompt,
        uv_mask=uv_mask,
        size=size,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        seed=seed,
    )
    
    if texture is None:
        print(f"✗ Texture generation failed")
        return False
    
    if apply_mask:
        print(f"\n🎨 Applying UV mask to texture...")
        texture = apply_uv_mask_to_texture(texture, uv_mask)
        print(f"✓ Mask applied")
    
    textures_dir = model_dir / "textures"
    textures_dir.mkdir(exist_ok=True)
    
    output_path = textures_dir / output_name
    texture.save(output_path)
    print(f"\n✓ Texture saved to: {output_path}")
    
    if output_name != "diffuse.png":
        diffuse_path = textures_dir / "diffuse.png"
        texture.save(diffuse_path)
        print(f"✓ Also saved as: {diffuse_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS: Texture generated for {organ_name}")
    print(f"{ '='*60}\n")
    
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate anatomically accurate textures for medical models using Stable Diffusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--organ', type=str, required=True, help='Organ name (must match folder in models directory)')
    parser.add_argument('--models-dir', type=str, default='output/models', help='Directory containing model folders')
    parser.add_argument('--size', type=int, default=1024, choices=[512, 768, 1024], help='Texture size in pixels')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale for generation')
    parser.add_argument('--steps', type=int, default=30, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='sd_texture.png', help='Output filename for the texture')
    parser.add_argument('--no-mask', action='store_true', help='Do not apply UV mask to final texture')
    
    args = parser.parse_args()
    
    success = generate_organ_texture(
        organ_name=args.organ,
        models_dir=args.models_dir,
        size=args.size,
        guidance_scale=args.guidance,
        num_steps=args.steps,
        seed=args.seed,
        output_name=args.output,
        apply_mask=not args.no_mask
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()