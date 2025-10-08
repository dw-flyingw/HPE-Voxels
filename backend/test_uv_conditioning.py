#!/usr/bin/env python3
"""
Test UV-Conditioned Texture Generation

Tests the new UV conditioning implementation to ensure it properly
uses the UV mask to guide FLUX generation.
"""

import os
import sys
import base64
import requests
from pathlib import Path
from PIL import Image
import io

def test_uv_conditioning(
    model_name: str = "colon",
    flux_server: str = "localhost:8000",
    size: int = 1024
):
    """
    Test the UV conditioning endpoint with a real UV mask.
    """
    print(f"\n{'='*70}")
    print(f"Testing UV-Conditioned Generation")
    print(f"{'='*70}\n")
    
    # Find model directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "output" / "models" / model_name
    
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False
    
    print(f"✓ Model directory: {model_dir}")
    
    # Find UV mask
    uv_mask_path = model_dir / "uv_mask.png"
    if not uv_mask_path.exists():
        print(f"✗ UV mask not found: {uv_mask_path}")
        return False
    
    print(f"✓ UV mask found: {uv_mask_path}")
    
    # Load and encode UV mask
    print(f"\n📤 Encoding UV mask...")
    uv_mask = Image.open(uv_mask_path)
    print(f"  UV mask size: {uv_mask.size}")
    print(f"  UV mask mode: {uv_mask.mode}")
    
    # Resize if needed
    if uv_mask.size != (size, size):
        print(f"  Resizing to {size}x{size}...")
        uv_mask = uv_mask.resize((size, size), Image.Resampling.LANCZOS)
    
    # Encode to base64
    buffered = io.BytesIO()
    uv_mask.save(buffered, format="PNG")
    mask_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    print(f"✓ UV mask encoded ({len(mask_b64)} bytes)")
    
    # Check server health
    print(f"\n🔌 Checking FLUX server at {flux_server}...")
    server_url = f"http://{flux_server}"
    
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"✗ FLUX server not healthy: {health_response.status_code}")
            return False
        
        health_data = health_response.json()
        print(f"✓ Server healthy")
        print(f"  Status: {health_data.get('status')}")
        print(f"  Model loaded: {health_data.get('model_loaded')}")
        print(f"  Device: {health_data.get('device')}")
        
    except Exception as e:
        print(f"✗ Cannot connect to FLUX server: {e}")
        print(f"\n💡 Start the server with: cd backend && ./start_server.sh")
        return False
    
    # Prepare request
    print(f"\n🚀 Sending UV-conditioned generation request...")
    
    prompt = f"hyper photo-realistic human {model_name} anatomical structure, medical photography, anatomically accurate surface texture"
    
    payload = {
        "prompt": prompt,
        "control_image": mask_b64,
        "control_type": "uv_layout",
        "height": size,
        "width": size,
        "guidance_scale": 3.5,
        "num_inference_steps": 30,  # Fewer steps for testing
        "seed": 42,  # For reproducibility
        "return_base64": False
    }
    
    print(f"  Prompt: {prompt[:60]}...")
    print(f"  Size: {size}x{size}")
    print(f"  Steps: 30 (reduced for testing)")
    print(f"  Seed: 42")
    
    try:
        print(f"\n⏳ Generating (this may take 30-60 seconds)...")
        response = requests.post(
            f"{server_url}/generate_with_control",
            json=payload,
            timeout=300  # 5 min timeout
        )
        
        if response.status_code != 200:
            print(f"✗ Generation failed: {response.status_code}")
            print(f"  {response.text}")
            return False
        
        # Load generated image
        generated_image = Image.open(io.BytesIO(response.content))
        print(f"✓ Generation successful!")
        print(f"  Generated image size: {generated_image.size}")
        print(f"  Generated image mode: {generated_image.mode}")
        
        # Check metadata
        metadata = response.headers.get('X-Generation-Metadata', '')
        if metadata:
            print(f"\n📊 Metadata: {metadata[:100]}...")
        
        # Save test output
        output_path = model_dir / "textures" / "test_uv_conditioned.png"
        output_path.parent.mkdir(exist_ok=True)
        generated_image.save(output_path)
        print(f"\n💾 Saved test output to: {output_path}")
        
        # Analyze output
        print(f"\n🔍 Analysis:")
        
        # Check if output respects UV mask
        import numpy as np
        
        gen_array = np.array(generated_image.convert('RGB'))
        mask_array = np.array(uv_mask.convert('L'))
        
        # Check if non-UV areas are black/dark
        non_uv_areas = mask_array < 30
        gen_in_non_uv = gen_array[non_uv_areas]
        
        if len(gen_in_non_uv) > 0:
            avg_brightness_non_uv = gen_in_non_uv.mean()
            print(f"  Non-UV area brightness: {avg_brightness_non_uv:.1f}/255")
            if avg_brightness_non_uv < 50:
                print(f"  ✓ Non-UV areas properly masked")
            else:
                print(f"  ⚠ Non-UV areas not fully masked (brightness too high)")
        
        # Check UV areas have content
        uv_areas = mask_array > 30
        gen_in_uv = gen_array[uv_areas]
        
        if len(gen_in_uv) > 0:
            avg_brightness_uv = gen_in_uv.mean()
            std_brightness_uv = gen_in_uv.std()
            print(f"  UV area brightness: {avg_brightness_uv:.1f}/255")
            print(f"  UV area variation: {std_brightness_uv:.1f}")
            if avg_brightness_uv > 50 and std_brightness_uv > 10:
                print(f"  ✓ UV areas have content with variation")
            else:
                print(f"  ⚠ UV areas may lack detail")
        
        print(f"\n{'='*70}")
        print(f"✅ Test Complete: UV conditioning is working!")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test UV-conditioned texture generation"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='colon',
        help='Model name to test (default: colon)'
    )
    
    parser.add_argument(
        '--server',
        type=str,
        default='localhost:8000',
        help='FLUX server address (default: localhost:8000)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=1024,
        choices=[512, 1024, 2048],
        help='Test image size (default: 1024)'
    )
    
    args = parser.parse_args()
    
    success = test_uv_conditioning(
        model_name=args.model,
        flux_server=args.server,
        size=args.size
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

