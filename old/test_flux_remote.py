#!/usr/bin/env python3
"""
Test script for remote Flux server connectivity and UV-guided generation
"""

import os
import sys
import requests
import json
from PIL import Image
import io
import base64

def test_flux_server_connection(flux_url):
    """Test basic connectivity to Flux server"""
    print(f"Testing connection to Flux server: {flux_url}")
    
    try:
        # Test health endpoint
        health_url = f"{flux_url.rstrip('/')}/health"
        print(f"Checking health endpoint: {health_url}")
        
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Flux server is healthy!")
        print(f"   Status: {data.get('status', 'unknown')}")
        print(f"   Device: {data.get('device', 'unknown')}")
        print(f"   GPU: {data.get('gpu_name', 'unknown')}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_uv_guided_endpoint(flux_url):
    """Test if the UV-guided generation endpoint exists"""
    print(f"\nTesting UV-guided generation endpoint...")
    
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (256, 256), (128, 128, 128))
        
        # Convert to base64
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Test payload
        payload = {
            "prompt": "test prompt",
            "control_image": test_image_b64,
            "control_type": "uv_layout",
            "width": 256,
            "height": 256,
            "guidance_scale": 3.5,
            "num_inference_steps": 10,  # Low steps for testing
            "return_base64": True
        }
        
        endpoint_url = f"{flux_url.rstrip('/')}/generate_with_control"
        print(f"Testing endpoint: {endpoint_url}")
        
        response = requests.post(endpoint_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            print("✅ UV-guided generation endpoint is working!")
            return True
        else:
            print(f"❌ Endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("🔬 Flux Remote Server Test")
    print("=" * 50)
    
    # Get server URL from environment or user input
    flux_url = os.getenv('FLUX_SERVER_URL')
    
    if not flux_url:
        print("No FLUX_SERVER_URL environment variable found.")
        print("Please set it with your remote server URL:")
        print("export FLUX_SERVER_URL=http://your-server:8000")
        print()
        
        # Try to get from user input
        flux_url = input("Enter your Flux server URL (e.g., http://your-server:8000): ").strip()
        if not flux_url:
            print("❌ No URL provided. Exiting.")
            return 1
    
    print(f"Using Flux server: {flux_url}")
    
    # Test basic connectivity
    if not test_flux_server_connection(flux_url):
        print("\n❌ Cannot connect to Flux server. Please check:")
        print("   1. Server URL is correct")
        print("   2. Server is running")
        print("   3. Network connectivity")
        print("   4. Firewall settings")
        return 1
    
    # Test UV-guided endpoint
    if not test_uv_guided_endpoint(flux_url):
        print("\n❌ UV-guided generation endpoint not available.")
        print("Please ensure your Flux server has the updated code with:")
        print("   - /generate_with_control endpoint")
        print("   - UVGuidedGenerationRequest model")
        return 1
    
    print("\n✅ All tests passed! Your remote Flux server is ready for UV-guided texture generation.")
    print("\nYou can now run:")
    print("   python generate_colon_flux_texture.py --size 1024 --overwrite")
    print("   python generate_flux_uv_textures.py --organ colon --size 1024 --overwrite")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
