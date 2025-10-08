#!/usr/bin/env python3
"""
Setup script for remote Flux server configuration
"""

import os
import sys

def setup_flux_remote():
    """Setup remote Flux server configuration"""
    print("🚀 Flux Remote Server Setup")
    print("=" * 40)
    
    print("\nThis script will help you configure your remote Flux server for UV-guided texture generation.")
    print("\nPrerequisites:")
    print("1. Remote Flux server is running with updated code (includes /generate_with_control endpoint)")
    print("2. You have network access to the server")
    print("3. The server is running FLUX.1-dev model")
    
    print("\n" + "=" * 40)
    
    # Get server URL
    current_url = os.getenv('FLUX_SERVER_URL')
    if current_url:
        print(f"Current FLUX_SERVER_URL: {current_url}")
        use_current = input("Use current URL? (y/n): ").strip().lower()
        if use_current == 'y':
            flux_url = current_url
        else:
            flux_url = input("Enter your Flux server URL: ").strip()
    else:
        print("Examples:")
        print("  http://192.168.1.100:8000")
        print("  https://flux-server.example.com:8000")
        print("  http://your-server-ip:8000")
        print()
        flux_url = input("Enter your Flux server URL: ").strip()
    
    if not flux_url:
        print("❌ No URL provided. Exiting.")
        return 1
    
    # Validate URL format
    if not flux_url.startswith(('http://', 'https://')):
        flux_url = f"http://{flux_url}"
    
    print(f"\nUsing Flux server: {flux_url}")
    
    # Create environment setup commands
    print("\n📝 Environment Setup")
    print("=" * 40)
    print("\nAdd this to your shell profile (.bashrc, .zshrc, etc.):")
    print(f"export FLUX_SERVER_URL=\"{flux_url}\"")
    
    # Create a .env file for the current session
    env_content = f"""# Flux Remote Server Configuration
FLUX_SERVER_URL={flux_url}
"""
    
    env_file = ".env.flux"
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Created temporary config file: {env_file}")
    print("\nTo use this configuration for the current session, run:")
    print(f"source {env_file}")
    
    # Test connection
    print(f"\n🔬 Testing connection to {flux_url}...")
    
    try:
        import requests
        health_url = f"{flux_url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        
        print("✅ Connection successful!")
        
        # Check if it has the new endpoint
        try:
            root_url = flux_url.rstrip('/')
            response = requests.get(root_url, timeout=5)
            if 'generate_with_control' in response.text:
                print("✅ UV-guided generation endpoint detected!")
            else:
                print("⚠️  UV-guided generation endpoint not found.")
                print("   Please ensure your remote server has the updated code.")
        except:
            print("⚠️  Could not check for UV-guided endpoint.")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. Server URL is correct")
        print("2. Server is running")
        print("3. Network connectivity")
        return 1
    
    print(f"\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python test_flux_remote.py")
    print("2. Run: python generate_colon_flux_texture.py --size 1024 --overwrite")
    print("3. Use the Model Viewer with Flux UV-Guided generation")
    
    return 0

if __name__ == "__main__":
    sys.exit(setup_flux_remote())
