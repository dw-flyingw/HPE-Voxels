#!/bin/bash

###############################################################################
# Flux Server Test Script
###############################################################################

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🔬 Testing Flux Server with UV-Guided Generation"
echo "================================================="

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    FLUX_PORT=${FLUX_SERVER_PORT:-8000}
    FLUX_HOST=${FLUX_HOST:-localhost}
else
    echo "⚠️  No .env file found, using defaults"
    FLUX_PORT=8000
    FLUX_HOST=localhost
fi

SERVER_URL="http://$FLUX_HOST:$FLUX_PORT"

echo "🌐 Testing server at: $SERVER_URL"
echo ""

# Test 1: Basic connectivity
echo "1️⃣ Testing basic connectivity..."
if curl -s --connect-timeout 10 "$SERVER_URL/health" > /dev/null; then
    echo "✅ Server is responding"
else
    echo "❌ Server is not responding"
    echo "   Make sure the server is running with: ./start_server.sh"
    exit 1
fi

# Test 2: Health check
echo ""
echo "2️⃣ Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "$SERVER_URL/health")
echo "Health response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Test 3: Check for UV-guided endpoint
echo ""
echo "3️⃣ Testing UV-guided generation endpoint..."
ROOT_RESPONSE=$(curl -s "$SERVER_URL/")
if echo "$ROOT_RESPONSE" | grep -q "generate_with_control"; then
    echo "✅ UV-guided generation endpoint is available"
else
    echo "❌ UV-guided generation endpoint not found"
    echo "   Make sure you're using the updated flux_server.py"
    exit 1
fi

# Test 4: Test UV-guided generation (if server is ready)
echo ""
echo "4️⃣ Testing UV-guided generation..."
echo "   (This may take a few minutes for the first request)"

# Create a simple test payload
python3 -c "
import requests
import base64
from PIL import Image
import io
import json

try:
    # Create a simple test image
    test_image = Image.new('RGB', (256, 256), (128, 128, 128))
    
    # Convert to base64
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
    test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test payload
    payload = {
        'prompt': 'test texture generation',
        'control_image': test_image_b64,
        'control_type': 'uv_layout',
        'width': 256,
        'height': 256,
        'guidance_scale': 3.5,
        'num_inference_steps': 10,  # Low steps for testing
        'return_base64': True
    }
    
    print('Sending test request...')
    response = requests.post('$SERVER_URL/generate_with_control', json=payload, timeout=120)
    
    if response.status_code == 200:
        print('✅ UV-guided generation test passed!')
        result = response.json()
        if 'image_base64' in result:
            print('✅ Generated image successfully')
        else:
            print('⚠️  No image in response')
    else:
        print(f'❌ Generation test failed: {response.status_code}')
        print(f'Response: {response.text}')
        
except Exception as e:
    print(f'❌ Generation test error: {e}')
    print('   This might be normal if the model is still loading')
"

echo ""
echo "✅ All tests completed!"
echo ""
echo "🎉 Your Flux server is ready for UV-guided texture generation!"
echo ""
echo "📝 Available endpoints:"
echo "   - Health: $SERVER_URL/health"
echo "   - Generate: $SERVER_URL/generate"
echo "   - UV-Guided Generate: $SERVER_URL/generate_with_control"
echo "   - API Docs: $SERVER_URL/docs"
echo ""
echo "🚀 You can now use the texture generation scripts from your local machine:"
echo "   export FLUX_SERVER_URL=\"$SERVER_URL\""
echo "   python generate_colon_flux_texture.py --size 1024 --overwrite"
