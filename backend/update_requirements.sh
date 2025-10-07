#!/bin/bash

###############################################################################
# Update Requirements Script
# Adds UV-guided generation dependencies to requirements.txt
###############################################################################

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📦 Updating Requirements for UV-Guided Generation"
echo "================================================="

# Backup original requirements.txt
if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements.txt.backup
    echo "✅ Backed up original requirements.txt"
fi

# Create updated requirements.txt with UV-guided generation dependencies
cat > requirements.txt << 'EOF'
# Core FLUX.1-dev dependencies
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0

# Web server dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
python-multipart>=0.0.6

# Image processing dependencies
Pillow>=9.5.0
numpy>=1.24.0
scipy>=1.10.0

# HTTP client for testing
requests>=2.31.0

# Environment management
python-dotenv>=1.0.0

# Optional: Better logging
colorlog>=6.7.0

# Optional: Performance monitoring
psutil>=5.9.0
EOF

echo "✅ Updated requirements.txt with UV-guided generation dependencies"
echo ""
echo "📋 Added packages:"
echo "   - numpy: For UV coordinate processing"
echo "   - scipy: For image filtering and morphological operations"
echo "   - Pillow: For image manipulation"
echo "   - requests: For HTTP client functionality"
echo ""
echo "🚀 To install the updated requirements:"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "📝 Original requirements.txt backed up as requirements.txt.backup"
