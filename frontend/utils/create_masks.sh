#!/bin/bash
#
# create_masks.sh
# Convenience script to create UV masks for FLUX.1 texture generation
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Run the UV mask creation script
echo "Creating UV masks for FLUX.1 texture generation..."
echo ""

$PYTHON_CMD frontend/utils/create_uv_mask.py "$@"

echo ""
echo "Done! UV masks are ready for FLUX.1 texture generation."

