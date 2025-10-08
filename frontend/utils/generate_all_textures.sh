#!/bin/bash
# Generate textures for all medical organ models using FLUX.1

# Default settings
SIZE=${SIZE:-1024}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-3.5}
MODELS_DIR=${MODELS_DIR:-"output/models"}

echo "=================================="
echo "FLUX Texture Generation Batch Tool"
echo "=================================="
echo ""
echo "Settings:"
echo "  Size: ${SIZE}x${SIZE}"
echo "  Steps: ${STEPS}"
echo "  Guidance: ${GUIDANCE}"
echo "  Models Dir: ${MODELS_DIR}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
MODELS_PATH="${PROJECT_ROOT}/${MODELS_DIR}"

# Check if models directory exists
if [ ! -d "$MODELS_PATH" ]; then
    echo "✗ Error: Models directory not found: $MODELS_PATH"
    exit 1
fi

# Find all model directories
MODEL_NAMES=($(ls -d "${MODELS_PATH}"/*/ 2>/dev/null | xargs -n 1 basename))

if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
    echo "✗ Error: No model directories found in $MODELS_PATH"
    exit 1
fi

echo "Found ${#MODEL_NAMES[@]} models:"
for name in "${MODEL_NAMES[@]}"; do
    echo "  - $name"
done
echo ""

# Ask for confirmation
read -p "Generate textures for all models? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting texture generation..."
echo ""

# Counter for success/failure
SUCCESS=0
FAILED=0

# Generate texture for each model
for organ in "${MODEL_NAMES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Processing: $organ"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python3 "${SCRIPT_DIR}/generate_flux_texture.py" \
        --organ "$organ" \
        --models-dir "$MODELS_DIR" \
        --size "$SIZE" \
        --steps "$STEPS" \
        --guidance "$GUIDANCE"
    
    if [ $? -eq 0 ]; then
        ((SUCCESS++))
        echo "✓ Success: $organ"
    else
        ((FAILED++))
        echo "✗ Failed: $organ"
    fi
    
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary:"
echo "  Total: ${#MODEL_NAMES[@]}"
echo "  Success: $SUCCESS"
echo "  Failed: $FAILED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All textures generated successfully!"
    exit 0
else
    echo "⚠ Some textures failed to generate. Check the output above for details."
    exit 1
fi

