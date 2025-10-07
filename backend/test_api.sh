#!/bin/bash

###############################################################################
# FLUX.1-dev API Test Script
# Tests the Docker-deployed FLUX backend API
###############################################################################

set -e

# Configuration
HOST="${FLUX_HOST:-localhost}"
PORT="${FLUX_SERVER_PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"

echo "=========================================="
echo "FLUX.1-dev API Test"
echo "=========================================="
echo "Testing: $BASE_URL"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
echo "--------------------"
response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/health" || echo "000")
http_code=$(echo "$response" | tail -n 1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "200" ]; then
    echo "✓ Health check passed"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
else
    echo "✗ Health check failed (HTTP $http_code)"
    echo "$body"
    exit 1
fi

echo ""

# Test 2: Root Endpoint
echo "Test 2: Root Endpoint"
echo "---------------------"
response=$(curl -s "${BASE_URL}/")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
echo ""

# Test 3: Generate Simple Image
echo "Test 3: Generate Test Image"
echo "---------------------------"
echo "Prompt: 'A simple red circle on white background'"
echo "This will take ~30-60 seconds depending on GPU..."
echo ""

start_time=$(date +%s)

curl -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A simple red circle on white background",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20
  }' \
  --output test_output.png \
  -w "\nHTTP Status: %{http_code}\n" \
  -s

end_time=$(date +%s)
duration=$((end_time - start_time))

if [ -f "test_output.png" ]; then
    file_size=$(ls -lh test_output.png | awk '{print $5}')
    echo "✓ Image generated successfully"
    echo "  File: test_output.png"
    echo "  Size: $file_size"
    echo "  Time: ${duration}s"
    
    # Try to open the image (macOS)
    if command -v open &> /dev/null; then
        echo ""
        echo "Opening image..."
        open test_output.png
    fi
else
    echo "✗ Image generation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All Tests Passed!"
echo "=========================================="
echo ""
echo "API Endpoints:"
echo "  Health:  ${BASE_URL}/health"
echo "  Generate: ${BASE_URL}/generate"
echo "  Docs:     ${BASE_URL}/docs"
echo ""
echo "View API documentation in browser:"
echo "  open ${BASE_URL}/docs"
echo ""

