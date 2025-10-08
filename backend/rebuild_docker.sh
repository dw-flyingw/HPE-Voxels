#!/bin/bash
# Rebuild Docker container with updated UV conditioning dependencies

echo "=========================================="
echo "Rebuilding FLUX Backend Docker Container"
echo "=========================================="
echo ""

# Stop existing containers
echo "📦 Stopping existing containers..."
docker compose down

# Rebuild with no cache to ensure fresh build
echo ""
echo "🔨 Rebuilding container (this may take 5-10 minutes)..."
docker compose build --no-cache

# Start the service
echo ""
echo "🚀 Starting FLUX backend..."
docker compose up -d

# Wait a moment for startup
sleep 3

# Show logs
echo ""
echo "📋 Container logs:"
docker compose logs --tail=50

echo ""
echo "=========================================="
echo "✅ Rebuild complete!"
echo "=========================================="
echo ""
echo "Check status with: docker compose ps"
echo "View logs with:    docker compose logs -f"
echo "Test with:         curl http://localhost:8000/health"
echo ""

